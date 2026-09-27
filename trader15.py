"""
15 分钟级缠论单仓模拟交易（只模拟，不下真实订单）。

规则（chan15_lab.py 控制变量测试得出的组合：区间套 + 大方向过滤 + 只做三类买卖点）：
  入场：15m 新确认的三买做多 / 三卖做空，按当前价成交（不计滑点），不加仓
  仓位：按信号 / 趋势强弱评分动态分档 0.5 / 1 / 2 / 3 / 5 倍（sizing.py；120 天回测比同等平均仓位多赚约 39%、回撤更小）
  过滤：① 大方向：1H*0.6 + 4H*0.4 趋势分不能明显相反（多单要求 > -1，空单要求 < +1）
        ② 区间套：1H 缠论方向（走势 + 近期买卖点）不能相反
        ③ 行情性质：只做趋势——1H 近 48 根 K 线趋势效率（净涨跌 / 逐根涨跌绝对值之和）≥ 25%，
           震荡行情里的突破不做（120 天回测：回撤 246U→138U，9 月由 −111U 转为 +37U）
  止损：买卖点价格外 0.5 ATR(15m)；用 1 分钟 K 线高低点逐根判定，按止损价原价成交
  离场：止损，或持仓中 15m 出现反向买卖点（按当时价格平仓）；不设固定止盈
  手续费：每笔平仓每 1 倍扣 2U（1 倍 = ETH 1 个 / BTC 0.03 个，5 倍即扣 10U）

用法：python trader15.py run | report
"""
import json
import logging
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import chan  # noqa: E402
import okx_client as ox  # noqa: E402
import sizing  # noqa: E402
import ta  # noqa: E402

TZ = timezone(timedelta(hours=8))
CFG_PATH = ROOT / "trader15_config.json"
DEFAULT_CFG = {
    "instruments": {"ETH-USDT-SWAP": {"unit": 1.0}, "BTC-USDT-SWAP": {"unit": 0.03}},
    "fee_per_unit": 2.0,
    "entry_points": ["三买", "三卖"],
    "trend_filter": True,
    "trend_threshold": 1.0,
    "nest_filter": True,
    "min_trend_eff_1h": 0.25,  # 1H 趋势效率下限，0 表示不过滤
    "sizing": "step",          # 动态仓位：step=按评分分档 0.5/1/2/3/5 倍，linear=0.5~5 倍线性，""=固定 1 倍（见 sizing.py）
    "stop_buffer_atr": 0.5,
    "fresh_bars": 3,            # 买卖点确认后多少根 15m 内还算“新”
    # 告警推送（可选，留空即关闭）：钉钉 / 企业微信群机器人 Webhook，或 Telegram 机器人
    "alert_webhook": "",
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "alert_after_errors": 3,    # 连续失败几次运行才告警
    "alert_trades": False,      # 开仓 / 平仓时是否也推送
}
LONG = {"一买", "二买", "三买", "盘背买"}
SHORT = {"一卖", "二卖", "三卖", "盘背卖"}
DB = ROOT / "trader15.db"
HEARTBEAT = ROOT / "heartbeat.json"
LOG = ROOT / "trader15.log"
log = logging.getLogger("t15")


def setup_logging():
    log.setLevel(logging.INFO)
    h = RotatingFileHandler(LOG, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S"))
    log.addHandler(h)
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        log.addHandler(logging.StreamHandler(sys.stdout))


def load_cfg():
    if not CFG_PATH.exists():
        CFG_PATH.write_text(json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**DEFAULT_CFG, **json.loads(CFG_PATH.read_text(encoding="utf-8"))}


def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.executescript("""
    create table if not exists trades(
        id integer primary key autoincrement, inst text, side text, point text, point_ts integer, point_px real,
        qty real, mult real, entry_ts integer, entry_px real, stop real,
        exit_ts integer, exit_px real, exit_reason text, gross real, fee real, net real,
        last_checked_ts integer, context text);
    create table if not exists used_points(inst text, type text, ts integer, primary key(inst, type, ts));
    create table if not exists signals(ts integer, inst text, price real, decision text, detail text);
    """)
    return con


def now_ms():
    return int(time.time() * 1000)


def bj(ts):
    return datetime.fromtimestamp(ts / 1000, TZ).strftime("%m-%d %H:%M") if ts else "-"


def signal(con, inst, price, decision, detail):
    con.execute("insert into signals values(?,?,?,?,?)", (now_ms(), inst, price, decision, detail)); con.commit()
    log.info(f"[{inst}] {decision}：{detail}")
    if decision in ("开仓", "平仓") and _CFG.get("alert_trades"):
        alert(_CFG, f"[{inst}] {decision}：{detail}")


_CFG = {}


def alert(cfg, text):
    """推送告警；失败只记日志，不影响交易逻辑。"""
    import requests
    text = f"【模拟交易】{text}"
    try:
        if cfg.get("alert_webhook"):   # 钉钉 / 企业微信群机器人通用格式
            requests.post(cfg["alert_webhook"], json={"msgtype": "text", "text": {"content": text}}, timeout=10)
        if cfg.get("telegram_bot_token") and cfg.get("telegram_chat_id"):
            requests.post(f"https://api.telegram.org/bot{cfg['telegram_bot_token']}/sendMessage",
                          json={"chat_id": cfg["telegram_chat_id"], "text": text}, timeout=10)
    except Exception as e:
        log.error(f"告警推送失败：{e}")


def read_heartbeat():
    try:
        return json.loads(HEARTBEAT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_heartbeat(ok, err=None):
    hb = read_heartbeat()
    now = now_ms()
    hb["last_run_ts"] = now
    if ok:
        hb["last_ok_ts"] = now
        hb["consecutive_errors"] = 0
    else:
        hb["consecutive_errors"] = hb.get("consecutive_errors", 0) + 1
        hb["last_error"] = str(err)[:300]
        hb["last_error_ts"] = now
    HEARTBEAT.write_text(json.dumps(hb, ensure_ascii=False, indent=1), encoding="utf-8")
    return hb


def fresh_points(df15, cfg):
    ch = chan.analyze(df15)
    n = len(df15)
    return [(p["type"], int(df15.ts.iloc[p["i"]]), float(p["price"])) for p in ch["points"] if p["i"] >= n - cfg["fresh_bars"]]


def candles_1m_since(inst, since_ms):
    rows = ox.get("/api/v5/market/candles", {"instId": inst, "bar": "1m", "limit": 300})
    while rows and int(rows[-1][0]) > since_ms and len(rows) < 3000:
        more = ox.get("/api/v5/market/history-candles", {"instId": inst, "bar": "1m", "limit": 100, "after": rows[-1][0]})
        if not more:
            break
        rows += more
    return sorted({(int(r[0]), float(r[2]), float(r[3])) for r in rows if r[8] == "1" and int(r[0]) >= since_ms})


def close_trade(con, cfg, t, px, ts_, reason):
    s = 1 if t["side"] == "long" else -1
    gross = (px - t["entry_px"]) * s * t["qty"]
    fee = cfg["fee_per_unit"] * t["mult"]
    con.execute("update trades set exit_ts=?, exit_px=?, exit_reason=?, gross=?, fee=?, net=?, last_checked_ts=? where id=?",
                (ts_, px, reason, gross, fee, gross - fee, ts_, t["id"])); con.commit()
    return gross - fee


def handle(con, cfg, inst, unit):
    price = float(ox.ticker(inst)["last"])
    df15 = ox.candles(inst, "15m", 400)
    fresh = fresh_points(df15, cfg)
    t = con.execute("select * from trades where inst=? and exit_ts is null", (inst,)).fetchone()
    if t:
        long = t["side"] == "long"
        last = t["last_checked_ts"]
        for ts_, h, l in candles_1m_since(inst, last):          # 1) 止损：1 分钟 K 线逐根判定
            if (l <= t["stop"]) if long else (h >= t["stop"]):
                net = close_trade(con, cfg, t, t["stop"], ts_ + 60_000, "止损")
                signal(con, inst, price, "平仓", f"#{t['id']} 止损 {t['entry_px']:.6g}→{t['stop']:.6g} 净{net:+.2f}U")
                return
            last = ts_ + 60_000
        con.execute("update trades set last_checked_ts=? where id=?", (last, t["id"])); con.commit()
        want = SHORT if long else LONG                            # 2) 反向买卖点离场
        opp = next(((tp, ts_) for tp, ts_, _ in fresh if tp in want and ts_ >= t["entry_ts"] - 45 * 60_000
                     and ts_ > t["point_ts"]), None)
        if opp:
            net = close_trade(con, cfg, t, price, now_ms(), f"反向{opp[0]}")
            signal(con, inst, price, "平仓", f"#{t['id']} 15m 出现{opp[0]}，按 {price:.6g} 离场 净{net:+.2f}U")
            return
        upnl = (price - t["entry_px"]) * (1 if long else -1) * t["qty"]
        signal(con, inst, price, "持仓", f"#{t['id']} {t['point']}{'多' if long else '空'} {t['mult']:g} 倍 @{t['entry_px']:.6g} 止损{t['stop']:.6g} 浮盈{upnl:+.2f}U")
        return

    cands = []
    for tp, ts_, px in reversed(fresh):
        if tp not in cfg["entry_points"]:
            continue
        if con.execute("select 1 from used_points where inst=? and type=? and ts=?", (inst, tp, ts_)).fetchone():
            continue
        con.execute("insert into used_points values(?,?,?)", (inst, tp, ts_)); con.commit()
        cands.append((tp, ts_, px))
    if not cands:
        signal(con, inst, price, "观望", "15m 没有新的三买 / 三卖")
        return
    df1h = ox.candles(inst, "1H", 400); df4h = ox.candles(inst, "4H", 300)
    r1, _, _ = ta.analyze_tf(df1h, "1H", 2.0); r4, _, _ = ta.analyze_tf(df4h, "4H", 2.0)
    trend = 0.6 * r1["trend_score"] + 0.4 * r4["trend_score"]
    ch1 = chan.analyze(df1h); bias = chan.bias(ch1)
    atr = float(ta.atr(df15).iloc[-1])
    c48 = df1h.c.values[-49:]
    eff = abs(c48[-1] - c48[0]) / (sum(abs(c48[i] - c48[i - 1]) for i in range(1, len(c48))) or 1)
    notes = []
    if eff < cfg["min_trend_eff_1h"]:
        signal(con, inst, price, "放弃", f"{'、'.join(c[0] for c in cands)}：1H 趋势效率 {eff:.0%} < {cfg['min_trend_eff_1h']:.0%}，"
                                        f"行情在震荡，不做震荡里的突破")
        return
    for tp, ts_, px in cands:
        side = 1 if tp in LONG else -1
        if cfg["trend_filter"] and trend * side <= -cfg["trend_threshold"]:
            notes.append(f"{tp}与大方向相反（趋势分 {trend:+.2f}）"); continue
        if cfg["nest_filter"] and bias * side < 0:
            notes.append(f"{tp}与 1H 缠论方向相反（1H 走势{ch1['trend']}，近期{ch1.get('recent') or '无'}）"); continue
        stop = px - side * cfg["stop_buffer_atr"] * atr
        if (stop - price) * side >= 0:
            notes.append(f"{tp}（{px:.6g}）已被价格打穿，结构失效"); continue
        ts_now = now_ms()
        adx4h = float(r4["adx"]["adx"])
        score = sizing.strength(trend, eff, ch1["trend"], adx4h, side)
        mult = sizing.size_from_score(score, cfg["sizing"]) if cfg.get("sizing") else 1.0
        qty = unit * mult
        ctx = {"trend": round(trend, 2), "chan1h_trend": ch1["trend"], "chan1h_bias": bias, "atr15": round(atr, 6),
               "trend_eff_1h": round(eff, 3), "adx4h": round(adx4h, 1), "score": round(score, 3)}
        cur = con.execute("""insert into trades(inst, side, point, point_ts, point_px, qty, mult, entry_ts, entry_px, stop,
                             last_checked_ts, context) values(?,?,?,?,?,?,?,?,?,?,?,?)""",
                          (inst, "long" if side > 0 else "short", tp, ts_, px, qty, mult, ts_now, price, stop,
                           ts_now // 60_000 * 60_000 + 60_000, json.dumps(ctx, ensure_ascii=False)))
        con.commit()
        signal(con, inst, price, "开仓", f"#{cur.lastrowid} {tp}{'做多' if side > 0 else '做空'} {mult:g} 倍（评分 {score:.2f}）@{price:.6g}"
                                        f"（买卖点 {px:.6g}），止损 {stop:.6g}，风险 {abs(price - stop) * qty:.2f}U；"
                                        f"大方向 {trend:+.2f}，1H 缠论{ch1['trend']}，1H 趋势效率 {eff:.0%}，4H ADX {adx4h:.0f}")
        return
    signal(con, inst, price, "放弃", "；".join(notes))


def run_once():
    cfg = load_cfg(); _CFG.update(cfg); con = db()
    prev = read_heartbeat()
    errors = []
    if not ox.reachable():
        # 断网：本次什么都不做；恢复后用 1 分钟 K 线从上次检查点回放，不会漏掉期间的止损
        errors.append("连不上 OKX（网络波动或被墙），本次跳过，恢复后自动补算")
        con.execute("insert into signals values(?,?,?,?,?)", (now_ms(), "-", None, "错误", errors[0])); con.commit()
        log.error(errors[0])
    else:
        for inst, ic in cfg["instruments"].items():
            try:
                handle(con, cfg, inst, ic["unit"])
            except Exception as e:
                errors.append(f"{inst}: {e}")
                log.error(f"[{inst}] 出错：{e}\n{traceback.format_exc()}")
                con.execute("insert into signals values(?,?,?,?,?)", (now_ms(), inst, None, "错误", str(e)[:300])); con.commit()
    hb = write_heartbeat(not errors, "；".join(errors))
    n = cfg.get("alert_after_errors", 3)
    if errors and hb["consecutive_errors"] == n:
        alert(cfg, f"已连续 {n} 次运行失败：{hb['last_error']}")
    if not errors and prev.get("consecutive_errors", 0) >= n:
        alert(cfg, f"已恢复正常（此前连续失败 {prev['consecutive_errors']} 次）")
    try:
        write_dashboard(con, cfg)
    except Exception as e:
        log.error(f"看板生成失败：{e}")
    con.close()


def stats(con, inst=None):
    rows = con.execute("select * from trades where exit_ts is not null" + (" and inst=?" if inst else "") + " order by exit_ts",
                       (inst,) if inst else ()).fetchall()
    w = [r for r in rows if r["net"] > 0]; l = [r for r in rows if r["net"] <= 0]
    eq = peak = mdd = 0.0; curve = []
    for r in rows:
        eq += r["net"]; peak = max(peak, eq); mdd = max(mdd, peak - eq); curve.append(eq)
    gw = sum(r["net"] for r in w); gl = -sum(r["net"] for r in l)
    return {"n": len(rows), "w": len(w), "l": len(l), "wr": len(w) / len(rows) * 100 if rows else 0.0, "net": eq,
            "gross": sum(r["gross"] for r in rows), "fees": sum(r["fee"] for r in rows),
            "avg_w": gw / len(w) if w else 0.0, "avg_l": -gl / len(l) if l else 0.0,
            "pf": gw / gl if gl else (float("inf") if gw else 0.0), "mdd": mdd, "curve": curve}


def report():
    con = db(); s = stats(con)
    print(f"已平仓 {s['n']} 笔 | 胜 {s['w']} 负 {s['l']} | 胜率 {s['wr']:.1f}% | 净利 {s['net']:+.2f}U（手续费 {s['fees']:.1f}U）"
          f" | 盈亏因子 {s['pf']:.2f} | 最大回撤 {s['mdd']:.2f}U")
    for t in con.execute("select * from trades order by entry_ts desc limit 30"):
        print(f"#{t['id']} {t['inst']} {t['side']} {t['point']} {bj(t['entry_ts'])} @{t['entry_px']:.6g} 止损{t['stop']:.6g} → "
              + (f"{bj(t['exit_ts'])} {t['exit_px']:.6g} {t['exit_reason']} 净{t['net']:+.2f}U" if t["exit_ts"] else "持仓中"))


def write_dashboard(con, cfg):
    s = stats(con); per = {i: stats(con, i) for i in cfg["instruments"]}
    opens = con.execute("select * from trades where exit_ts is null").fetchall()
    closed = con.execute("select * from trades where exit_ts is not null order by exit_ts desc limit 100").fetchall()
    sigs = con.execute("select * from signals order by ts desc limit 40").fetchall()
    first = con.execute("select min(ts) from signals").fetchone()[0]
    prices = {}
    for i in cfg["instruments"]:
        try:
            prices[i] = float(ox.ticker(i)["last"])
        except Exception:
            prices[i] = None
    money = lambda v: f'<span class="{"up" if v > 0 else "down" if v < 0 else ""}">{v:+.2f}</span>'
    pts = [0.0] + s["curve"]
    if len(pts) > 1:
        lo_, hi_ = min(pts), max(pts); sp = (hi_ - lo_) or 1; W, Hh = 600, 160
        path = " ".join(f"{'M' if k == 0 else 'L'}{k / (len(pts) - 1) * W:.1f},{Hh - (y - lo_) / sp * Hh:.1f}" for k, y in enumerate(pts))
        z = Hh - (0 - lo_) / sp * Hh
        svg = (f'<svg viewBox="0 -6 {W} {Hh + 12}" preserveAspectRatio="none" class="curve"><line x1="0" x2="{W}" y1="{z:.1f}" y2="{z:.1f}" '
               f'class="zero"/><path d="{path}" class="{"lu" if s["net"] >= 0 else "ld"}"/></svg>')
    else:
        svg = '<p class="muted">还没有平仓记录，第一笔平仓后显示权益曲线。</p>'
    def orow(t):
        p = prices.get(t["inst"]); s_ = 1 if t["side"] == "long" else -1
        up = f"{money((p - t['entry_px']) * s_ * t['qty'])}" if p else "-"
        return (f"<tr><td>{t['inst'].split('-')[0]}</td><td class='{'up' if s_ > 0 else 'down'}'>{t['point']}{'多' if s_ > 0 else '空'}</td>"
                f"<td>{t['mult']:g}×</td><td>{json.loads(t['context'] or '{}').get('score', '-')}</td>"
                f"<td>{t['entry_px']:.6g}</td><td>{p and f'{p:.6g}'}</td><td class='down'>{t['stop']:.6g}</td><td>{up}</td><td>{bj(t['entry_ts'])}</td></tr>")
    open_rows = "".join(orow(t) for t in opens) or '<tr><td colspan="9" class="muted">无持仓</td></tr>'
    closed_rows = "".join(
        f"<tr><td>{t['id']}</td><td>{t['inst'].split('-')[0]}</td><td class='{'up' if t['side'] == 'long' else 'down'}'>{t['point']}</td>"
        f"<td>{bj(t['entry_ts'])}</td><td>{bj(t['exit_ts'])}</td><td>{t['entry_px']:.6g}</td><td>{t['exit_px']:.6g}</td>"
        f"<td>{t['exit_reason']}</td><td>{t['mult']:g}×</td><td>{money(t['net'])}</td></tr>" for t in closed) or '<tr><td colspan="10" class="muted">暂无</td></tr>'
    per_rows = "".join(f"<tr><td>{i}</td><td>{cfg['instruments'][i]['unit']}</td><td>{p['n']}</td><td>{p['wr']:.1f}%</td>"
                       f"<td>{money(p['net'])}</td><td>{p['fees']:.0f}</td><td>{p['mdd']:.1f}</td></tr>" for i, p in per.items())
    sig_rows = "".join(f"<tr><td>{bj(g['ts'])}</td><td>{g['inst'].split('-')[0]}</td><td>{g['decision']}</td><td class='d'>{g['detail']}</td></tr>" for g in sigs)
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    hb = read_heartbeat()
    ce = hb.get("consecutive_errors", 0)
    # 服务端写入生成时间，页面里的脚本用浏览器当前时间判断是否停更（程序完全停止时也能发现）
    health = (f'<div id="health" class="{"bad" if ce else "ok"}" data-ts="{now_ms()}">'
              f'{"⚠️ 最近连续 " + str(ce) + " 次运行失败：" + hb.get("last_error", "") if ce else "● 运行正常"}'
              f'｜最后成功运行 {bj(hb.get("last_ok_ts"))}</div>'
              '<script>(function(){var e=document.getElementById("health");var m=(Date.now()-(+e.dataset.ts))/60000;'
              'if(m>15){e.className="bad";e.textContent="⚠️ 看板已 "+Math.round(m)+" 分钟没有更新，程序可能已停止（正常每 5 分钟更新一次）";}})();</script>')
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>缠论模拟交易看板</title>
<style>
:root{{--bg:#0b0e11;--card:#161a1f;--line:#2a3038;--text:#e8eaed;--sub:#8b929c;--up:#16c784;--down:#ea3943}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,"Microsoft YaHei",sans-serif;padding:16px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:0 0 8px}} .muted{{color:var(--sub)}} .up{{color:var(--up)}} .down{{color:var(--down)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:14px 0}}
.kpi,.card{{background:var(--card);border-radius:12px;padding:12px 14px}} .card{{margin-bottom:12px;overflow-x:auto}}
.kpi b{{display:block;font-size:22px;font-variant-numeric:tabular-nums}} .kpi span{{color:var(--sub);font-size:12px}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;white-space:nowrap}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}} th{{color:var(--sub);font-weight:500;font-size:12px}}
td.d{{white-space:normal;min-width:260px;color:var(--sub)}}
#health{{margin:6px 0;padding:6px 10px;border-radius:8px;font-size:13px}} .ok{{background:#10281f;color:var(--up)}} .bad{{background:#3a1518;color:#ff8a8a}}
.curve{{width:100%;height:170px}} .curve path{{fill:none;stroke-width:2}} .lu{{stroke:var(--up)}} .ld{{stroke:var(--down)}} .zero{{stroke:var(--line);stroke-dasharray:4 4}}
</style></head><body>
<h1>15 分钟级缠论模拟交易</h1>
{health}
<div class="muted">只模拟，不下真实单｜15m 三买 / 三卖入场，区间套（1H 缠论）+ 大方向 + 只做趋势（1H 趋势效率≥{cfg['min_trend_eff_1h']:.0%}）｜按强弱动态 0.5~5 倍｜止损在买卖点外 0.5 ATR，反向买卖点离场｜
每笔每 1 倍扣 {cfg['fee_per_unit']}U，不计滑点｜每 5 分钟运行｜开始于 {bj(first)}｜更新于 {datetime.now(TZ):%m-%d %H:%M:%S}</div>
<div class="grid">
<div class="kpi"><span>胜率</span><b>{s['wr']:.1f}%</b><span>{s['w']} 胜 / {s['l']} 负</span></div>
<div class="kpi"><span>净利润</span><b>{money(s['net'])} U</b><span>毛利 {s['gross']:+.2f}</span></div>
<div class="kpi"><span>累计手续费</span><b>{s['fees']:.1f} U</b><span>{s['n']} 笔已平仓</span></div>
<div class="kpi"><span>盈亏因子</span><b>{pf}</b><span>均盈 {s['avg_w']:+.2f} / 均亏 {s['avg_l']:+.2f}</span></div>
<div class="kpi"><span>最大回撤</span><b>{s['mdd']:.2f} U</b><span>按已平仓计</span></div>
</div>
<div class="card"><h2>权益曲线（净利润累计）</h2>{svg}</div>
<div class="card"><h2>当前持仓</h2><table><tr><th>合约</th><th>方向</th><th>倍数</th><th>评分</th><th>入场</th><th>现价</th><th>止损</th><th>浮盈U</th><th>开仓</th></tr>{open_rows}</table></div>
<div class="card"><h2>分合约</h2><table><tr><th>合约</th><th>1倍数量</th><th>笔数</th><th>胜率</th><th>净利U</th><th>手续费U</th><th>最大回撤U</th></tr>{per_rows}</table></div>
<div class="card"><h2>已平仓（最近 100 笔）</h2><table><tr><th>#</th><th>合约</th><th>买卖点</th><th>开仓</th><th>平仓</th><th>入场</th><th>出场</th><th>结果</th><th>倍数</th><th>净利U</th></tr>{closed_rows}</table></div>
<div class="card"><h2>最近分析记录</h2><table><tr><th>时间</th><th>合约</th><th>决策</th><th>说明</th></tr>{sig_rows}</table></div>
</body></html>"""
    (ROOT / "web").mkdir(exist_ok=True)
    (ROOT / "web" / "index.html").write_text(html, encoding="utf-8")


LOCK = ROOT / "trader15.lock"


def run_locked():
    """同一时间只允许一个实例（定时器和看门狗可能同时触发）；超过 5 分钟的锁视为上次异常退出留下的，直接清掉。"""
    import os
    try:
        if LOCK.exists() and time.time() - LOCK.stat().st_mtime > 300:
            LOCK.unlink()
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log.info("已有实例在运行，本次跳过")
        return
    try:
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        run_once()
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    setup_logging()
    {"run": run_locked, "report": report}.get(sys.argv[1] if len(sys.argv) > 1 else "run", run_locked)()
