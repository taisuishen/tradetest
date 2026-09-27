"""
小时级别模拟交易：马丁式分批挂单（只模拟，不下真实订单）。

方向：4H（0.6）+ 1D（0.4）加权趋势分，只做顺势（《方法论》：短趋势服从长趋势）
位置：1H / 4H / 1D 结构聚类（前高低点、EMA55、维加斯、布林、一目、斐波那契、成交密集区、挂单墙、整数关口）
挂单：顺势方向上选 5 个不同的结构支撑（做空为阻力），按 0.2 / 0.4 / 0.8 / 1.6 / 3.2 倍挂限价单
止损：全部层共用一个止损，放在最深一层结构外 0.5 ATR(1H)；触发后全部平仓
止盈：均价 + tp_atr × ATR(1H)，但不超过均价上方最近的强阻力；每成交一层按新均价重算
更新：一层都没成交时，每次运行按最新结构重挂；一旦有成交，挂单与止损固定
成交：用 1 分钟 K 线高低点逐根回放，挂单价 / 止损价 / 止盈价原价成交（不计滑点）；
      同一根 K 线里先处理成交、再判止损，新成交的那根不判止盈（保守）
手续费：平仓时按总仓位扣，每 1 倍（ETH 1 个 / BTC 0.03 个）2U

注意：《方法论》第 25 讲“赔钱的头寸绝不加码或摊平”，马丁加仓与之相悖。
      这种打法胜率通常很高，但一次打穿全部层就是大亏，务必同时看净利润和最大回撤。

用法：python paper_trader.py run | report
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

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import okx_client as ox  # noqa: E402
import chan  # noqa: E402
import ta  # noqa: E402
from okx_analyzer import orderbook_analysis, volume_profile  # noqa: E402

TZ = timezone(timedelta(hours=8))
CFG_PATH = ROOT / "paper_config.json"
DEFAULT_CFG = {
    "instruments": {"ETH-USDT-SWAP": {"unit": 1.0}, "BTC-USDT-SWAP": {"unit": 0.03}},
    "layer_multipliers": [0.2, 0.4, 0.8, 1.6, 3.2],
    "fee_per_unit": 2.0,          # 每 1 倍仓位平仓扣 2U
    "trend_threshold": 1.0,       # 4H*0.6 + 1D*0.4 的趋势分阈值（-3~+3）
    "min_level_strength": 3.0,    # 挂单位置的最低重合强度
    "min_layer_gap_atr": 0.6,     # 相邻两层至少间隔多少 ATR(1H)
    "max_ladder_depth_atr": 10.0, # 最深一层离现价不超过多少 ATR(1H)
    "min_layers": 3,              # 结构支撑少于几层就不挂
    "stop_buffer_atr": 0.5,
    "tp_atr": 2.0,
    "cooldown_after_loss_min": 120,
    "cooldown_after_win_min": 30,
    "use_chan": True,             # 缠论：1H 走势/买卖点过滤方向、中枢与笔端点加入结构位、三类买卖点提前离场
}
DB = ROOT / "paper_trades.db"
LOG = ROOT / "paper_trader.log"
log = logging.getLogger("paper")


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
    create table if not exists campaigns(
        id integer primary key autoincrement,
        inst text, side text, status text,           -- pending / active / closed / cancelled
        created_ts integer, updated_ts integer,
        layers text,                                 -- json: [{px, mult, qty, filled_ts, sources}]
        stop real, tp real, atr real, unit real, obstacles text,
        avg_px real, qty real, max_layer integer,
        exit_ts integer, exit_px real, exit_reason text,
        gross real, fee real, net real,
        last_checked_ts integer, context text);
    create table if not exists signals(ts integer, inst text, price real, decision text, detail text);
    """)
    return con


def now_ms():
    return int(time.time() * 1000)


def bj(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, TZ).strftime("%m-%d %H:%M") if ts_ms else "-"


# ---------------- 小时级关键位 ----------------
def levels_1h(price, r1h, piv1h, r4h, piv4h, r1d, piv1d, df1h, walls, extra=()):
    L = []
    def add(p, src, w):
        if p and np.isfinite(p) and abs(p / price - 1) < 0.15:
            L.append((float(p), src, w))
    for piv, tf, w in ((piv1h, "1H", 1.5), (piv4h, "4H", 2.5), (piv1d, "1D", 3.0)):
        for i, p, t in piv[-10:]:
            add(p, f"{tf}{'前高' if t == 'H' else '前低'}", w)
    for r, tf, w in ((r1h, "1H", 1.0), (r4h, "4H", 1.8), (r1d, "1D", 2.2)):
        add(r["ema_group"]["ema55"], f"{tf}EMA55", w)
        add(r["vegas"]["ema144"], f"{tf}维加斯144", w)
        add(r["vegas"]["ema169"], f"{tf}维加斯169", w)
        add(r["boll"]["upper"], f"{tf}布林上轨", w * 0.6)
        add(r["boll"]["lower"], f"{tf}布林下轨", w * 0.6)
        add(r["boll"]["mid"], f"{tf}布林中轨", w * 0.5)
        add(r["ichimoku"]["kijun"], f"{tf}基准线", w * 0.7)
        add(r["ichimoku"]["cloud_top"], f"{tf}云上沿", w * 0.6)
        add(r["ichimoku"]["cloud_bottom"], f"{tf}云下沿", w * 0.6)
        fb = r.get("fib")
        if fb:
            for k in ("0.382", "0.5", "0.618"):
                add(fb["retracement"][k], f"{tf}斐波{k}", w)
            add(fb["retracement"]["0.764"], f"{tf}斐波0.764", w / 2)
    vp = volume_profile(df1h.iloc[-336:], price)  # 近 14 天
    add(vp["poc"], "成交最密集价", 2.5)
    for p in vp["hvn"]:
        add(p, "成交密集区", 1.2)
    for w_ in walls:
        add(w_["price"], f"{'买' if '买' in w_['side'] else '卖'}单墙{w_['size_usd'] / 1e6:.1f}M", 1.2)
    for p, src, w in extra:
        add(p, src, w)
    mag = 10 ** int(np.floor(np.log10(price)))
    for m in (mag / 2, mag / 10):
        for k in range(-6, 7):
            add((np.floor(price / m) + k) * m, "整数关口", 1.2 if m == mag / 2 else 0.6)
    a = r1h["atr"]
    tol, span = 0.3 * a, 0.8 * a
    groups, cur = [], []
    for x in sorted(L):
        if cur and (x[0] - cur[-1][0] > tol or x[0] - cur[0][0] > span):
            groups.append(cur); cur = []
        cur.append(x)
    if cur:
        groups.append(cur)
    out = []
    for c in groups:
        ws = sum(w for _, _, w in c)
        out.append({"price": sum(p * w for p, _, w in c) / ws, "lo": c[0][0], "hi": c[-1][0], "strength": ws,
                    "sources": sorted({s for _, s, _ in c})})
    return out


def fetch(inst):
    d = {"df1h": ox.candles(inst, "1H", 400), "df4h": ox.candles(inst, "4H", 300), "df1d": ox.candles(inst, "1D", 300)}
    d["price"] = float(ox.ticker(inst)["last"])
    fh = ox.funding_history(inst, 90)
    rates = np.array([float(x["realizedRate"] or x["fundingRate"]) for x in fh])
    d["frank"] = float((rates < float(ox.funding(inst)["fundingRate"])).mean() * 100)
    ct_val = float(ox.instrument(inst)["ctVal"])
    d["walls"] = orderbook_analysis(inst, ct_val, d["price"], 2, 2)["persistent_walls"]
    return d


def plan(d, cfg, unit):
    """纯函数：根据当前结构给出分批挂单计划。返回 (计划 或 None, 说明, 上下文)。"""
    price = d["price"]
    r1h, piv1h, _ = ta.analyze_tf(d["df1h"], "1H", 2.0)
    r4h, piv4h, _ = ta.analyze_tf(d["df4h"], "4H", 2.0)
    r1d, piv1d, _ = ta.analyze_tf(d["df1d"], "1D", 1.5)
    a = r1h["atr"]
    trend = 0.6 * r4h["trend_score"] + 0.4 * r1d["trend_score"]
    th = cfg["trend_threshold"]
    side = "long" if trend >= th else ("short" if trend <= -th else None)
    ctx = {"trend": round(trend, 2), "t1h": r1h["trend_score"], "t4h": r4h["trend_score"], "t1d": r1d["trend_score"],
           "regime4h": r4h["regime"], "atr1h": round(a, 6), "funding_rank": round(d["frank"])}
    if side is None:
        return None, f"方向不明（4H {r4h['trend_score']:+.1f} / 1D {r1d['trend_score']:+.1f}，加权 {trend:+.2f}），不挂单", ctx
    if side == "long" and d["frank"] > 95:
        return None, "资金费率处于30天极高位，多头拥挤，不挂多单", ctx
    if side == "short" and d["frank"] < 5:
        return None, "资金费率处于30天极低位，空头拥挤，不挂空单", ctx
    if side == "long" and any("顶背离（已确认" in x for x in r4h["divergences"]):
        return None, "4H 已确认顶背离，不挂多单", ctx
    if side == "short" and any("底背离（已确认" in x for x in r4h["divergences"]):
        return None, "4H 已确认底背离，不挂空单", ctx

    extra = []
    if cfg.get("use_chan"):
        ch1 = chan.analyze(d["df1h"]); ch4 = chan.analyze(d["df4h"], recent_bars=6)
        cb = chan.bias(ch1)
        ctx.update({"chan_trend_1h": ch1["trend"], "chan_recent_1h": ch1.get("recent", []), "chan_bias": cb})
        if side == "long" and cb < 0:
            return None, f"缠论 1H 偏空（走势{ch1['trend']}，近期{ch1.get('recent') or '无买卖点'}），不挂多单", ctx
        if side == "short" and cb > 0:
            return None, f"缠论 1H 偏多（走势{ch1['trend']}，近期{ch1.get('recent') or '无买卖点'}），不挂空单", ctx
        for ch, tf, w in ((ch1, "1H", 2.5), (ch4, "4H", 3.5)):
            for zg, zd in ch.get("zs_levels", []):
                extra += [(zg, f"{tf}中枢上沿", w), (zd, f"{tf}中枢下沿", w)]
        extra += [(p_, "1H笔端点", 1.2) for p_ in ch1["stroke_points"]]
    lv = levels_1h(price, r1h, piv1h, r4h, piv4h, r1d, piv1d, d["df1h"], d["walls"], extra)
    long = side == "long"
    # 多单挂在下方支撑、空单挂在上方阻力，由近到远，相邻两层间隔 ≥ min_gap
    cands = [x for x in lv if x["strength"] >= cfg["min_level_strength"] and
             ((x["price"] < price - 0.2 * a) if long else (x["price"] > price + 0.2 * a)) and
             abs(x["price"] - price) <= cfg["max_ladder_depth_atr"] * a]
    cands.sort(key=lambda x: -x["price"] if long else x["price"])
    picked = []
    for x in cands:
        if not picked or abs(x["price"] - picked[-1]["price"]) >= cfg["min_layer_gap_atr"] * a:
            picked.append(x)
        if len(picked) == len(cfg["layer_multipliers"]):
            break
    if len(picked) < cfg["min_layers"]:
        return None, f"方向{'多' if long else '空'}，但 {cfg['max_ladder_depth_atr']:.0f} ATR 内只有 {len(picked)} 个合格结构位，不挂", ctx
    layers = [{"px": x["price"], "mult": m, "qty": m * unit, "filled_ts": None, "sources": x["sources"][:5],
               "strength": round(x["strength"], 1)} for x, m in zip(picked, cfg["layer_multipliers"])]
    deep = picked[len(layers) - 1]
    stop = deep["lo"] - cfg["stop_buffer_atr"] * a if long else deep["hi"] + cfg["stop_buffer_atr"] * a
    obstacles = sorted([x["lo"] if long else x["hi"] for x in lv if x["strength"] >= cfg["min_level_strength"] and
                        ((x["lo"] > layers[-1]["px"]) if long else (x["hi"] < layers[-1]["px"]))])
    max_loss = sum(abs(L_["px"] - stop) * L_["qty"] for L_ in layers) + cfg["fee_per_unit"] * sum(L_["mult"] for L_ in layers)
    ctx.update({"max_loss_if_all_filled": round(max_loss, 2), "levels_used": len(layers)})
    return {"side": side, "layers": layers, "stop": stop, "atr": a, "obstacles": obstacles}, \
        (f"{'多' if long else '空'}单分 {len(layers)} 层挂在 " + " / ".join(f"{L_['px']:.6g}×{L_['mult']}" for L_ in layers) +
         f"，统一止损 {stop:.6g}，全成交后打止损最多亏 {max_loss:.1f}U"), ctx


def take_profit(c, cfg, avg):
    """按当前均价重算止盈：均价 + tp_atr×ATR，但不越过最近的强阻力，且至少覆盖手续费 1.5 倍。"""
    a = c["atr"]; long = c["side"] == "long"
    obst = json.loads(c["obstacles"])
    fee_per_coin = cfg["fee_per_unit"] / c["unit"]
    base = avg + cfg["tp_atr"] * a if long else avg - cfg["tp_atr"] * a
    near = [o for o in obst if (o > avg + 0.3 * a if long else o < avg - 0.3 * a)]
    if near:
        o = min(near) if long else max(near)
        base = min(base, o - 0.1 * a) if long else max(base, o + 0.1 * a)
    min_dist = max(0.5 * a, 1.5 * fee_per_coin)
    return max(base, avg + min_dist) if long else min(base, avg - min_dist)


# ---------------- K 线回放 ----------------
def candles_1m_since(inst, since_ms):
    rows = ox.get("/api/v5/market/candles", {"instId": inst, "bar": "1m", "limit": 300})
    while rows and int(rows[-1][0]) > since_ms and len(rows) < 3000:
        more = ox.get("/api/v5/market/history-candles", {"instId": inst, "bar": "1m", "limit": 100, "after": rows[-1][0]})
        if not more:
            break
        rows += more
    return sorted({(int(r[0]), float(r[2]), float(r[3])) for r in rows if r[8] == "1" and int(r[0]) >= since_ms})


def replay(c, cfg, bars, bar_ms=60_000):
    """逐根处理：挂单成交 → 止损 → 止盈。bars 为 (开盘时间, 最高, 最低)。"""
    layers = json.loads(c["layers"]); long = c["side"] == "long"
    stop, tp = c["stop"], c["tp"]
    last = c["last_checked_ts"]
    res = {"closed": False}
    for ts_, h, l in bars:
        filled_now = False
        for L_ in layers:
            if L_["filled_ts"] is None and ((l <= L_["px"]) if long else (h >= L_["px"])):
                L_["filled_ts"] = ts_; filled_now = True
        last = ts_ + bar_ms
        f = [L_ for L_ in layers if L_["filled_ts"]]
        qty = sum(L_["qty"] for L_ in f)
        if not qty:
            continue
        avg = sum(L_["px"] * L_["qty"] for L_ in f) / qty
        if filled_now:
            tp = take_profit(c, cfg, avg)
        hit_sl = (l <= stop) if long else (h >= stop)
        hit_tp = not filled_now and ((h >= tp) if long else (l <= tp))
        if hit_sl or hit_tp:
            px = stop if hit_sl else tp
            gross = (px - avg) * qty * (1 if long else -1)
            fee = cfg["fee_per_unit"] * sum(L_["mult"] for L_ in f)
            res.update(closed=True, exit_ts=last, exit_px=px, exit_reason="止损" if hit_sl else "止盈",
                       gross=gross, fee=fee, net=gross - fee)
            break
    f = [L_ for L_ in layers if L_["filled_ts"]]
    qty = sum(L_["qty"] for L_ in f)
    res.update(layers=layers, tp=tp, qty=qty, avg_px=(sum(L_["px"] * L_["qty"] for L_ in f) / qty) if qty else None,
               max_layer=len(f), last_checked_ts=last)
    return res


def chan_exit(side, df1h, since_ms):
    """持仓后 1H 出现反向三类买卖点（多单遇三卖、空单遇三买）→ 提前离场。返回触发说明或 None。"""
    ch = chan.analyze(df1h)
    want = "三卖" if side == "long" else "三买"
    for p in ch["points"]:
        if p["type"] == want and int(df1h.ts.iloc[p["i"]]) >= since_ms:
            return f"缠论1H{want}（{p['price']:.6g}）"
    return None


def close_at(c, cfg, layers, px, ts_, reason):
    f = [L_ for L_ in layers if L_["filled_ts"]]
    qty = sum(L_["qty"] for L_ in f); avg = sum(L_["px"] * L_["qty"] for L_ in f) / qty
    gross = (px - avg) * qty * (1 if c["side"] == "long" else -1)
    fee = cfg["fee_per_unit"] * sum(L_["mult"] for L_ in f)
    return {"closed": True, "exit_ts": ts_, "exit_px": px, "exit_reason": reason, "gross": gross, "fee": fee,
            "net": gross - fee, "layers": layers, "tp": c["tp"], "qty": qty, "avg_px": avg, "max_layer": len(f),
            "last_checked_ts": ts_}


def save_replay(con, c, r):
    status = "closed" if r["closed"] else ("active" if r["qty"] else "pending")
    con.execute("""update campaigns set status=?, updated_ts=?, layers=?, tp=?, avg_px=?, qty=?, max_layer=?,
                   last_checked_ts=?, exit_ts=?, exit_px=?, exit_reason=?, gross=?, fee=?, net=? where id=?""",
                (status, now_ms(), json.dumps(r["layers"], ensure_ascii=False), r["tp"], r["avg_px"], r["qty"], r["max_layer"],
                 r["last_checked_ts"], r.get("exit_ts"), r.get("exit_px"), r.get("exit_reason"),
                 r.get("gross"), r.get("fee"), r.get("net"), c["id"]))
    con.commit()
    return status


def in_cooldown(con, cfg, inst):
    t = con.execute("select exit_ts, net from campaigns where inst=? and status='closed' order by exit_ts desc limit 1", (inst,)).fetchone()
    if not t:
        return None
    mins = cfg["cooldown_after_loss_min"] if t["net"] <= 0 else cfg["cooldown_after_win_min"]
    left = (t["exit_ts"] + mins * 60_000 - now_ms()) / 60_000
    return f"上一轮{'亏损' if t['net'] <= 0 else '盈利'}平仓后冷却中，还剩 {left:.0f} 分钟" if left > 0 else None


def signal(con, inst, price, decision, detail):
    con.execute("insert into signals values(?,?,?,?,?)", (now_ms(), inst, price, decision, detail))
    con.commit()
    log.info(f"[{inst}] {decision}：{detail}")


def run_once():
    cfg = load_cfg()
    con = db()
    for inst, ic in cfg["instruments"].items():
        unit = ic["unit"]
        try:
            c = con.execute("select * from campaigns where inst=? and status in ('pending','active')", (inst,)).fetchone()
            price = float(ox.ticker(inst)["last"])
            if c:
                r = replay(c, cfg, candles_1m_since(inst, c["last_checked_ts"]))
                status = save_replay(con, c, r)
                if status == "closed":
                    signal(con, inst, price, "平仓", f"#{c['id']} {r['exit_reason']} 成交{r['max_layer']}层 均价{r['avg_px']:.6g}→{r['exit_px']:.6g} "
                                                    f"毛利{r['gross']:+.2f} 手续费{r['fee']:.1f} 净{r['net']:+.2f}U")
                elif status == "active" and cfg.get("use_chan"):
                    first_fill = min(L_["filled_ts"] for L_ in r["layers"] if L_["filled_ts"])
                    why_exit = chan_exit(c["side"], ox.candles(inst, "1H", 400), first_fill)
                    if why_exit:
                        c2 = con.execute("select * from campaigns where id=?", (c["id"],)).fetchone()
                        r2 = close_at(c2, cfg, r["layers"], price, now_ms(), "缠论离场")
                        save_replay(con, c2, r2)
                        signal(con, inst, price, "平仓", f"#{c['id']} {why_exit}，提前离场 成交{r2['max_layer']}层 均价{r2['avg_px']:.6g}→{price:.6g} 净{r2['net']:+.2f}U")
                        continue
                if status == "active":
                    upnl = (price - r["avg_px"]) * r["qty"] * (1 if c["side"] == "long" else -1)
                    signal(con, inst, price, "持仓", f"#{c['id']} 已成交{r['max_layer']}层 均价{r['avg_px']:.6g} 数量{r['qty']:.4g} "
                                                    f"止盈{r['tp']:.6g} 止损{c['stop']:.6g} 浮盈{upnl:+.2f}U")
                    continue
                elif status == "pending":  # 一层都没成交：按最新结构重挂或撤单
                    p, why, ctx = plan(fetch(inst), cfg, unit)
                    if p is None or p["side"] != c["side"]:
                        con.execute("update campaigns set status='cancelled', updated_ts=? where id=?", (now_ms(), c["id"])); con.commit()
                        signal(con, inst, price, "撤单", f"#{c['id']} 挂单未成交，条件变化：{why}")
                        if p is None:
                            continue
                    else:
                        con.execute("update campaigns set layers=?, stop=?, atr=?, obstacles=?, updated_ts=?, context=? where id=?",
                                    (json.dumps(p["layers"], ensure_ascii=False), p["stop"], p["atr"], json.dumps(p["obstacles"]),
                                     now_ms(), json.dumps(ctx, ensure_ascii=False), c["id"])); con.commit()
                        signal(con, inst, price, "挂单", f"#{c['id']} 更新：{why}")
                        continue
            cd = in_cooldown(con, cfg, inst)
            if cd:
                signal(con, inst, price, "冷却", cd)
                continue
            p, why, ctx = plan(fetch(inst), cfg, unit)
            if p is None:
                signal(con, inst, price, "观望", why)
                continue
            ts_ = now_ms()
            cur = con.execute("""insert into campaigns(inst, side, status, created_ts, updated_ts, layers, stop, tp, atr, unit,
                                 obstacles, qty, max_layer, last_checked_ts, context) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                              (inst, p["side"], "pending", ts_, ts_, json.dumps(p["layers"], ensure_ascii=False), p["stop"], None,
                               p["atr"], unit, json.dumps(p["obstacles"]), 0, 0, ts_ // 60_000 * 60_000 + 60_000,
                               json.dumps(ctx, ensure_ascii=False)))
            con.commit()
            signal(con, inst, price, "挂单", f"#{cur.lastrowid} 新建：{why}")
        except Exception as e:
            log.error(f"[{inst}] 出错：{e}\n{traceback.format_exc()}")
            con.execute("insert into signals values(?,?,?,?,?)", (now_ms(), inst, None, "错误", str(e)[:300])); con.commit()
    write_dashboard(con, cfg)
    con.close()


# ---------------- 统计与看板 ----------------
def stats(con, inst=None):
    rows = con.execute("select * from campaigns where status='closed'" + (" and inst=?" if inst else "") + " order by exit_ts",
                       (inst,) if inst else ()).fetchall()
    n = len(rows)
    wins = [r for r in rows if r["net"] > 0]; losses = [r for r in rows if r["net"] <= 0]
    eq = peak = mdd = 0.0; curve = []
    for r in rows:
        eq += r["net"]; peak = max(peak, eq); mdd = max(mdd, peak - eq); curve.append(eq)
    gw = sum(r["net"] for r in wins); gl = -sum(r["net"] for r in losses)
    layers = {k: sum(1 for r in rows if r["max_layer"] == k) for k in range(1, 6)}
    return {"trades": n, "wins": len(wins), "losses": len(losses), "win_rate": len(wins) / n * 100 if n else 0.0,
            "gross": sum(r["gross"] for r in rows), "fees": sum(r["fee"] for r in rows), "net": eq,
            "avg_win": gw / len(wins) if wins else 0.0, "avg_loss": -gl / len(losses) if losses else 0.0,
            "profit_factor": gw / gl if gl else (float("inf") if gw else 0.0), "max_dd": mdd, "curve": curve, "layers": layers}


def report():
    con = db(); s = stats(con)
    print(f"已平仓 {s['trades']} 轮 | 胜 {s['wins']} 负 {s['losses']} | 胜率 {s['win_rate']:.1f}% | 毛利 {s['gross']:+.2f}U "
          f"手续费 {s['fees']:.2f}U 净利 {s['net']:+.2f}U | 盈亏因子 {s['profit_factor']:.2f} | 最大回撤 {s['max_dd']:.2f}U | 成交层数分布 {s['layers']}")
    for c in con.execute("select * from campaigns where status in ('pending','active')"):
        L = json.loads(c["layers"])
        print(f"#{c['id']} {c['inst']} {c['side']} {c['status']} 层: " + " / ".join(f"{x['px']:.6g}×{x['mult']}{'✓' if x['filled_ts'] else ''}" for x in L)
              + f" 止损{c['stop']:.6g} 止盈{c['tp'] or 0:.6g}")
    for c in con.execute("select * from campaigns where status='closed' order by exit_ts desc limit 20"):
        print(f"#{c['id']} {c['inst']} {c['side']} {bj(c['created_ts'])}→{bj(c['exit_ts'])} {c['max_layer']}层 均价{c['avg_px']:.6g}"
              f"→{c['exit_px']:.6g} {c['exit_reason']} 净{c['net']:+.2f}U")


def write_dashboard(con, cfg):
    s = stats(con)
    per = {i: stats(con, i) for i in cfg["instruments"]}
    live = con.execute("select * from campaigns where status in ('pending','active')").fetchall()
    closed = con.execute("select * from campaigns where status='closed' order by exit_ts desc limit 100").fetchall()
    sigs = con.execute("select * from signals order by ts desc limit 40").fetchall()
    first = con.execute("select min(ts) from signals").fetchone()[0]
    prices = {}
    for i in cfg["instruments"]:
        try:
            prices[i] = float(ox.ticker(i)["last"])
        except Exception:
            prices[i] = None

    def money(v):
        return f'<span class="{"up" if v > 0 else "down" if v < 0 else ""}">{v:+.2f}</span>'
    def f6(v):
        return f"{v:.6g}" if v else "-"
    pts = [0.0] + s["curve"]
    if len(pts) > 1:
        lo_, hi_ = min(pts), max(pts); span = (hi_ - lo_) or 1; W, H = 600, 160
        path = " ".join(f"{'M' if k == 0 else 'L'}{k / (len(pts) - 1) * W:.1f},{H - (y - lo_) / span * H:.1f}" for k, y in enumerate(pts))
        zero = H - (0 - lo_) / span * H
        svg = (f'<svg viewBox="0 -6 {W} {H + 12}" preserveAspectRatio="none" class="curve"><line x1="0" x2="{W}" y1="{zero:.1f}" '
               f'y2="{zero:.1f}" class="zero"/><path d="{path}" class="{"line-up" if s["net"] >= 0 else "line-down"}"/></svg>')
    else:
        svg = '<p class="muted">还没有平仓记录，第一轮平仓后显示权益曲线。</p>'

    def live_card(c):
        L = json.loads(c["layers"]); long = c["side"] == "long"; p = prices.get(c["inst"])
        rows = "".join(f"<tr><td>L{k + 1}</td><td>{x['px']:.6g}</td><td>×{x['mult']}</td><td>{x['qty']:.4g}</td>"
                       f"<td>{'✅ ' + bj(x['filled_ts']) if x['filled_ts'] else '挂单中'}</td>"
                       f"<td class='detail'>{'、'.join(x['sources'][:4])}</td></tr>" for k, x in enumerate(L))
        upnl = f"｜浮盈 {money((p - c['avg_px']) * c['qty'] * (1 if long else -1))}U" if c["qty"] and p else ""
        ctx = json.loads(c["context"] or "{}")
        return (f"<div class='card'><h2>{c['inst'].split('-')[0]} <span class='{'up' if long else 'down'}'>{'做多' if long else '做空'}</span> "
                f"#{c['id']} · {'已成交 ' + str(c['max_layer']) + ' 层' if c['qty'] else '等待第一层成交'}</h2>"
                f"<div class='muted'>现价 {f6(p)}｜统一止损 <span class='down'>{c['stop']:.6g}</span>｜"
                f"止盈 <span class='up'>{f6(c['tp']) if c['tp'] else '首层成交后计算'}</span>｜均价 {f6(c['avg_px'])}｜"
                f"数量 {c['qty'] or 0:.4g}{upnl}｜全部成交后打止损最多亏 {ctx.get('max_loss_if_all_filled', '-')}U</div>"
                f"<table><tr><th>层</th><th>挂单价</th><th>倍数</th><th>数量</th><th>状态</th><th>结构来源</th></tr>{rows}</table></div>")
    live_html = "".join(live_card(c) for c in live) or "<div class='card muted'>当前没有挂单或持仓</div>"
    per_rows = "".join(f"<tr><td>{i}</td><td>{cfg['instruments'][i]['unit']}</td><td>{p['trades']}</td><td>{p['win_rate']:.1f}%</td>"
                       f"<td>{money(p['net'])}</td><td>{p['fees']:.1f}</td><td>{p['max_dd']:.1f}</td></tr>" for i, p in per.items())
    closed_rows = "".join(
        f"<tr><td>{c['id']}</td><td>{c['inst'].split('-')[0]}</td><td class=\"{'up' if c['side'] == 'long' else 'down'}\">{'多' if c['side'] == 'long' else '空'}</td>"
        f"<td>{bj(c['created_ts'])}</td><td>{bj(c['exit_ts'])}</td><td>{c['max_layer']}</td><td>{c['qty']:.4g}</td><td>{c['avg_px']:.6g}</td>"
        f"<td>{c['exit_px']:.6g}</td><td>{c['exit_reason']}</td><td>{c['fee']:.1f}</td><td>{money(c['net'])}</td></tr>" for c in closed) \
        or '<tr><td colspan="12" class="muted">暂无</td></tr>'
    sig_rows = "".join(f"<tr><td>{bj(g['ts'])}</td><td>{g['inst'].split('-')[0]}</td><td>{g['decision']}</td><td class='detail'>{g['detail']}</td></tr>" for g in sigs)
    pf = "∞" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
    mults = " / ".join(str(m) for m in cfg["layer_multipliers"])
    btc_unit = cfg["instruments"].get("BTC-USDT-SWAP", {}).get("unit", "-")
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>模拟交易看板</title>
<style>
:root{{--bg:#0b0e11;--card:#161a1f;--line:#2a3038;--text:#e8eaed;--sub:#8b929c;--up:#16c784;--down:#ea3943;--wait:#f0b90b}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,"Microsoft YaHei",sans-serif;padding:16px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:0 0 8px}}
.muted{{color:var(--sub)}} .up{{color:var(--up)}} .down{{color:var(--down)}} .warn{{color:var(--wait)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:14px 0}}
.kpi,.card{{background:var(--card);border-radius:12px;padding:12px 14px}}
.kpi b{{display:block;font-size:22px;font-variant-numeric:tabular-nums}} .kpi span{{color:var(--sub);font-size:12px}}
.card{{margin-bottom:12px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;white-space:nowrap;margin-top:8px}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}} th{{color:var(--sub);font-weight:500;font-size:12px}}
td.detail{{white-space:normal;min-width:220px;color:var(--sub)}}
.curve{{width:100%;height:170px}} .curve path{{fill:none;stroke-width:2}} .line-up{{stroke:var(--up)}} .line-down{{stroke:var(--down)}}
.zero{{stroke:var(--line);stroke-dasharray:4 4}}
</style></head><body>
<h1>小时级马丁分批模拟交易</h1>
<div class="muted">只模拟，不下真实单｜分层倍数 {mults}（1 倍 = ETH 1 个 / BTC {btc_unit} 个）｜平仓每 1 倍扣 {cfg['fee_per_unit']}U，不计滑点｜
每 5 分钟运行｜开始于 {bj(first)}｜更新于 {datetime.now(TZ):%m-%d %H:%M:%S}</div>
<div class="warn" style="margin-top:6px">马丁加仓胜率通常很高，但一次打穿全部层就是大亏，请同时看净利润和最大回撤。</div>
<div class="grid">
<div class="kpi"><span>胜率</span><b>{s['win_rate']:.1f}%</b><span>{s['wins']} 胜 / {s['losses']} 负</span></div>
<div class="kpi"><span>净利润</span><b>{money(s['net'])} U</b><span>毛利 {s['gross']:+.2f}</span></div>
<div class="kpi"><span>累计手续费</span><b>{s['fees']:.2f} U</b><span>{s['trades']} 轮已平仓</span></div>
<div class="kpi"><span>盈亏因子</span><b>{pf}</b><span>均盈 {s['avg_win']:+.2f} / 均亏 {s['avg_loss']:+.2f}</span></div>
<div class="kpi"><span>最大回撤</span><b>{s['max_dd']:.2f} U</b><span>按已平仓计</span></div>
<div class="kpi"><span>成交层数分布</span><b style="font-size:15px">{' '.join(f'{k}层:{v}' for k, v in s['layers'].items())}</b><span>每轮最深成交到第几层</span></div>
</div>
<div class="card"><h2>权益曲线（净利润累计）</h2>{svg}</div>
{live_html}
<div class="card"><h2>分合约</h2><table><tr><th>合约</th><th>1倍数量</th><th>轮数</th><th>胜率</th><th>净利U</th><th>手续费U</th><th>最大回撤U</th></tr>{per_rows}</table></div>
<div class="card"><h2>已平仓（最近 100 轮）</h2><table><tr><th>#</th><th>合约</th><th>方向</th><th>挂单</th><th>平仓</th><th>层数</th><th>数量</th><th>均价</th><th>出场</th><th>结果</th><th>手续费</th><th>净利U</th></tr>{closed_rows}</table></div>
<div class="card"><h2>最近分析记录</h2><table><tr><th>时间</th><th>合约</th><th>决策</th><th>说明</th></tr>{sig_rows}</table></div>
</body></html>"""
    (ROOT / "web").mkdir(exist_ok=True)
    (ROOT / "web" / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    setup_logging()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run_once()
    elif cmd == "report":
        report()
