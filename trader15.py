"""
15 分钟级缠论单仓交易（默认只模拟；live_trading=true 时同步下单到 OKX）。

规则 v5（chan15_lab2.py 控制变量测试 + bt_v5.py 两段回测：8 个品种、窄中枢、只做多、不追高、低效率信号需指标确认）：
  品种：ETH / BTC / SOL / XRP / DOGE / SUI / ZEC / HYPE，每个品种各自一个仓位，1 倍 ≈ 2500U 名义价值
  入场：15m 新确认的三买做多，按当前价成交（不计滑点），不加仓；不做空
        （2025 全年样本外回测：空单 145 笔全年 −34%，连 BTC −17% 的 11 月也亏；只做多 +83% / 回撤 24.8%。
          v3 曾多空都做，只因 2026-06 一个月空单有效，不稳健，已撤回）
  仓位：固定 1 倍（与回测一致）；sizing=step 可改回按评分分档 0.5 / 1 / 2 / 3 / 5 倍（sizing.py）
  过滤：① 大方向：1H*0.6 + 4H*0.4 趋势分不能明显相反（多单要求 > -1，空单要求 < +1）
        ② 区间套：1H 缠论方向（走势 + 近期买卖点）不能相反
        ③ 行情性质：只做趋势——1H 近 48 根 K 线趋势效率（净涨跌 / 逐根涨跌绝对值之和）≥ 20%；
           10–20% 的低效率信号需 TradeTrack 多周期指标评分确认（1H ≥ 40 且 4H ≥ 20，ttrack.py）
        ④ 窄中枢：15m 最近中枢宽度（ZG − ZD）≤ 1.5 倍 ATR(15m)，宽幅震荡后的突破不做
           （8 品种 90 天回测：胜率 25–28% → 50%，回撤约降到三分之一）
        ⑤ 不追高：现价离买卖点不超过 1.75 倍 ATR(15m)
        （v5 两段回测，每笔名义 = 权益 × 1 复利：最近 180 天 v4 +141% / 回撤 17% → +277% / 15%；
          2025 全年 +83% / 25% → +115% / 17%）
  止损：买卖点价格外 0.5 ATR(15m)；用 1 分钟 K 线高低点逐根判定，按止损价原价成交
  离场：止损，或持仓中 15m 出现反向买卖点（按当时价格平仓）；不设固定止盈
  费用：交易手续费 = 成交金额 × OKX 吃单费率（开平各一次；配置只读 API Key 时用账户真实费率，否则 Lv1 标准 0.05%），
        另加持仓期间 OKX 实际资金费（fees.py）；fee_mode=fixed 可改回每 1 倍固定 2U
  OKX 下单（可选，live_trading=true）：照着模拟交易同步开平仓，一律逐仓、开仓时附带交易所止损单，每轮对账（okx_trade.py）

用法：python trader15.py loop     常驻：算完一轮马上接着算（部署默认方式）
      python trader15.py run      只跑一轮
      python trader15.py report   统计
"""
import json
import logging
import os
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
import fees  # noqa: E402
import sizing  # noqa: E402
import ta  # noqa: E402

TZ = timezone(timedelta(hours=8))
CFG_PATH = ROOT / "trader15_config.json"
STRATEGY_VERSION = 5
DEFAULT_CFG = {
    "strategy_version": STRATEGY_VERSION,
    # 1 倍 ≈ 2500U 名义价值（按 2026-09 价格折算）
    "instruments": {"ETH-USDT-SWAP": {"unit": 1.0}, "BTC-USDT-SWAP": {"unit": 0.03}, "SOL-USDT-SWAP": {"unit": 20},
                    "XRP-USDT-SWAP": {"unit": 1700}, "DOGE-USDT-SWAP": {"unit": 26000}, "SUI-USDT-SWAP": {"unit": 2200},
                    "ZEC-USDT-SWAP": {"unit": 1.8}, "HYPE-USDT-SWAP": {"unit": 28}},
    "fee_mode": "okx",          # okx：成交金额 × OKX 吃单费率 + 实际资金费（见 fees.py）；fixed：每 1 倍固定 fee_per_unit
    "fee_per_unit": 2.0,
    "entry_points": ["三买"],   # 只做多；加上 "三卖" 就多空都做（2025 年回测空单全年亏损，不建议）
    "trend_filter": True,
    "trend_threshold": 1.0,
    "nest_filter": True,
    "min_trend_eff_1h": 0.20,  # 1H 趋势效率下限，0 表示不过滤
    "weak_eff_min": 0.10,      # 1H 趋势效率在 [weak_eff_min, min_trend_eff_1h) 的“低效率”信号，需 TradeTrack 评分确认才做（ttrack.py）
    "weak_tt_1h": 40,          # 低效率信号要求 TradeTrack 1H 评分（多单 ≥，空单 ≤ 负值）
    "weak_tt_4h": 20,          # 低效率信号要求 TradeTrack 4H 评分
    "max_chase_atr": 1.75,     # 现价离买卖点超过几倍 ATR(15m) 就不追，0 表示不限
    "max_zs_width_atr": 1.5,   # 15m 最近中枢宽度上限（倍 ATR），0 表示不过滤
    "loop_interval_sec": 1,    # 常驻模式两轮之间的最短间隔（秒），0 表示算完马上接着算
    "sizing": "",              # 动态仓位：step=按评分分档 0.5/1/2/3/5 倍，linear=0.5~5 倍线性，""=固定 1 倍（见 sizing.py）
    "stop_buffer_atr": 0.5,
    "fresh_bars": 3,            # 买卖点确认后多少根 15m 内还算“新”
    # 告警推送（可选，留空即关闭）：钉钉 / 企业微信群机器人 Webhook，或 Telegram 机器人
    "alert_webhook": "",
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "alert_after_errors": 3,    # 连续失败几次运行才告警
    "alert_trades": False,      # 开仓 / 平仓时是否也推送
    # OKX 下单（见 okx_trade.py）：模拟交易照常运行并做决策，开启后每笔开平仓同步到 OKX；一律逐仓（多空方向都支持）
    "live_trading": False,      # true：同步下单到 OKX（需 okx_api.json 里的交易权限 Key）
    "live_leverage": 10,        # 逐仓杠杆（10 倍：强平约在开仓价下方 9.5%，回测里所有止损都在强平之前）
    "live_sizing": "equity",    # equity：每笔名义价值 = 开仓时 OKX 账户 USDT 权益 × live_equity_frac；fixed：模拟仓位数量 × live_size_factor
    "live_equity_frac": 1.0,    # equity 模式下每笔名义价值占权益的比例（1.0 = 每笔 1 倍，10 倍杠杆时每笔保证金 = 权益 10%；多个品种同时持仓时合计会超过 1 倍）
    "live_size_factor": 1.0,    # fixed 模式：OKX 下单数量 = 模拟仓位数量 × 该系数（1.0 即每个品种约 2500U 名义价值）
    "live_allow_real": False,   # 安全锁：Key 不是模拟盘（okx_api.json 里 simulated 不为 true）时，必须设为 true 才会下真实订单
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


# 策略升级时由新默认值覆盖的配置项（其余如告警、费用设置保留用户自己的）
STRATEGY_KEYS = ("instruments", "entry_points", "min_trend_eff_1h", "max_zs_width_atr", "sizing")


def migrate_cfg(raw):
    """旧版配置升级到当前策略，先备份旧配置。
    v1 → 当前：覆盖全部策略项（v1 与之后的规则差别很大）。v2 / v3 / v4 → 当前：只改入场点（v4 起只做多）；
    v5 新增的设置项（weak_* / max_chase_atr）由 load_cfg 自动补上默认值。
    已平仓记录归档到 trader15_v{旧版本}.db，新策略从零开始统计；未平仓位（含 OKX 上在跟踪的）带到新库继续管理。"""
    user = json.loads(raw)
    old = user.get("strategy_version", 1)
    if old >= STRATEGY_VERSION:
        return user
    CFG_PATH.with_name(f"trader15_config.v{old}.json").write_text(raw, encoding="utf-8")
    note = f"旧配置备份为 trader15_config.v{old}.json"
    if DB.exists():            # 先归档再写新配置：归档失败（如 Windows 上文件被占用）时下次启动会重试
        arch = ROOT / f"trader15_v{old}.db"
        DB.rename(arch)
        n = carry_open_trades(arch)
        note += f"，已平仓记录归档为 {arch.name}" + (f"，{n} 笔未平仓位带到新库继续跟踪" if n else "")
    if old < 2:
        user.update({k: DEFAULT_CFG[k] for k in STRATEGY_KEYS})
    else:
        user["entry_points"] = DEFAULT_CFG["entry_points"]        # v2 / v3 → 当前：只改入场点
    user["strategy_version"] = STRATEGY_VERSION
    CFG_PATH.write_text(json.dumps(user, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"策略升级 v{old} → v{STRATEGY_VERSION}：{note}")
    return user


def carry_open_trades(arch):
    """归档旧库时，把还没平的仓位（模拟未平，或 OKX 上还在持仓 / 待补记）连同已用过的买卖点带到新库，
    保留原编号，由新策略接着按原规则止损 / 离场、继续和 OKX 对账；否则交易所上的仓位会没人管。返回带过去的笔数。"""
    old = db(arch)
    rows = old.execute("select * from trades where exit_ts is null or live_status in ('open', 'closing')").fetchall()
    pts = old.execute("select * from used_points").fetchall()
    old.close()
    con = db()
    for r in rows:
        ks = list(r.keys())
        con.execute(f"insert into trades({','.join(ks)}) values({','.join('?' * len(ks))})", tuple(r))
    con.executemany("insert or ignore into used_points values(?,?,?)", [tuple(p) for p in pts])
    con.commit(); con.close()
    return len(rows)


def load_cfg():
    if not CFG_PATH.exists():
        CFG_PATH.write_text(json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2), encoding="utf-8")
    user = migrate_cfg(CFG_PATH.read_text(encoding="utf-8"))
    missing = [k for k in DEFAULT_CFG if k not in user]
    if missing:                # 新版本新增的设置项写进配置文件（取默认值），方便直接打开修改；已有的值不动
        user.update({k: DEFAULT_CFG[k] for k in missing})
        CFG_PATH.write_text(json.dumps(user, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"配置文件补充新设置项（默认值）：{'、'.join(missing)}")
    return {**DEFAULT_CFG, **user}


def db(path=None):
    con = sqlite3.connect(path or DB)
    con.row_factory = sqlite3.Row
    con.executescript("""
    create table if not exists trades(
        id integer primary key autoincrement, inst text, side text, point text, point_ts integer, point_px real,
        qty real, mult real, entry_ts integer, entry_px real, stop real,
        exit_ts integer, exit_px real, exit_reason text, gross real, fee real, net real,
        last_checked_ts integer, context text, funding real default 0);
    create table if not exists used_points(inst text, type text, ts integer, primary key(inst, type, ts));
    create table if not exists signals(ts integer, inst text, price real, decision text, detail text);
    create index if not exists idx_signals_inst_ts on signals(inst, ts);
    create index if not exists idx_signals_ts on signals(ts);
    """)
    cols = [r[1] for r in con.execute("pragma table_info(trades)")]
    # 旧库升级；live_*：OKX 实际成交（live_status：open 持仓中 / closing 已平待补记 / closed 已平 / failed 下单失败）
    for c, t in (("funding", "real default 0"), ("live_status", "text"), ("live_sz", "real"), ("live_entry", "real"),
                 ("live_exit", "real"), ("live_fee", "real"), ("live_funding", "real"), ("live_net", "real"),
                 ("live_exit_ts", "integer"), ("live_note", "text")):
        if c not in cols:
            con.execute(f"alter table trades add column {c} {t}")
    con.commit()
    return con


def now_ms():
    return int(time.time() * 1000)


def bj(ts):
    return datetime.fromtimestamp(ts / 1000, TZ).strftime("%m-%d %H:%M") if ts else "-"


QUIET = {"观望", "持仓", "放弃"}   # 每分钟都在重复的状态：同类记录最多 5 分钟记一条，免得刷屏


def signal(con, inst, price, decision, detail):
    if decision in QUIET:
        last = con.execute("select ts, decision, detail from signals where inst=? order by ts desc limit 1", (inst,)).fetchone()
        if last and last["decision"] == decision and now_ms() - last["ts"] < 300_000 and (decision != "放弃" or last["detail"] == detail):
            return
    con.execute("insert into signals values(?,?,?,?,?)", (now_ms(), inst, price, decision, detail)); con.commit()
    log.info(f"[{inst}] {decision}：{detail}")
    if decision in ("开仓", "平仓", "实盘") and _CFG.get("alert_trades"):
        alert(_CFG, f"[{inst}] {decision}：{detail}")


_CFG = {}
_MODE = {}
_LAST_DASH = [0.0]


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
    import os
    hb["last_run_ts"] = now
    hb["pid"] = os.getpid()
    hb["mode"] = "loop" if _MODE.get("loop") else "once"
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
    """返回 (新确认的买卖点 [(类型, 时间, 价格)], 15m 最近中枢 {zg, zd} 或 None)。"""
    ch = chan.analyze(df15)
    n = len(df15)
    return [(p["type"], int(df15.ts.iloc[p["i"]]), float(p["price"])) for p in ch["points"] if p["i"] >= n - cfg["fresh_bars"]], ch.get("last_zs")


def candles_1m_since(inst, since_ms):
    rows = ox.get("/api/v5/market/candles", {"instId": inst, "bar": "1m", "limit": 300})
    while rows and int(rows[-1][0]) > since_ms and len(rows) < 3000:
        more = ox.get("/api/v5/market/history-candles", {"instId": inst, "bar": "1m", "limit": 100, "after": rows[-1][0]})
        if not more:
            break
        rows += more
    return sorted({(int(r[0]), float(r[2]), float(r[3])) for r in rows if r[8] == "1" and int(r[0]) >= since_ms})


def costs(cfg, t, exit_px, end_ms):
    """返回 (交易手续费, 资金费, 费率说明)。fee_mode=okx：成交金额 × OKX 吃单费率（开平各一次）+ 持仓期间实际资金费；
    fixed：每 1 倍固定 fee_per_unit。"""
    if cfg.get("fee_mode", "okx") != "okx":
        return cfg["fee_per_unit"] * t["mult"], 0.0, f"固定每 1 倍 {cfg['fee_per_unit']}U"
    r = fees.rates(t["inst"])
    s = 1 if t["side"] == "long" else -1
    fee = fees.trade_fee(t["entry_px"], exit_px, t["qty"], r["taker"])
    fund = fees.funding_cost(s, t["qty"], t["entry_px"], t["entry_ts"], end_ms, fees.funding_history(t["inst"], t["entry_ts"]))
    return fee, fund, f"吃单 {r['taker'] * 100:.3f}%（{r['source']}）"


def close_trade(con, cfg, t, px, ts_, reason):
    s = 1 if t["side"] == "long" else -1
    gross = (px - t["entry_px"]) * s * t["qty"]
    fee, fund, _ = costs(cfg, t, px, ts_)
    net = gross - fee - fund
    con.execute("update trades set exit_ts=?, exit_px=?, exit_reason=?, gross=?, fee=?, funding=?, net=?, last_checked_ts=? where id=?",
                (ts_, px, reason, gross, fee, fund, net, ts_, t["id"])); con.commit()
    return net


# ---------------- OKX 同步下单（okx_trade.py）----------------
# 模拟交易仍是唯一的决策者；OKX 上只是照着开平仓。止损单开仓时就挂在交易所上，
# 所以交易所可能比模拟交易先止损；每轮对账，平掉的仓位从 positions-history 补记真实成交价、手续费、资金费。
_LIVE_WARNED = set()


def live_mode(cfg):
    """返回 (模式说明或 None, 错误说明或 None)。模式说明：“OKX 模拟盘” / “OKX 实盘”。"""
    if not cfg.get("live_trading"):
        return None, None
    import okx_trade
    c = okx_trade.creds()
    if not c:
        return None, "已开启 live_trading，但没有配置 OKX API Key（okx_api.json 或环境变量），本轮只做模拟交易"
    if not c["simulated"] and not cfg.get("live_allow_real"):
        return None, ("已开启 live_trading，Key 不是模拟盘（okx_api.json 里 simulated 不为 true），"
                      "但 live_allow_real 没有设为 true：安全锁生效，不下真实订单")
    return ("OKX 模拟盘" if c["simulated"] else "OKX 实盘"), None


def live_qty(cfg, inst, qty):
    """OKX 下单币数与说明。equity 模式按开仓时的账户权益算，并先检查可用保证金够不够（不够就不下单，说清楚原因）。"""
    import okx_trade
    if cfg.get("live_sizing", "equity") != "equity":
        k = cfg.get("live_size_factor", 1.0)
        return qty * k, f"模拟仓位 × {k:g}"
    b = okx_trade.balance_usdt()
    frac = cfg.get("live_equity_frac", 1.0)
    px = float(ox.ticker(inst)["last"])
    notional = b["eq"] * frac
    need = notional / cfg.get("live_leverage", 10)
    if need > b["availBal"]:
        raise RuntimeError(f"可用保证金不足：按权益 {b['eq']:,.0f}U × {frac:g} 开 {notional:,.0f}U，逐仓 {cfg.get('live_leverage', 10)} 倍"
                           f"需保证金 {need:,.0f}U，可用只有 {b['availBal']:,.0f}U（其他品种持仓占用了保证金）")
    return notional / px, f"按权益 {b['eq']:,.0f}U × {frac:g} ≈ {notional:,.0f}U"


def live_open(con, cfg, tid, inst, side, qty, stop):
    import okx_trade
    lev = cfg.get("live_leverage", 10)
    try:
        q, size_note = live_qty(cfg, inst, qty)
        r = okx_trade.open_position(inst, side, q, stop, lev, tid)
    except Exception as e:
        # 下单请求本身出错时，订单可能其实已成交：以交易所持仓为准，有持仓就接着跟踪并确认止损单
        try:
            pos = okx_trade.position(inst, side)
            if pos:
                note = f"下单返回异常（{e}），但交易所已有持仓，按持仓跟踪；" + okx_trade.ensure_stop(inst, side, stop, tid)
                con.execute("update trades set live_status='open', live_sz=?, live_entry=?, live_note=? where id=?",
                            (pos["sz"], pos["avgPx"], note[:300], tid)); con.commit()
                signal(con, inst, pos["avgPx"], "实盘", f"#{tid} {note}")
                return
        except Exception as e2:
            e = f"{e}；核对持仓也失败：{e2}"
        con.execute("update trades set live_status='failed', live_note=? where id=?", (str(e)[:300], tid)); con.commit()
        signal(con, inst, None, "错误", f"#{tid} OKX 开仓失败：{e}"[:300])
        alert(cfg, f"[{inst}] #{tid} OKX 开仓失败：{e}")
        return
    con.execute("update trades set live_status='open', live_sz=?, live_entry=?, live_fee=?, live_note=? where id=?",
                (r["sz"], r["avgPx"], r["fee"], r.get("note"), tid)); con.commit()
    signal(con, inst, r["avgPx"], "实盘", f"#{tid} {cfg['_live']}逐仓 {lev} 倍开{'多' if side > 0 else '空'} {r['sz']:g} 张（{size_note}）@{r['avgPx']:.6g}，"
                                         f"止损单 {r['stop']:.6g}" + (f"；{r['note']}" if r.get("note") else ""))


def _side(t):
    return 1 if t["side"] == "long" else -1


def live_close(con, cfg, tid, inst):
    """模拟交易平仓后，把 OKX 上对应的仓位也平掉（交易所已经止损的话只撤剩余止损单）。"""
    import okx_trade
    t = con.execute("select side from trades where id=?", (tid,)).fetchone()
    try:
        okx_trade.close_position(inst, _side(t), tid)
    except Exception as e:
        signal(con, inst, None, "错误", f"#{tid} OKX 平仓失败，下一轮重试（止损单仍在交易所上）：{e}"[:300])
        alert(cfg, f"[{inst}] #{tid} OKX 平仓失败：{e}")
        return
    con.execute("update trades set live_status='closing', live_exit_ts=? where id=?", (now_ms(), tid)); con.commit()
    try:
        live_record(con, tid, inst)
    except Exception as e:
        live_error(con, inst, f"#{tid} OKX 平仓记录暂时查不到，下一轮再查：{e}")


def live_record(con, tid, inst):
    """从 OKX 仓位历史补记真实平仓结果；平仓后几秒内可能还查不到，下一轮再查。"""
    import okx_trade
    t = con.execute("select * from trades where id=?", (tid,)).fetchone()
    rec = okx_trade.closed_record(inst, t["entry_ts"], _side(t))
    if not rec:
        if now_ms() - (t["live_exit_ts"] or now_ms()) > 30 * 60_000:     # 30 分钟仍查不到：结束对账，留待人工核对
            con.execute("update trades set live_status='closed', live_note=trim(coalesce(live_note, '') || ' OKX 未查到平仓记录，请人工核对') "
                        "where id=?", (tid,)); con.commit()
            signal(con, inst, None, "错误", f"#{tid} OKX 平仓 30 分钟后仍查不到仓位历史，请到 OKX 人工核对")
        return
    con.execute("""update trades set live_status='closed', live_exit=?, live_net=?, live_fee=?, live_funding=?, live_exit_ts=?,
                   live_note=trim(coalesce(live_note, '') || ' ' || ?) where id=?""",
                (rec["closeAvgPx"], rec["net"], rec["fee"], rec["funding"], rec["uTime"], rec["type"], tid)); con.commit()
    signal(con, inst, rec["closeAvgPx"], "实盘", f"#{tid} OKX {rec['type']} {rec['openAvgPx']:.6g}→{rec['closeAvgPx']:.6g} "
                                                 f"净{rec['net']:+.2f}U（手续费 {rec['fee']:.2f}U、资金费 {rec['funding']:+.2f}U）")


def live_reconcile(con, cfg, inst):
    """每轮对账：模拟已平而 OKX 未平 → 补平；OKX 已无持仓（止损单成交 / 人工平仓 / 强平）→ 补记真实结果。"""
    import okx_trade
    for t in con.execute("select * from trades where inst=? and live_status in ('open', 'closing')", (inst,)).fetchall():
        if t["live_status"] == "open":
            if t["exit_ts"] is not None:
                live_close(con, cfg, t["id"], inst)
                continue
            if okx_trade.position(inst, _side(t)):
                continue
            con.execute("update trades set live_status='closing', live_exit_ts=? where id=?", (now_ms(), t["id"])); con.commit()
            signal(con, inst, None, "实盘", f"#{t['id']} OKX 上的持仓已平（止损单成交或人工平仓），模拟交易继续按自己的规则跟踪")
            try:
                okx_trade.cancel_stops(inst)
            except Exception:
                pass
        live_record(con, t["id"], inst)


def live_error(con, inst, msg):
    """OKX 接口出错只记录、不影响模拟交易；同一合约同一条错误 5 分钟内只记一次。"""
    k = (inst, msg[:80])
    if time.time() - _LIVE_ERR_TS.get(k, 0) >= 300:
        _LIVE_ERR_TS[k] = time.time()
        signal(con, inst, None, "错误", msg[:300])


_LIVE_ERR_TS = {}


def handle(con, cfg, inst, unit):
    if cfg.get("_live"):
        try:
            live_reconcile(con, cfg, inst)
        except Exception as e:
            live_error(con, inst, f"OKX 对账出错（模拟交易照常进行）：{e}")
    price = float(ox.ticker(inst)["last"])
    df15 = ox.candles(inst, "15m", 400)
    fresh, zs15 = fresh_points(df15, cfg)
    t = con.execute("select * from trades where inst=? and exit_ts is null", (inst,)).fetchone()
    if t:
        long = t["side"] == "long"
        last = t["last_checked_ts"]
        for ts_, h, l in candles_1m_since(inst, last):          # 1) 止损：1 分钟 K 线逐根判定
            if (l <= t["stop"]) if long else (h >= t["stop"]):
                net = close_trade(con, cfg, t, t["stop"], ts_ + 60_000, "止损")
                signal(con, inst, price, "平仓", f"#{t['id']} 止损 {t['entry_px']:.6g}→{t['stop']:.6g} 净{net:+.2f}U")
                if cfg.get("_live") and t["live_status"] == "open":
                    live_close(con, cfg, t["id"], inst)
                return
            last = ts_ + 60_000
        con.execute("update trades set last_checked_ts=? where id=?", (last, t["id"])); con.commit()
        want = SHORT if long else LONG                            # 2) 反向买卖点离场
        opp = next(((tp, ts_) for tp, ts_, _ in fresh if tp in want and ts_ >= t["entry_ts"] - 45 * 60_000
                     and ts_ > t["point_ts"]), None)
        if opp:
            net = close_trade(con, cfg, t, price, now_ms(), f"反向{opp[0]}")
            signal(con, inst, price, "平仓", f"#{t['id']} 15m 出现{opp[0]}，按 {price:.6g} 离场 净{net:+.2f}U")
            if cfg.get("_live") and t["live_status"] == "open":
                live_close(con, cfg, t["id"], inst)
            return
        upnl = (price - t["entry_px"]) * (1 if long else -1) * t["qty"]
        fee, fund, _ = costs(cfg, t, price, now_ms())
        signal(con, inst, price, "持仓", f"#{t['id']} {t['point']}{'多' if long else '空'} {t['mult']:g} 倍 @{t['entry_px']:.6g} 止损{t['stop']:.6g} "
                                        f"浮盈{upnl:+.2f}U（扣除预计手续费 {fee:.2f}U、已发生资金费 {fund:+.2f}U 后 {upnl - fee - fund:+.2f}U）")
        return

    cands = []
    for tp, ts_, px in reversed(fresh):
        if tp not in cfg["entry_points"]:
            continue
        if con.execute("select 1 from used_points where inst=? and type=? and ts=?", (inst, tp, ts_)).fetchone():
            continue
        con.execute("insert into used_points values(?,?,?)", (inst, tp, ts_)); con.commit()
        cands.append((tp, ts_, px))
    if not cands:          # 没有新信号：不记录（心跳照常更新，看板顶部可确认程序在运行）
        return
    df1h = ox.candles(inst, "1H", 400); df4h = ox.candles(inst, "4H", 300)
    r1, _, _ = ta.analyze_tf(df1h, "1H", 2.0); r4, _, _ = ta.analyze_tf(df4h, "4H", 2.0)
    trend = 0.6 * r1["trend_score"] + 0.4 * r4["trend_score"]
    ch1 = chan.analyze(df1h); bias = chan.bias(ch1)
    atr = float(ta.atr(df15).iloc[-1])
    zs_w = abs(zs15["zg"] - zs15["zd"]) / atr if zs15 and atr else None     # 15m 最近中枢宽度（倍 ATR）
    c48 = df1h.c.values[-49:]
    eff = abs(c48[-1] - c48[0]) / (sum(abs(c48[i] - c48[i - 1]) for i in range(1, len(c48))) or 1)
    notes = []
    weak_min = cfg.get("weak_eff_min") or cfg["min_trend_eff_1h"]
    if eff < weak_min:
        signal(con, inst, price, "放弃", f"{'、'.join(c[0] for c in cands)}：1H 趋势效率 {eff:.0%} < {weak_min:.0%}，"
                                        f"行情在震荡，不做震荡里的突破")
        return
    weak = bool(eff < cfg["min_trend_eff_1h"])    # 低效率（弱）信号：要 TradeTrack 1H / 4H 指标评分确认
    tt = {}
    if weak:
        import ttrack
        for tf, dfx in (("1H", df1h), ("4H", df4h)):
            x = dfx.iloc[-300:]
            r = ttrack.analyze(x.o.values, x.h.values, x.l.values, x.c.values, x.volQuote.values, price, tf)
            tt[tf] = r["score"] if r else 0
    for tp, ts_, px in cands:
        side = 1 if tp in LONG else -1
        if cfg["trend_filter"] and trend * side <= -cfg["trend_threshold"]:
            notes.append(f"{tp}与大方向相反（趋势分 {trend:+.2f}）"); continue
        if cfg["nest_filter"] and bias * side < 0:
            notes.append(f"{tp}与 1H 缠论方向相反（1H 走势{ch1['trend']}，近期{ch1.get('recent') or '无'}）"); continue
        if weak and (tt["1H"] * side < cfg.get("weak_tt_1h", 40) or tt["4H"] * side < cfg.get("weak_tt_4h", 20)):
            notes.append(f"{tp}：1H 趋势效率 {eff:.0%} 偏低，TradeTrack 评分 1H {tt['1H']:+d} / 4H {tt['4H']:+d} "
                         f"未达 {cfg.get('weak_tt_1h', 40)} / {cfg.get('weak_tt_4h', 20)}，不做"); continue
        if cfg.get("max_chase_atr") and (price - px) * side > cfg["max_chase_atr"] * atr:
            notes.append(f"{tp}：现价离买卖点 {px:.6g} 已 {(price - px) * side / atr:.2f} 倍 ATR > {cfg['max_chase_atr']:g}，追得太高不做"); continue
        stop = px - side * cfg["stop_buffer_atr"] * atr
        if (stop - price) * side >= 0:
            notes.append(f"{tp}（{px:.6g}）已被价格打穿，结构失效"); continue
        if cfg.get("max_zs_width_atr") and (zs_w is None or zs_w > cfg["max_zs_width_atr"]):
            notes.append(f"{tp}：15m 中枢宽 {'-' if zs_w is None else f'{zs_w:.2f}'} 倍 ATR > {cfg['max_zs_width_atr']:g}，"
                         f"宽幅震荡后的突破不做"); continue
        ts_now = now_ms()
        adx4h = float(r4["adx"]["adx"])
        score = sizing.strength(trend, eff, ch1["trend"], adx4h, side)
        mult = sizing.size_from_score(score, cfg["sizing"]) if cfg.get("sizing") else 1.0
        qty = unit * mult
        ctx = {"trend": round(trend, 2), "chan1h_trend": ch1["trend"], "chan1h_bias": bias, "atr15": round(atr, 6),
               "trend_eff_1h": round(eff, 3), "adx4h": round(adx4h, 1), "score": round(score, 3), "zs_w_atr": zs_w and round(zs_w, 2),
               "chase_atr": round((price - px) * side / atr, 2), "weak": weak, **({"tt1h": tt["1H"], "tt4h": tt["4H"]} if weak else {})}
        cur = con.execute("""insert into trades(inst, side, point, point_ts, point_px, qty, mult, entry_ts, entry_px, stop,
                             last_checked_ts, context) values(?,?,?,?,?,?,?,?,?,?,?,?)""",
                          (inst, "long" if side > 0 else "short", tp, ts_, px, qty, mult, ts_now, price, stop,
                           ts_now // 60_000 * 60_000 + 60_000, json.dumps(ctx, ensure_ascii=False)))
        con.commit()
        wk = f"【低效率信号，TradeTrack 确认 1H {tt['1H']:+d} / 4H {tt['4H']:+d}】" if weak else ""
        signal(con, inst, price, "开仓", f"#{cur.lastrowid} {tp}{'做多' if side > 0 else '做空'}{wk} {mult:g} 倍（评分 {score:.2f}）@{price:.6g}"
                                        f"（买卖点 {px:.6g}），止损 {stop:.6g}，风险 {abs(price - stop) * qty:.2f}U；"
                                        f"大方向 {trend:+.2f}，1H 缠论{ch1['trend']}，1H 趋势效率 {eff:.0%}，15m 中枢宽 {'-' if zs_w is None else f'{zs_w:.2f}'} ATR，4H ADX {adx4h:.0f}")
        if cfg.get("_live"):
            live_open(con, cfg, cur.lastrowid, inst, side, qty, stop)
        return
    signal(con, inst, price, "放弃", "；".join(notes))


def run_once():
    cfg = load_cfg(); _CFG.update(cfg); con = db()
    prev = read_heartbeat()
    errors = []
    cfg["_live"], live_err = live_mode(cfg)
    if live_err:
        errors.append(live_err)
        if live_err not in _LIVE_WARNED:          # 同一个配置问题只写一次日志
            _LIVE_WARNED.add(live_err); log.error(live_err)
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
    # 常驻模式下看板最多每 10 秒重写一次（有开平仓时立即重写）
    traded = con.execute("select count(*) from signals where ts>? and decision in ('开仓','平仓')", (now_ms() - 15_000,)).fetchone()[0]
    if not _MODE.get("loop") or traded or time.time() - _LAST_DASH[0] >= 10:
        try:
            write_dashboard(con, cfg)
            _LAST_DASH[0] = time.time()
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
            "gross": sum(r["gross"] for r in rows), "fees": sum(r["fee"] for r in rows), "funding": sum(r["funding"] or 0 for r in rows),
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
    LIVE_ST = {"open": "持仓", "closing": "已平待补记", "failed": "下单失败"}
    def live_cell(t):
        st = t["live_status"]
        if not st:
            return "<td class='muted'>-</td>"
        if st == "closed":
            return f"<td>{money(t['live_net'] or 0)}</td>" if t["live_net"] is not None else "<td class='muted'>待核对</td>"
        tip = (t["live_note"] or "").replace("'", "")
        sz = f" {t['live_sz']:g} 张" if st == "open" and t["live_sz"] else ""
        return f"<td title='{tip}'>{LIVE_ST.get(st, st)}{sz}</td>"
    lv = con.execute("select count(*) n, coalesce(sum(live_net), 0) net, coalesce(sum(live_fee), 0) fee, "
                     "coalesce(sum(live_net > 0), 0) w from trades where live_status='closed' and live_net is not null").fetchone()
    live_now = cfg.get("_live")
    lev_, frac_ = cfg.get("live_leverage", 10), cfg.get("live_equity_frac", 1.0)
    live_size = (f"每笔名义 = 权益 × {frac_:g}，保证金 = 权益 {frac_ / lev_:.0%}" if cfg.get("live_sizing", "equity") == "equity"
                 else f"数量 = 模拟仓位 × {cfg.get('live_size_factor', 1.0):g}")
    live_desc = (f"{live_now}同步下单（逐仓 {lev_} 倍，{live_size}）"
                 if live_now else "只模拟，不下真实单")
    live_bal = ""
    if live_now:
        try:
            import okx_trade
            b = okx_trade.balance_usdt()
            per_txt = (f"每笔 OKX 名义约 {b['eq'] * frac_:,.0f}U（保证金约 {b['eq'] * frac_ / lev_:,.0f}U）"
                   if cfg.get("live_sizing", "equity") == "equity" else "每笔 OKX 数量 = 模拟 1 倍数量 × " + f"{cfg.get('live_size_factor', 1.0):g}")
            live_bal = (f'<div class="card"><b>OKX 账户</b>｜USDT 权益 {b["eq"]:,.2f}U，可用 {b["availBal"]:,.2f}U｜{per_txt}'
                        f'｜模拟交易固定每个品种约 2500U（与回测一致），OKX 按权益开仓，两边盈亏金额不同，比较看收益率</div>')
        except Exception as e:
            live_bal = f'<div class="card muted">OKX 账户余额暂时读取失败：{str(e)[:120]}</div>'
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
                f"<td>{t['entry_px']:.6g}</td><td>{p and f'{p:.6g}'}</td><td class='down'>{t['stop']:.6g}</td><td>{up}</td><td>{bj(t['entry_ts'])}</td>{live_cell(t)}</tr>")
    open_rows = "".join(orow(t) for t in opens) or '<tr><td colspan="10" class="muted">无持仓</td></tr>'
    closed_rows = "".join(
        f"<tr><td>{t['id']}</td><td>{t['inst'].split('-')[0]}</td><td class='{'up' if t['side'] == 'long' else 'down'}'>{t['point']}</td>"
        f"<td>{bj(t['entry_ts'])}</td><td>{bj(t['exit_ts'])}</td><td>{t['entry_px']:.6g}</td><td>{t['exit_px']:.6g}</td>"
        f"<td>{t['exit_reason']}</td><td>{t['mult']:g}×</td><td>{t['fee']:.2f}</td><td>{(t['funding'] or 0):+.2f}</td><td>{money(t['net'])}</td>{live_cell(t)}</tr>" for t in closed) or '<tr><td colspan="13" class="muted">暂无</td></tr>'
    per_rows = "".join(f"<tr><td>{i}</td><td>{cfg['instruments'][i]['unit']}</td><td>{p['n']}</td><td>{p['wr']:.1f}%</td>"
                       f"<td>{money(p['net'])}</td><td>{p['fees']:.0f}</td><td>{p['mdd']:.1f}</td></tr>" for i, p in per.items())
    sig_rows = "".join(f"<tr><td>{bj(g['ts'])}</td><td>{g['inst'].split('-')[0]}</td><td>{g['decision']}</td><td class='d'>{g['detail']}</td></tr>" for g in sigs)
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    if cfg.get("fee_mode", "okx") == "okx":
        r_ = fees.rates(next(iter(cfg["instruments"])))
        fee_desc = f"手续费按 OKX 吃单费率 {r_['taker'] * 100:.3f}%（{r_['source']}），另计实际资金费"
    else:
        fee_desc = f"每笔每 1 倍扣 {cfg['fee_per_unit']}U"
    zs_desc = (f"+ 窄中枢（≤{cfg['max_zs_width_atr']:g} ATR）" if cfg.get("max_zs_width_atr") else "") + \
              (f" + 不追高（≤{cfg['max_chase_atr']:g} ATR）" if cfg.get("max_chase_atr") else "") + \
              (f" + 效率 {cfg['weak_eff_min']:.0%}–{cfg['min_trend_eff_1h']:.0%} 需指标确认（1H≥{cfg.get('weak_tt_1h', 40)} 4H≥{cfg.get('weak_tt_4h', 20)}）"
               if cfg.get("weak_eff_min") and cfg["weak_eff_min"] < cfg["min_trend_eff_1h"] else "")
    size_desc = "按强弱动态 0.5~5 倍" if cfg.get("sizing") else "固定 1 倍"
    hb = read_heartbeat()
    ce = hb.get("consecutive_errors", 0)
    # 服务端写入生成时间，页面里的脚本用浏览器当前时间判断是否停更（程序完全停止时也能发现）
    health = (f'<div id="health" class="{"bad" if ce else "ok"}" data-ts="{now_ms()}">'
              f'{"⚠️ 最近连续 " + str(ce) + " 次运行失败：" + hb.get("last_error", "") if ce else "● 运行正常"}'
              f'｜最后成功运行 {bj(hb.get("last_ok_ts"))}</div>'
              '<script>(function(){var e=document.getElementById("health");var m=(Date.now()-(+e.dataset.ts))/60000;'
              'if(m>5){e.className="bad";e.textContent="⚠️ 看板已 "+Math.round(m)+" 分钟没有更新，程序可能已停止（正常每 10 秒更新一次）";}})();</script>')
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>缠论交易看板</title>
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
<div class="muted">{live_desc}｜策略 v{cfg.get('strategy_version', 1)}｜{len(cfg['instruments'])} 个品种｜15m {' / '.join(cfg['entry_points'])}入场，区间套（1H 缠论）+ 大方向 + 只做趋势（1H 趋势效率≥{cfg['min_trend_eff_1h']:.0%}）{zs_desc}｜{size_desc}｜止损在买卖点外 0.5 ATR，反向买卖点离场｜
{fee_desc}，不计滑点｜常驻循环实时扫描｜开始于 {bj(first)}｜更新于 {datetime.now(TZ):%m-%d %H:%M:%S}</div>
{live_bal}
<div class="grid">
<div class="kpi"><span>胜率</span><b>{s['wr']:.1f}%</b><span>{s['w']} 胜 / {s['l']} 负</span></div>
<div class="kpi"><span>净利润</span><b>{money(s['net'])} U</b><span>毛利 {s['gross']:+.2f}</span></div>
<div class="kpi"><span>累计手续费 / 资金费</span><b>{s['fees']:.1f} / {s['funding']:+.1f} U</b><span>{s['n']} 笔已平仓</span></div>
<div class="kpi"><span>盈亏因子</span><b>{pf}</b><span>均盈 {s['avg_w']:+.2f} / 均亏 {s['avg_l']:+.2f}</span></div>
<div class="kpi"><span>最大回撤</span><b>{s['mdd']:.2f} U</b><span>按已平仓计</span></div>
<div class="kpi"><span>OKX 实际净利</span><b>{money(lv['net'])} U</b><span>{lv['n']} 笔｜胜 {lv['w']}｜手续费 {lv['fee']:.1f}</span></div>
</div>
<div class="card"><h2>权益曲线（净利润累计）</h2>{svg}</div>
<div class="card"><h2>当前持仓</h2><table><tr><th>合约</th><th>方向</th><th>倍数</th><th>评分</th><th>入场</th><th>现价</th><th>止损</th><th>浮盈U</th><th>开仓</th><th>OKX</th></tr>{open_rows}</table></div>
<div class="card"><h2>分合约</h2><table><tr><th>合约</th><th>模拟 1 倍数量</th><th>笔数</th><th>胜率</th><th>净利U</th><th>手续费U</th><th>最大回撤U</th></tr>{per_rows}</table></div>
<div class="card"><h2>已平仓（最近 100 笔）</h2><table><tr><th>#</th><th>合约</th><th>买卖点</th><th>开仓</th><th>平仓</th><th>入场</th><th>出场</th><th>结果</th><th>倍数</th><th>手续费</th><th>资金费</th><th>净利U</th><th>OKX 净利U</th></tr>{closed_rows}</table></div>
<div class="card"><h2>最近分析记录</h2><table><tr><th>时间</th><th>合约</th><th>决策</th><th>说明</th></tr>{sig_rows}</table></div>
</body></html>"""
    (ROOT / "web").mkdir(exist_ok=True)
    (ROOT / "web" / "index.html").write_text(html, encoding="utf-8")


LOCK = ROOT / "trader15.lock"
_LOCK_FH = [None]


def acquire_lock():
    """操作系统级文件锁（Linux flock / Windows msvcrt.locking）：进程一旦退出或被杀，锁由系统立即释放，
    不会留下“假锁”，也不需要靠 PID 或心跳去猜另一个实例是否还活着。成功返回 True。"""
    import os
    fh = open(LOCK, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False
    _LOCK_FH[0] = fh
    return True


def release_lock():
    fh = _LOCK_FH[0]
    if fh:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        fh.close()
        _LOCK_FH[0] = None


def run_locked():
    """只跑一轮；如果常驻进程或另一个单次运行正在进行，就跳过。"""
    if not acquire_lock():
        log.info("已有实例在运行，本次跳过")
        return
    try:
        run_once()
    finally:
        release_lock()


def loop():
    """常驻模式：算完一轮马上接着算（loop_interval_sec 为两轮之间的最短间隔，默认 1 秒，0 表示完全不停）。
    整个进程生命周期持有系统文件锁，保证只有一个常驻实例；单轮出错只记录不退出；
    进程本身挂掉由 systemd Restart=always / 看门狗拉起。"""
    if not acquire_lock():
        log.info("已有常驻进程或单次运行正在进行（系统文件锁被占用），本进程退出")
        return
    cfg = load_cfg()
    _MODE["loop"] = True
    log.info(f"常驻循环启动（PID {os.getpid()}，每轮最短间隔 {cfg.get('loop_interval_sec', 1)} 秒）")
    n = 0
    while True:
        t0 = time.time()
        try:
            run_once()
        except Exception as e:                     # 不让任何异常把循环带走
            log.error(f"本轮异常：{e}\n{traceback.format_exc()}")
            time.sleep(5)
        n += 1
        if n % 3600 == 0:
            log.info(f"常驻循环已运行 {n} 轮")
        time.sleep(max(0.0, float(load_cfg().get("loop_interval_sec", 1)) - (time.time() - t0)))


if __name__ == "__main__":
    import atexit
    import faulthandler
    if sys.stderr is None:           # pythonw（Windows 后台运行）没有 stderr：崩溃信息写到文件，避免“无声退出”
        sys.stderr = open(ROOT / "trader15.err", "a", encoding="utf-8", buffering=1)
    faulthandler.enable(sys.stderr)  # 连解释器级的硬崩溃也能留下调用栈
    setup_logging()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "loop":
        atexit.register(lambda: log.info(f"常驻进程退出（PID {os.getpid()}）"))
    try:
        {"run": run_locked, "report": report, "loop": loop}.get(cmd, run_locked)()
    except BaseException as e:       # 包括 KeyboardInterrupt / SystemExit，记下退出原因
        log.error(f"进程异常退出：{type(e).__name__} {e}\n{traceback.format_exc()}")
        raise
