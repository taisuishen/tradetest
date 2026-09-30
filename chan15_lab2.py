"""
15m 缠论实验台 v2（控制变量）：多品种、每根 15m 收盘预计算特征，5m K 线撮合（同根先判止损），每个合约一个仓位。
最近 30 天（“上月”）用来找思路，再往前 60 天（“验证”）检验是不是只对上个月有效；每组还看“去掉最赚 5 笔”后的净利，
判断是不是只靠少数大单。费用：开仓 / 止损 / 反向离场按吃单费率，挂单成交和止盈按挂单费率，另计实际资金费。

得出窄中枢规则（trader15.py v2）：8 个品种 + 1H 趋势效率 ≥20% + 15m 窄中枢（≤1.5 ATR），固定 1 倍；
v3 在此基础上多空都做（180 天按月对比见 bt_monthly.py）。
试过但不采用的：震荡里反做三买三卖、布林带回归、背驰买卖点、挂单回踩、固定止盈、跟踪止损、持仓时限、
不追高、信号 K 线同向、RSI 上限、15m / 1H 走势同向、ADX 门槛、止损后冷却、按时段过滤。

用法：python chan15_lab2.py [天数=90]      首次运行会拉 K 线并预计算（9 个品种约 10 分钟），结果缓存在 bt/
"""
import os, sys, time, pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
import chan15_lab as lab

M15, H, D = 900_000, 3_600_000, 86_400_000
LONG, SHORT = lab.LONG, lab.SHORT
OUT = Path(__file__).parent / "bt"
_G = {}


def _init(d):
    _G.update(d)


def task15(T):
    import chan, ta
    df = _G["df15"]
    s = df[df.ts + M15 <= T].iloc[-400:].reset_index(drop=True)
    if len(s) < 300:
        return T, None
    ch = chan.analyze(s)
    n = len(s)
    up, mid, lo = ta.boll(s.c, 20, 2)
    rsi = ta.rsi(s.c)
    c = s.c.values[-33:]
    eff15 = abs(c[-1] - c[0]) / (abs(pd.Series(c).diff()).sum() or 1)
    return T, {"fresh": [(p["type"], int(s.ts.iloc[p["i"]]), float(p["price"])) for p in ch["points"] if p["i"] >= n - 3],
               "close": float(s.c.iloc[-1]), "open": float(s.o.iloc[-1]), "high": float(s.h.iloc[-1]), "low": float(s.l.iloc[-1]),
               "atr": float(ta.atr(s).iloc[-1]), "trend15": ch["trend"], "zs": ch.get("last_zs"),
               "bb_up": float(up.iloc[-1]), "bb_mid": float(mid.iloc[-1]), "bb_lo": float(lo.iloc[-1]),
               "rsi": float(rsi.iloc[-1]), "rsi_prev": float(rsi.iloc[-2]), "eff15": float(eff15),
               "hi20": float(s.h.iloc[-21:-1].max()), "lo20": float(s.l.iloc[-21:-1].min())}


def live_insts():
    """实盘配置里的品种和 1 倍数量。"""
    import trader15
    return tuple((i, c["unit"]) for i, c in trader15.DEFAULT_CFG["instruments"].items())


def precompute(days, insts=(("ETH-USDT-SWAP", 1.0), ("BTC-USDT-SWAP", 0.03)), end_ms=None):
    """end_ms：回测区间的结束时间（默认到现在），区间为 end_ms 之前的 days 天。数据不够的品种跳过。"""
    import kcache, fees
    data = {}
    for inst, unit in insts:
        if unit is None:                     # 1 倍 ≈ 2500U 名义价值，和 ETH 1 个、BTC 0.03 个相当
            unit = 2500 / float(kcache.candles(inst, "1H", 2, progress=False).c.iloc[-1])
        t0 = time.time()
        kw = {"progress": False, "end_ms": end_ms, "per_sec": 9}
        df15 = kcache.candles(inst, "15m", 400 + days * 96 + 4, **kw)
        if len(df15) < 400:
            print(f"{inst} 该区间 K 线不足（{len(df15)} 根，可能还没上线），跳过", flush=True)
            continue
        df1h = kcache.candles(inst, "1H", 400 + days * 24 + 2, **kw)
        df4h = kcache.candles(inst, "4H", 300 + days * 6 + 2, **kw)
        df5 = kcache.candles(inst, "5m", days * 288 + 12, **kw)
        end = (end_ms // M15 * M15) if end_ms else int(df15.ts.iloc[-1]) + M15
        Ts = list(range(end - days * 96 * M15, end + 1, M15))
        Th = sorted({t // H * H for t in Ts})
        w = max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(w, initializer=_init, initargs=({"df15": df15},)) as ex:
            p15 = dict(ex.map(task15, Ts, chunksize=40))
        with ProcessPoolExecutor(w, initializer=lab._init, initargs=({"df1h": df1h, "df4h": df4h},)) as ex:
            p1h = dict(ex.map(lab.task1h, Th, chunksize=10))
        # 回测一律按 OKX Lv1 标准费率，不读账户费率（配了 API Key 时账户费率可能是 VIP 或模拟盘的，会让回测偏乐观）
        fm = {"taker": fees.DEFAULT["taker"], "maker": fees.DEFAULT["maker"],
              "funding": [x for x in fees.funding_history(inst, end - (days + 1) * D) if x[0] <= end]}
        data[inst] = (unit, Ts, p15, p1h, list(zip(df5.ts.astype("int64"), df5.h, df5.l, df5.c)), fm)
        print(f"{inst} 预计算 {time.time() - t0:.0f}s", flush=True)
    return data


# ---------------- 入场信号 ----------------
def sig_trend(info, hr, v, used):
    """原主策略：15m 新三买 / 三卖 + 大方向 + 区间套 + 1H 趋势效率。"""
    for t, ts_, px in reversed(info["fresh"]):
        if t not in v.get("types", {"三买", "三卖"}) or (t, ts_) in used:
            continue
        used.add((t, ts_))
        side = 1 if t in LONG else -1
        if not hr:
            continue
        if v.get("trend_filter", True) and hr["trend"] * side <= -1.0:
            continue
        if v.get("nest", True) and hr["bias"] * side < 0:
            continue
        if hr["eff1h"] < v.get("eff_min", 0.25):
            continue
        if v.get("eff_max") is not None and hr["eff1h"] >= v["eff_max"]:
            continue
        if v.get("trend_min") is not None and hr["trend"] * side < v["trend_min"]:
            continue
        if v.get("nest_strict") and hr["bias"] * side <= 0:
            continue
        if v.get("chan1h_same") and hr["trend1h_chan"] != ("上涨" if side > 0 else "下跌"):
            continue
        if v.get("adx4h_min") and hr["adx4h"] < v["adx4h_min"]:
            continue
        if v.get("adx1h_min") and hr["adx1h"] < v["adx1h_min"]:
            continue
        if v.get("trend15_same") and info["trend15"] != ("上涨" if side > 0 else "下跌"):
            continue
        if v.get("max_chase_atr") is not None and (info["close"] - px) * side > v["max_chase_atr"] * info["atr"]:
            continue                                   # 离买卖点太远，追高
        if v.get("bull_bar") and (info["close"] - info["open"]) * side <= 0:
            continue                                   # 信号 K 线要收成同向实体
        if v.get("rsi_cap") and (info["rsi"] if side > 0 else 100 - info["rsi"]) > v["rsi_cap"]:
            continue                                   # 15m 已超买 / 超卖，不追
        if v.get("btc_align") and v.get("btc") is not None:
            bh = v["btc"].get(T_NOW[0] // H * H)
            if not bh or bh["trend"] * side <= 0:
                continue                               # 大盘（BTC 1H/4H）不同向
        if v.get("skip_hours") and (T_NOW[0] // H + 8) % 24 in v["skip_hours"]:
            continue                               # 北京时间某些时段不做
        if v.get("gate") and not v["gate"](T_NOW[0], side, info, hr):
            continue                                   # 外部过滤（如 TradeTrack 多周期指标 / 支撑压力，见 bt_ttrack.py）
        if v.get("stop_mode") == "zs" and info["zs"]:
            stop = (info["zs"]["zg"] if side > 0 else info["zs"]["zd"]) - side * 0.2 * info["atr"]
        else:
            stop = px - side * v.get("stop_buf", 0.5) * info["atr"]
        if (stop - info["close"]) * side >= 0:
            continue
        if v.get("max_risk_atr") and abs(info["close"] - stop) > v["max_risk_atr"] * info["atr"]:
            continue
        o = {"side": side, "tag": t, "stop": stop, "point_ts": ts_,
             "feat": {"chase": (info["close"] - px) * side / info["atr"], "risk_atr": abs(info["close"] - stop) / info["atr"],
                      "eff1h": hr["eff1h"], "trend": hr["trend"] * side, "adx4h": hr["adx4h"], "adx1h": hr["adx1h"],
                      "rsi": info["rsi"] if side > 0 else 100 - info["rsi"], "eff15": info["eff15"],
                      "hour": (T_NOW[0] // H + 8) % 24, "side": side, "trend15": info["trend15"], "chan1h": hr["trend1h_chan"],
                      "btc": (v["btc"].get(T_NOW[0] // H * H) or {}).get("trend", 0) * side if v.get("btc") else 0,
                      "atr_pct": info["atr"] / info["close"],
                      "zs_w": abs(info["zs"]["zg"] - info["zs"]["zd"]) / info["atr"] if info["zs"] else None}}
        fz = o["feat"]
        if v.get("zs_w_max") and (fz["zs_w"] is None or fz["zs_w"] > v["zs_w_max"]):
            continue                                   # 中枢太宽：宽幅震荡后的突破不做
        if v.get("atr_pct_min") and fz["atr_pct"] < v["atr_pct_min"]:
            continue                                   # 波动太小：手续费占风险比例太高
        if v.get("risk_pct_min") and fz["risk_atr"] * fz["atr_pct"] < v["risk_pct_min"]:
            continue                                   # 止损距离占价格比例太小
        if v.get("sizing"):
            import sizing
            o["mult"] = sizing.size_from_score(sizing.strength(hr["trend"], hr["eff1h"], hr["trend1h_chan"], hr["adx4h"], side), v["sizing"])
        if v.get("entry") == "limit":          # 挂单等回踩到买卖点附近，而不是追价
            o.update(kind="limit", px=px + side * v.get("limit_atr", 0.2) * info["atr"], expire=v.get("limit_bars", 4))
        if v.get("entry") == "dip":            # 挂单在信号收盘价下方 dip_atr 倍 ATR 等回落（见 bt_v5_opt.py）
            lim = info["close"] - side * v.get("dip_atr", 0.3) * info["atr"]
            if (lim - stop) * side <= 0:
                continue
            o.update(kind="limit", px=lim, expire=v.get("limit_bars", 4))
        return o
    return None


def sig_fade3(info, hr, v, used):
    """震荡里反做三买 / 三卖：震荡中的突破多半是假突破。"""
    for t, ts_, px in reversed(info["fresh"]):
        if t not in {"三买", "三卖"} or (t, ts_) in used:
            continue
        used.add((t, ts_))
        if not hr or hr["eff1h"] >= v.get("eff_max", 0.25):
            continue
        side = -1 if t in LONG else 1
        stop = info["close"] - side * v.get("sl_atr", 1.0) * info["atr"]
        return {"side": side, "tag": "反" + t, "stop": stop, "point_ts": ts_}
    return None


def sig_revert(info, hr, v, used):
    """震荡均值回归：1H 趋势效率低时，15m 收在布林带外且 RSI 超买 / 超卖，就往中轨方向做。"""
    if not hr or hr["eff1h"] >= v.get("eff_max", 0.25):
        return None
    c, a = info["close"], info["atr"]
    side = 0
    if c < info["bb_lo"] and info["rsi"] < v.get("rsi_lo", 30):
        side = 1
    elif c > info["bb_up"] and info["rsi"] > v.get("rsi_hi", 70):
        side = -1
    if not side:
        return None
    if v.get("need_turn") and not ((info["rsi"] - info["rsi_prev"]) * side > 0):
        return None
    if v.get("with_trend") and hr["trend"] * side <= -1.0:
        return None
    o = {"side": side, "tag": "回归多" if side > 0 else "回归空", "stop": c - side * v.get("sl_atr", 1.5) * a}
    tpm = v.get("tp", "mid")
    o["tp"] = info["bb_mid"] if tpm == "mid" else (info["bb_up"] if side > 0 else info["bb_lo"]) if tpm == "band" else c + side * tpm * a
    return o


def sig_bc(info, hr, v, used):
    """震荡里做背驰类买卖点（一买 / 二买 / 盘背买，卖点反之），止盈看中枢中间。"""
    for t, ts_, px in reversed(info["fresh"]):
        if t not in v.get("types", {"一买", "二买", "盘背买", "一卖", "二卖", "盘背卖"}) or (t, ts_) in used:
            continue
        used.add((t, ts_))
        if not hr or hr["eff1h"] >= v.get("eff_max", 1.0):
            continue
        side = 1 if t in LONG else -1
        stop = px - side * v.get("stop_buf", 0.5) * info["atr"]
        if (stop - info["close"]) * side >= 0:
            continue
        o = {"side": side, "tag": t, "stop": stop, "point_ts": ts_}
        if v.get("tp") == "zs" and info["zs"]:
            tp = (info["zs"]["zg"] + info["zs"]["zd"]) / 2
            if (tp - info["close"]) * side > 0:
                o["tp"] = tp
        return o
    return None


T_NOW = [0]
SIGS = {"trend": sig_trend, "fade3": sig_fade3, "revert": sig_revert, "bc": sig_bc}


# ---------------- 撮合 ----------------
def simulate(d, v, t_from=None, t_to=None, step=M15):
    """v["sigs"]：按顺序尝试的信号模块列表 [(名字, 参数)]。通用离场参数放在各模块参数里：
    tp_rr 固定止盈倍数；be_r 浮盈几 R 后止损移到保本；opp 反向买卖点离场；max_bars 持仓超过多少根 15m 就平；
    trail_atr 按最高 / 最低价回撤几倍 ATR 跟踪止损；mult 固定倍数。"""
    unit, Ts, p15, p1h, b5, fm = d
    Ts = [t for t in Ts if (t_from is None or t >= t_from) and (t_to is None or t < t_to)]
    trades, pos, pend, used, j, last_stop = [], None, None, {}, 0, {}

    def close(px, ts_, why, rate_out):
        p = pos; q = unit * p["mult"]; rem = p.get("rem", 1.0)
        gross = (px - p["entry"]) * p["side"] * q * rem + p.get("realized", 0.0)
        fee = p["entry"] * q * p["rate_in"] + px * q * rem * rate_out + p.get("part_fee", 0.0)
        if why == "止损":
            last_stop[p["mod"]] = ts_
        fund = sum(p["side"] * r * q * p["entry"] for t, r in fm["funding"] if p["ts"] < t <= ts_)
        trades.append({"ts": p["ts"], "end": ts_, "tag": p["tag"], "mod": p["mod"], "side": p["side"], "entry": p["entry"],
                       "exit": px, "why": why, "mult": p["mult"], "R": gross / q / p["risk"],
                       "gross": gross, "fee": fee, "funding": fund, "net": gross - fee - fund, **p.get("feat", {})})

    def open_(o, entry, ts_, rate_in, mod, pv):
        risk = abs(entry - o["stop"])
        if risk <= 0 or (o["stop"] - entry) * o["side"] >= 0:
            return None
        tp = o.get("tp")
        if pv.get("tp_rr"):
            tp = entry + o["side"] * pv["tp_rr"] * risk
        if tp is not None and (tp - entry) * o["side"] <= 0:
            return None
        return {**o, "entry": entry, "ts": ts_, "risk": risk, "tp": tp, "rate_in": rate_in, "mod": mod, "pv": pv,
                "mult": o.get("mult", pv.get("mult", 1.0)), "best": entry, "be": False, "bars": 0}

    for T in Ts:
        info = p15.get(T)
        if not info:
            continue
        hr = p1h.get(T // H * H); T_NOW[0] = T
        # 1) 收盘时的离场判断：反向买卖点 / 持仓时间
        if pos:
            pv = pos["pv"]; pos["bars"] += 1
            if pv.get("opp", True) and pos.get("point_ts"):
                want = SHORT if pos["side"] > 0 else LONG
                opp = next((t for t, ts_, _ in info["fresh"] if t in want and ts_ > pos["point_ts"] and ts_ >= pos["ts"] - 45 * 60_000), None)
                if opp:
                    close(info["close"], T, "反向" + opp, fm["taker"]); pos = None
            if pos and pv.get("daily_exit") and T in pv["daily_exit"]:          # 日线收盘跌破 EMA 离场（Christian 式，见 bt_v5_christian.py）
                dc, de = pv["daily_exit"][T]
                if (dc - de) * pos["side"] < 0:
                    close(info["close"], T, "日线跌破EMA", fm["taker"]); pos = None
            if pos and pv.get("tt_exit") is not None and info.get("tt") and info["tt"]["1H"]["score"] * pos["side"] < pv["tt_exit"]:
                close(info["close"], T, "转向", fm["taker"]); pos = None      # TradeTrack 1H 评分转向（见 bt_ttrack.py）
            if pos and pv.get("max_bars") and pos["bars"] >= pv["max_bars"]:
                close(info["close"], T, "超时", fm["taker"]); pos = None
            if pos and pv.get("exit_fn"):          # 通用离场钩子：("close", 原因) 全平；("part", 比例) 先平一部分并移到保本（见 bt_v5_opt.py）
                act = pv["exit_fn"](info, pos, T)
                if act and act[0] == "close":
                    close(info["close"], T, act[1], fm["taker"]); pos = None
                elif act and act[0] == "part" and not pos.get("part"):
                    f, q, px_ = act[1], unit * pos["mult"], info["close"]
                    pos["realized"] = pos.get("realized", 0.0) + (px_ - pos["entry"]) * pos["side"] * q * f
                    pos["part_fee"] = pos.get("part_fee", 0.0) + px_ * q * f * fm["taker"]
                    pos["rem"] = pos.get("rem", 1.0) - f; pos["part"] = True
                    be = pos["entry"] * (1 + pos["side"] * 0.0012)
                    pos["stop"] = max(pos["stop"], be) if pos["side"] > 0 else min(pos["stop"], be); pos["be"] = True
        # 2) 找新信号
        if pos is None and pend is None:
            for mod, pv in v["sigs"]:
                if pv.get("cooldown") and T - last_stop.get(mod, -1e18) < pv["cooldown"] * step:
                    SIGS[mod](info, hr, {**pv, "eff_min": 9}, used.setdefault(mod, set()))   # 冷却期内的信号作废
                    continue
                o = SIGS[mod](info, hr, pv, used.setdefault(mod, set()))
                if not o:
                    continue
                if o.get("kind") == "limit":
                    pend = {**o, "mod": mod, "pv": pv, "left": o["expire"] * 3}
                else:
                    pos = open_(o, info["close"], T, fm["taker"], mod, pv)
                break
        # 3) 5m K 线：挂单成交 → 止损 → 保本 / 跟踪 → 止盈
        while j < len(b5) and b5[j][0] < T:
            j += 1
        k = j
        while k < len(b5) and b5[k][0] < T + step and (pos or pend):      # step：信号周期（默认 15 分钟，见 bt_tf.py）
            ts_, h, l, c = b5[k]
            if pend:
                s = pend["side"]
                if (l <= pend["px"]) if s > 0 else (h >= pend["px"]):
                    pos = open_(pend, pend["px"], ts_, fm["maker"], pend["mod"], pend["pv"]); pend = None
                else:
                    pend["left"] -= 1
                    if pend["left"] <= 0:
                        pend = None
                    k += 1; continue
            if pos:
                s, pv = pos["side"], pos["pv"]
                if (l <= pos["stop"]) if s > 0 else (h >= pos["stop"]):
                    close(pos["stop"], ts_ + 300_000, "保本" if pos["be"] else "止损", fm["taker"]); pos = None; k += 1; continue
                best = h if s > 0 else l
                pos["best"] = max(pos["best"], best) if s > 0 else min(pos["best"], best)
                if pv.get("part_r") and not pos.get("part") and (pos["best"] - pos["entry"]) * s >= pv["part_r"] * pos["risk"]:
                    f = pv.get("part_frac", 0.5); q = unit * pos["mult"]; px_ = pos["entry"] + s * pv["part_r"] * pos["risk"]
                    pos["realized"] = (px_ - pos["entry"]) * s * q * f; pos["part_fee"] = px_ * q * f * fm["maker"]
                    pos["rem"] = 1 - f; pos["part"] = True
                    if pv.get("part_be", True):
                        be = pos["entry"] * (1 + s * 0.0012)
                        pos["stop"] = max(pos["stop"], be) if s > 0 else min(pos["stop"], be); pos["be"] = True
                if pv.get("be_r") and not pos["be"] and (pos["best"] - pos["entry"]) * s >= pv["be_r"] * pos["risk"]:
                    be = pos["entry"] * (1 + s * 0.0012)             # 保本 + 覆盖手续费
                    pos["stop"] = max(pos["stop"], be) if s > 0 else min(pos["stop"], be); pos["be"] = True
                if pv.get("trail_atr") and (pos["best"] - pos["entry"]) * s >= pv.get("trail_after_r", 1.0) * pos["risk"]:
                    tr = pos["best"] - s * pv["trail_atr"] * info["atr"]
                    pos["stop"] = max(pos["stop"], tr) if s > 0 else min(pos["stop"], tr)
                if pos["tp"] is not None and ((h >= pos["tp"]) if s > 0 else (l <= pos["tp"])):
                    close(pos["tp"], ts_ + 300_000, "止盈", fm["maker"]); pos = None
            k += 1
    return pd.DataFrame(trades)


def stats(df, days):
    if not len(df):
        return {"笔数": 0, "周": 0, "胜率%": 0, "净利": 0, "每笔R": 0, "回撤": 0}
    eq = df.sort_values("end").net.cumsum(); mdd = float((eq.cummax().clip(lower=0) - eq).max())
    return {"笔数": len(df), "周": round(len(df) / days * 7, 1), "胜率%": round((df.net > 0).mean() * 100),
            "净利": round(df.net.sum()), "每笔R": round(df.R.mean(), 2), "手续费": round(df.fee.sum()), "回撤": round(mdd), "去前5": round(df.net.sum() - df.net.nlargest(5).sum())}


def load(days=90, insts=None, name="", end_ms=None):
    """预计算结果按品种分别缓存在 bt/（算完一个存一个，中途停掉不会丢）；兼容旧的整包缓存。"""
    import fees
    OUT.mkdir(exist_ok=True)
    insts = insts or (("ETH-USDT-SWAP", 1.0), ("BTC-USDT-SWAP", 0.03))
    tag = f"lab2_{days}{name}{'_' + time.strftime('%Y%m%d', time.gmtime(end_ms / 1000)) if end_ms else ''}"
    whole = OUT / f"{tag}.pkl"
    data = pickle.load(open(whole, "rb")) if whole.exists() else {}
    for inst, unit in insts:
        if inst in data:
            continue
        f = OUT / f"{tag}_{inst}.pkl"
        if not f.exists():
            pickle.dump(precompute(days, ((inst, unit),), end_ms), open(f, "wb"))
        data.update(pickle.load(open(f, "rb")))          # 数据不足而跳过的品种是空字典
    for v in data.values():                  # 旧缓存可能存的是账户费率：统一改回 Lv1 标准费率
        v[5]["taker"], v[5]["maker"] = fees.DEFAULT["taker"], fees.DEFAULT["maker"]
    return data


def compare(data, variants, recent=30, total=90):
    """variants: [(名字, v)]。上月 = 最近 recent 天；验证 = 再往前 total-recent 天。"""
    end = min(d[1][-1] for d in data.values())
    cut = end - recent * D
    rows = []
    for name, v in variants:
        a = pd.concat([simulate(d, v, t_from=cut) for d in data.values()])
        b = pd.concat([simulate(d, v, t_to=cut) for d in data.values()])
        sa, sb = stats(a, recent), stats(b, total - recent)
        rows.append({"方案": name, **{f"上月{k}": x for k, x in sa.items()}, "|": "|", **{f"验证{k}": x for k, x in sb.items() if k in ("笔数", "周", "胜率%", "净利", "去前5", "回撤")}})
    return pd.DataFrame(rows)


def compound(df, frac=1.0, tz="Asia/Shanghai"):
    """按开平仓事件顺序复利：开仓时名义 = 已实现权益 × frac，平仓时权益 += 名义 × 这笔收益率（df.ret）。
    返回 (逐笔平仓后的权益曲线 DataFrame[ts, eq, 月], 最多同时持仓)。"""
    ev = sorted([(r.ts, 1, k) for k, r in df.iterrows()] + [(r.end, 0, k) for k, r in df.iterrows()], key=lambda x: (x[0], x[1]))
    eq, notional, curve, n, mx = 1.0, {}, [], 0, 0
    for ts, is_open, k in ev:
        if is_open:
            notional[k] = eq * frac; n += 1; mx = max(mx, n)
        else:
            eq += notional[k] * df.at[k, "ret"]; n -= 1; curve.append((ts, eq))
    cv = pd.DataFrame(curve, columns=["ts", "eq"])
    cv["月"] = pd.to_datetime(cv.ts, unit="ms", utc=True).dt.tz_convert(tz).dt.strftime("%Y-%m")
    return cv, mx


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 320); pd.set_option("display.max_columns", 30)
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    insts = live_insts()
    data = {i: d for i, d in load(days, insts, "_live").items() if i in dict(insts)}     # 缓存里可能有已从实盘去掉的品种
    t = lambda **k: {"sigs": [("trend", {"eff_min": 0.2, **k})]}
    V = [("v1 规则（效率≥25%，多空，固定 1 倍）", t(eff_min=0.25)),
         ("效率≥20%，多空", t()),
         ("效率≥20%，只做多", t(types={"三买"})),
         ("窄中枢≤1.5ATR，多空", t(zs_w_max=1.5)),
         ("v2：窄中枢 + 只做多", t(zs_w_max=1.5, types={"三买"})),
         ("v2 + 1.5R 后保本", t(zs_w_max=1.5, types={"三买"}, be_r=1.5)),
         ("v2 + 动态分档", t(zs_w_max=1.5, types={"三买"}, sizing="step"))]
    print(compare(data, V, total=days).to_string(index=False))
