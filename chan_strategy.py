"""
纯缠论单仓策略（不加仓）：1H 新出现的买卖点开仓，1 倍仓位。

入场：1H 刚确认的一买 / 二买 / 三买 → 做多；一卖 / 二卖 / 三卖 → 做空
      大方向（4H*0.6 + 1D*0.4 趋势分）不能明显相反（多单要求 > -阈值，空单要求 < +阈值）
止损：买点价格外 0.5 ATR(1H)（空单为卖点价格外）
止盈：chan_tp_rr=None 时取方向上最近的强结构位（最多 3 倍风险，盈亏比 < 2 或扣费后 < 1.5 不做）；
      为数字时固定几倍风险；为 0 时不设止盈，只靠止损或反向买卖点离场
离场：止盈 / 止损，或持仓中出现反向买卖点（按 1H 收盘价离场）
"""
import chan
import ta
import paper_trader as pt

LONG_PTS = {"一买", "二买", "三买"}
SHORT_PTS = {"一卖", "二卖", "三卖"}


def decide(d, cfg, unit, used):
    """d: df1h / df4h / df1d / price。used: 已经交易过的买卖点下标集合（同一个点只做一次）。"""
    df1h = d["df1h"]; n = len(df1h); price = d["price"]
    ch = chan.analyze(df1h)
    fresh = [p for p in ch["points"] if p["i"] >= n - 3 and (p["type"], int(df1h.ts.iloc[p["i"]])) not in used]
    if not fresh:
        return None, "无新的缠论买卖点"
    p = fresh[-1]
    side = "long" if p["type"] in LONG_PTS else "short"
    r1h, piv1h, _ = ta.analyze_tf(df1h, "1H", 2.0)
    r4h, piv4h, _ = ta.analyze_tf(d["df4h"], "4H", 2.0)
    r1d, piv1d, _ = ta.analyze_tf(d["df1d"], "1D", 1.5)
    trend = 0.6 * r4h["trend_score"] + 0.4 * r1d["trend_score"]
    th = cfg["trend_threshold"]
    key = (p["type"], int(df1h.ts.iloc[p["i"]]))
    if (side == "long" and trend <= -th) or (side == "short" and trend >= th):
        return None, f"{p['type']} 与大方向相反（加权趋势 {trend:+.2f}），不做", key
    a = r1h["atr"]
    stop = p["price"] - 0.5 * a if side == "long" else p["price"] + 0.5 * a
    risk = abs(price - stop)
    if (side == "long" and stop >= price) or (side == "short" and stop <= price) or risk < 0.5 * a:
        return None, f"{p['type']} 已走远或止损太近，不追", key
    tp_rr = cfg.get("chan_tp_rr")   # None：取最近强结构位；数字：固定几倍风险；0：不设止盈，只靠止损 / 反向买卖点离场
    if tp_rr is not None:
        if tp_rr == 0:
            target = price * (100 if side == "long" else 0.01)
            return {"side": side, "point": p["type"], "entry": price, "stop": stop, "target": target, "rr": 0.0},                 f"{p['type']} {'做多' if side == 'long' else '做空'} @{price:.6g} 止损{stop:.6g}，不设止盈（反向买卖点离场）", key
        target = price + tp_rr * risk if side == "long" else price - tp_rr * risk
    else:
        lv = pt.levels_1h(price, r1h, piv1h, r4h, piv4h, r1d, piv1d, df1h, d.get("walls", []),
                          [(zg, "1H中枢上沿", 2.5) for zg, zd in ch.get("zs_levels", [])] +
                          [(zd, "1H中枢下沿", 2.5) for zg, zd in ch.get("zs_levels", [])])
        if side == "long":
            obst = [x["lo"] for x in lv if x["strength"] >= cfg["min_level_strength"] and x["lo"] > price + 0.5 * a]
            target = min(obst) - 0.1 * a if obst else price + 3 * risk
            target = min(target, price + 3 * risk)
        else:
            obst = [x["hi"] for x in lv if x["strength"] >= cfg["min_level_strength"] and x["hi"] < price - 0.5 * a]
            target = max(obst) + 0.1 * a if obst else price - 3 * risk
            target = max(target, price - 3 * risk)
    reward = abs(target - price)
    fee = cfg["fee_per_unit"]
    rr = reward / risk
    net_rr = (reward * unit - fee) / (risk * unit + fee)
    if rr < 2 or net_rr < 1.5:
        return None, f"{p['type']} 盈亏比 {rr:.2f}（扣费 {net_rr:.2f}）不足，放弃", key
    return {"side": side, "point": p["type"], "entry": price, "stop": stop, "target": target, "rr": rr}, \
        f"{p['type']} {'做多' if side == 'long' else '做空'} @{price:.6g} 止损{stop:.6g} 止盈{target:.6g} RR{rr:.2f}", key


def opposite_point(df1h, side, since_ms):
    ch = chan.analyze(df1h)
    want = SHORT_PTS if side == "long" else LONG_PTS
    for p in ch["points"]:
        if p["type"] in want and int(df1h.ts.iloc[p["i"]]) >= since_ms:
            return p["type"]
    return None
