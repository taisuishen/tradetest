"""
15 分钟级缠论单仓策略实验台（控制变量法）。

1. 并行预计算：每根 15m 收盘时刻的缠论结构（新确认的买卖点、最近的笔端点、ATR），
   以及每根 1H 收盘时刻的大方向趋势分（1H*0.6 + 4H*0.4）和 1H 缠论方向。
2. 在主进程里按时间顺序模拟每个方案（只改一个变量），止损 / 止盈用 5m K 线判定（同根先判止损）。

基准：所有买卖点都做、不加过滤、止损 = 买卖点价外 0.5 ATR(15m)、不设止盈（反向买卖点或止损离场）、1 倍仓位、每笔 2U。
用法：python chan15_lab.py [天数=60]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import os

import pandas as pd

M15, H = 900_000, 3_600_000
LONG = {"一买", "二买", "三买", "盘背买"}
SHORT = {"一卖", "二卖", "三卖", "盘背卖"}
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
    fresh = [(p["type"], int(s.ts.iloc[p["i"]]), float(p["price"])) for p in ch["points"] if p["i"] >= n - 3]
    bis = ch["bis"]
    last_bot = next(((b["end"], b["low"]) for b in reversed(bis) if not b["up"]), None)   # 最近一笔向下笔的终点（底）
    last_top = next(((b["end"], b["high"]) for b in reversed(bis) if b["up"]), None)
    return T, {"fresh": fresh, "close": float(s.c.iloc[-1]), "atr": float(ta.atr(s).iloc[-1]),
               "bot": (int(s.ts.iloc[last_bot[0]]), last_bot[1]) if last_bot else None,
               "top": (int(s.ts.iloc[last_top[0]]), last_top[1]) if last_top else None}


def task1h(T):
    import chan, ta
    d1, d4 = _G["df1h"], _G["df4h"]
    s1 = d1[d1.ts + H <= T].iloc[-400:].reset_index(drop=True)
    s4 = d4[d4.ts + 4 * H <= T].iloc[-300:].reset_index(drop=True)
    if len(s1) < 300 or len(s4) < 200:
        return T, None
    r1, _, _ = ta.analyze_tf(s1, "1H", 2.0)
    r4, _, _ = ta.analyze_tf(s4, "4H", 2.0)
    return T, {"trend": 0.6 * r1["trend_score"] + 0.4 * r4["trend_score"], "bias": chan.bias(chan.analyze(s1))}


def precompute(inst, days):
    import okx_client as ox
    df15 = ox.candles(inst, "15m", 400 + days * 96 + 4)
    df1h = ox.candles(inst, "1H", 400 + days * 24 + 2)
    df4h = ox.candles(inst, "4H", 300 + days * 6 + 2)
    df5 = ox.candles(inst, "5m", days * 288 + 12)
    end = int(df15.ts.iloc[-1]) + M15
    Ts = list(range(end - days * 96 * M15, end + 1, M15))
    Th = sorted({t // H * H for t in Ts})
    w = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(w, initializer=_init, initargs=({"df15": df15, "df1h": df1h, "df4h": df4h},)) as ex:
        p15 = dict(ex.map(task15, Ts, chunksize=40))
        p1h = dict(ex.map(task1h, Th, chunksize=10))
    return Ts, p15, p1h, list(zip(df5.ts.astype("int64"), df5.h, df5.l))


def simulate(Ts, p15, p1h, b5, unit, v, fee=2.0):
    """v: 方案参数
    types 允许的买卖点；trend_filter 大方向不相反；trend_min 大方向至少同向到多少（设了就替代 trend_filter）；
    nest 1H 缠论方向不相反；nest_strict 1H 缠论方向必须同向；stop_buf 止损缓冲（ATR）；
    exit 'opp' / 'trail'；tp_rr 固定止盈倍数；be_r 浮盈几倍风险后止损移到保本（开仓价 + 手续费）；
    part_r / part_frac 到几倍风险先平掉多少比例，part_be 分批后剩余仓位是否移到保本"""
    trades, pos, used, j = [], None, set(), 0
    for T in Ts:
        info = p15.get(T)
        if not info:
            continue
        hr = p1h.get(T // H * H)       # 截至该整点已收盘的 1H 信息（不偷看未来）
        # 1) 反向买卖点离场
        if pos:
            want = SHORT if pos["side"] > 0 else LONG
            opp = next((t for t, ts_, _ in info["fresh"] if t in want and ts_ >= pos["ts"]), None)
            if opp:
                trades.append(close(pos, info["close"], T, f"反向{opp}", unit, fee)); pos = None
        # 2) 跟踪止损：移到最近一个确认的反向笔端点外 0.2 ATR（只往有利方向移）
        if pos and v["exit"] == "trail":
            if pos["side"] > 0 and info["bot"] and info["bot"][0] > pos["ts"]:
                pos["stop"] = max(pos["stop"], info["bot"][1] - 0.2 * info["atr"])
            if pos["side"] < 0 and info["top"] and info["top"][0] > pos["ts"]:
                pos["stop"] = min(pos["stop"], info["top"][1] + 0.2 * info["atr"])
        # 3) 开仓
        if pos is None:
            for t, ts_, px in reversed(info["fresh"]):
                if t not in v["types"] or (t, ts_) in used:
                    continue
                used.add((t, ts_))
                side = 1 if t in LONG else -1
                if v.get("trend_min") is not None:
                    if not hr or hr["trend"] * side < v["trend_min"]:
                        continue
                elif v["trend_filter"] and hr and hr["trend"] * side <= -1.0:
                    continue
                if v["nest"] and hr and hr["bias"] * side < 0:
                    continue
                if v.get("nest_strict") and (not hr or hr["bias"] * side <= 0):
                    continue
                entry = info["close"]
                stop = px - side * v["stop_buf"] * info["atr"]
                if (stop - entry) * side >= 0:
                    continue                                  # 价格已回到买卖点另一侧，结构失效
                risk = abs(entry - stop)
                tp = entry + side * v["tp_rr"] * risk if v.get("tp_rr") else None
                pos = {"side": side, "type": t, "entry": entry, "stop": stop, "tp": tp, "ts": T, "risk": risk,
                       "rem": 1.0, "realized": 0.0, "be_done": False, "part_done": False}
                break
        if not pos:
            continue
        # 4) 5m K 线：止损 → 分批止盈 → 保本 → 止盈（同根先判止损，偏保守）
        while j < len(b5) and b5[j][0] < T:
            j += 1
        k = j
        while k < len(b5) and b5[k][0] < T + M15:
            ts_, h, l = b5[k]
            s = pos["side"]
            best = h if s > 0 else l
            if (l <= pos["stop"]) if s > 0 else (h >= pos["stop"]):
                why = "保本" if pos["be_done"] and abs(pos["stop"] - pos["entry"]) < pos["risk"] * 0.5 else "止损"
                trades.append(close(pos, pos["stop"], ts_ + 300_000, why, unit, fee)); pos = None; break
            be_px = pos["entry"] + s * fee / unit
            if v.get("part_r") and not pos["part_done"] and (best - pos["entry"]) * s >= v["part_r"] * pos["risk"]:
                px_ = pos["entry"] + s * v["part_r"] * pos["risk"]
                f = v.get("part_frac", 0.5)
                pos["realized"] += (px_ - pos["entry"]) * s * unit * f
                pos["rem"] -= f; pos["part_done"] = True
                if v.get("part_be", True):
                    pos["stop"] = max(pos["stop"], be_px) if s > 0 else min(pos["stop"], be_px); pos["be_done"] = True
            if v.get("be_r") and not pos["be_done"] and (best - pos["entry"]) * s >= v["be_r"] * pos["risk"]:
                pos["stop"] = max(pos["stop"], be_px) if s > 0 else min(pos["stop"], be_px); pos["be_done"] = True
            if pos["tp"] and ((h >= pos["tp"]) if s > 0 else (l <= pos["tp"])):
                trades.append(close(pos, pos["tp"], ts_ + 300_000, "止盈", unit, fee)); pos = None; break
            k += 1
    return pd.DataFrame(trades)


def close(pos, px, ts_, why, unit, fee):
    gross = pos.get("realized", 0.0) + (px - pos["entry"]) * pos["side"] * unit * pos.get("rem", 1.0)
    return {"ts": pos["ts"], "type": pos["type"], "side": "多" if pos["side"] > 0 else "空",
            "start": datetime.fromtimestamp(pos["ts"] / 1000).strftime("%m-%d %H:%M"),
            "end": datetime.fromtimestamp(ts_ / 1000).strftime("%m-%d %H:%M"),
            "entry": round(pos["entry"], 2), "exit": round(px, 2), "result": why, "net": round(gross - fee, 2)}


def stats(df):
    if not len(df):
        return {"笔数": 0, "胜率%": 0, "净利U": 0, "盈亏因子": 0, "最大回撤U": 0, "均盈": 0, "均亏": 0}
    w = df[df.net > 0]; l = df[df.net <= 0]
    eq = df.net.cumsum(); mdd = float((eq.cummax().clip(lower=0) - eq).max())
    return {"笔数": len(df), "胜率%": round(len(w) / len(df) * 100, 1), "净利U": round(df.net.sum(), 1),
            "盈亏因子": round(w.net.sum() / -l.net.sum(), 2) if len(l) and l.net.sum() else float("inf"),
            "最大回撤U": round(mdd, 1), "均盈": round(w.net.mean(), 1) if len(w) else 0, "均亏": round(l.net.mean(), 1) if len(l) else 0}


BASE = {"types": LONG | SHORT, "trend_filter": False, "nest": False, "stop_buf": 0.5, "exit": "opp", "tp_rr": None}
VARIANTS = [
    ("基准：全部买卖点、无过滤、反向信号离场", {}),
    ("只改：不做盘整背驰（原三类买卖点）", {"types": (LONG | SHORT) - {"盘背买", "盘背卖"}}),
    ("只改：只做三类买卖点", {"types": {"三买", "三卖"}}),
    ("只改：只做一类/二类/盘背（背驰类）", {"types": {"一买", "二买", "盘背买", "一卖", "二卖", "盘背卖"}}),
    ("只改：加大方向过滤（1H/4H 趋势分）", {"trend_filter": True}),
    ("只改：加区间套（1H 缠论方向不相反）", {"nest": True}),
    ("只改：止损缓冲 1.0 ATR", {"stop_buf": 1.0}),
    ("只改：止盈 2 倍风险", {"tp_rr": 2}),
    ("只改：止盈 3 倍风险", {"tp_rr": 3}),
    ("只改：笔端点跟踪止损", {"exit": "trail"}),
    ("组合：区间套 + 大方向过滤", {"nest": True, "trend_filter": True}),
    ("组合：区间套 + 只做三类", {"nest": True, "types": {"三买", "三卖"}}),
    ("组合：区间套 + 大方向 + 只做三类", {"nest": True, "trend_filter": True, "types": {"三买", "三卖"}}),
]

MAIN = {"nest": True, "trend_filter": True, "types": {"三买", "三卖"}}   # 当前主策略
OPT_VARIANTS = [
    ("主策略（区间套+大方向+只做三类）", {}),
    ("只改：浮盈 1R 后移到保本", {"be_r": 1.0}),
    ("只改：浮盈 1.5R 后移到保本", {"be_r": 1.5}),
    ("只改：1.5R 先平一半，剩余保本", {"part_r": 1.5, "part_frac": 0.5}),
    ("只改：2R 先平一半，剩余保本", {"part_r": 2.0, "part_frac": 0.5}),
    ("只改：1R 先平三成，剩余保本", {"part_r": 1.0, "part_frac": 0.3}),
    ("只改：大方向必须同向（趋势分≥0.5）", {"trend_min": 0.5}),
    ("只改：大方向必须同向（趋势分≥1）", {"trend_min": 1.0}),
    ("只改：1H 缠论方向必须同向", {"nest_strict": True}),
]


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    suite = sys.argv[2] if len(sys.argv) > 2 else "base"
    variants = [(n, {**MAIN, **d}) for n, d in OPT_VARIANTS] if suite == "opt" else VARIANTS
    rows = []
    for inst, unit in (("ETH-USDT-SWAP", 1.0), ("BTC-USDT-SWAP", 0.03)):
        t0 = datetime.now()
        Ts, p15, p1h, b5 = precompute(inst, days)
        mid = Ts[len(Ts) // 2]
        print(f"{inst} 预计算完成，用时 {(datetime.now() - t0).seconds}s；前半段截至 {datetime.fromtimestamp(mid / 1000):%m-%d}", flush=True)
        for name, delta in variants:
            df = simulate(Ts, p15, p1h, b5, unit, {**BASE, **delta})
            a = df[df.ts < mid] if len(df) else df
            b = df[df.ts >= mid] if len(df) else df
            sa, sb, sall = stats(a), stats(b), stats(df)
            rows.append({"合约": inst.split("-")[0], "方案": name, "全程胜率%": sall["胜率%"],
                         "前半笔数": sa["笔数"], "前半胜率%": sa["胜率%"], "前半净利": sa["净利U"],
                         "后半笔数": sb["笔数"], "后半胜率%": sb["胜率%"], "后半净利": sb["净利U"],
                         "全程净利": sall["净利U"], "最大回撤": sall["最大回撤U"]})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))
    tot = res.groupby("方案", sort=False)[["前半净利", "后半净利", "全程净利"]].sum()
    print("\n两个合约合计：\n" + tot.to_string())
    res.to_csv(f"bt/chan15_lab_{suite}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
