"""
v5（15m）开仓 / 离场点位优化：把其他策略的想法逐个嫁接到 v5 上（控制变量，每次只改一项），两段数据：
最近 180 天（8 个品种）/ 2025 全年（6 个品种）。每笔名义 = 权益 × 1 复利；吃单 0.05%，不计滑点。
多进程并行：任务按（时间段, 品种, 方案组）拆开，每个进程只加载自己那个品种的数据，默认用满全部 CPU 核。

开仓（额外条件，两个模块都加）：E1 15m 均线多头 / E2 1H 均线多头 / E3 信号 K 线放量 / E4 15m MACD 柱 > 0 /
     E5 15m SuperTrend 向上 / E6 1H SuperTrend 向上 / E7 随机 9-3 < 80 / E8 随机 60-10 ≥ 50 / E9、E10 回踩挂单
离场：X1 不看反向信号、15m 跌破 EMA20 走 / X2 反向信号或跌破 EMA20 / X3 反向信号或 15m SuperTrend 翻空 /
     X4 不看反向信号、1H SuperTrend 翻空走 / X5 满 5 小时平 1/3 并保本 / X6 1.5R 平 1/3 并保本 /
     X7 盈利中 9-3 上穿 80 平一半并保本 / X8 超过 8 小时仍亏就走 / X9 浮盈 2R 后跌破 EMA20 走

用法：python bt_v5_opt.py [方案名 …]      不带参数跑全部方案；带参数只跑指定方案（用于验证组合）
"""
import gzip
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import bt_top10 as T10
import bt_ttrack as B
import bt_v5 as V5
import chan15_lab2 as L
import ttrack

L.SIGS["weak"] = L.sig_trend
M15, H = L.M15, L.H
PERIODS = {"最近 180 天": (180, None, ()), "2025 全年": (365, V5.END25, ("ZEC-USDT-SWAP", "HYPE-USDT-SWAP"))}


# ---------------- 指标（15m / 1H），按收盘时刻 T 查 ----------------
def _raw(inst, bar):
    h = pickle.load(gzip.open(L.OUT.parent / "cache" / f"{inst}_{bar}.pkl.gz", "rb"))
    ts = sorted(h)
    return pd.DataFrame([h[t][:6] for t in ts], columns=["ts", "o", "h", "l", "c", "v"])


def indicators(inst):
    d = _raw(inst, "15m")
    c = d.c
    e10, e20, e50 = T10.ema(c, 10), T10.ema(c, 20), T10.ema(c, 50)
    _, st = ttrack.supertrend(d.h.values, d.l.values, c.values)
    k9, k60 = T10.stoch(d, 9, 3), T10.stoch(d, 60, 10)
    ind = pd.DataFrame({"c": c, "e10": e10, "e20": e20, "e50": e50, "vr": d.v / d.v.rolling(20).mean(),
                        "macd": T10.macd_hist(c), "st": st, "k9": k9, "k9p": k9.shift(1), "k60": k60})
    m15 = {int(t) + M15: row for t, row in zip(d.ts.values, ind.itertuples(index=False))}
    h = _raw(inst, "1H")
    he10, he20, he50 = T10.ema(h.c, 10), T10.ema(h.c, 20), T10.ema(h.c, 50)
    _, hst = ttrack.supertrend(h.h.values, h.l.values, h.c.values)
    h1 = {int(t) + H: (a > b > e, bool(s)) for t, a, b, e, s in zip(h.ts.values, he10, he20, he50, hst)}
    return m15, h1


# ---------------- 方案 ----------------
def g_ind(fn):
    def g(t, side, info, hr):
        x = info.get("ind")
        return bool(x) and fn(*x)
    return g


GATES = {
    "E1 15m 均线多头排列": lambda m, h: m.e10 > m.e20 > m.e50,
    "E2 1H 均线多头排列": lambda m, h: h[0],
    "E3 信号K线放量 ≥1.5 倍": lambda m, h: m.vr >= 1.5,
    "E4 15m MACD 柱 > 0": lambda m, h: m.macd > 0,
    "E5 15m SuperTrend 向上": lambda m, h: bool(m.st),
    "E6 1H SuperTrend 向上": lambda m, h: h[1],
    "E7 随机 9-3 < 80（不追超买）": lambda m, h: m.k9 < 80,
    "E8 随机 60-10 ≥ 50（慢线强）": lambda m, h: m.k60 >= 50,
}
ENTRY = {"E9 回踩挂单：收盘价下方 0.3 ATR": {"entry": "dip", "dip_atr": 0.3},
         "E10 回踩挂单：收盘价下方 0.15 ATR": {"entry": "dip", "dip_atr": 0.15}}


def x_close(fn, why):
    def f(info, pos, t):
        x = info.get("ind")
        return ("close", why) if x and fn(x[0], x[1], info, pos) else None
    return f


def x_part(fn, frac):
    def f(info, pos, t):
        x = info.get("ind")
        return ("part", frac) if x and fn(x[0], x[1], info, pos) else None
    return f


EXITS = {
    "X1 不看反向信号，15m 跌破 EMA20 走": ({"opp": False}, x_close(lambda m, h, i, p: m.c < m.e20, "跌破EMA20")),
    "X2 反向信号或 15m 跌破 EMA20": ({}, x_close(lambda m, h, i, p: m.c < m.e20, "跌破EMA20")),
    "X3 反向信号或 15m SuperTrend 翻空": ({}, x_close(lambda m, h, i, p: not m.st, "SuperTrend翻空")),
    "X4 不看反向信号，1H SuperTrend 翻空走": ({"opp": False}, x_close(lambda m, h, i, p: not h[1], "1H SuperTrend翻空")),
    "X5 满 5 小时平 1/3 并保本": ({}, x_part(lambda m, h, i, p: p["bars"] >= 20, 1 / 3)),
    "X6 1.5R 平 1/3 并保本": ({"part_r": 1.5, "part_frac": 1 / 3}, None),
    "X7 盈利中 9-3 上穿 80 平一半并保本": ({}, x_part(lambda m, h, i, p: m.c > p["entry"] and m.k9p < 80 <= m.k9, 0.5)),
    "X8 超过 8 小时仍亏就走": ({}, x_close(lambda m, h, i, p: p["bars"] >= 32 and m.c < p["entry"], "超时仍亏")),
    "X9 浮盈到 2R 后跌破 EMA20 走": ({}, x_close(lambda m, h, i, p: (p["best"] - p["entry"]) >= 2 * p["risk"] and m.c < m.e20, "2R后跌破EMA20")),
}
NAMES = ["v5（现在）"] + list(GATES) + list(ENTRY) + list(EXITS)


def sigs_for(names):
    """names：要叠加的改动（可多个，用于组合验证）。"""
    strong = {**V5.STRONG, **V5.CH}
    weak = {**V5.LOW, **V5.CH}
    gates, fns = [], []
    for n in names:
        if n in GATES:
            gates.append(g_ind(GATES[n]))
        elif n in ENTRY:
            strong.update(ENTRY[n]); weak.update(ENTRY[n])
        elif n in EXITS:
            kw, fn = EXITS[n]
            strong.update(kw); weak.update(kw)
            if fn:
                fns.append(fn)
    if fns:
        ef = lambda info, pos, t: next((a for a in (f(info, pos, t) for f in fns) if a), None)
        strong["exit_fn"] = weak["exit_fn"] = ef
    g_all = (lambda *a: all(g(*a) for g in gates)) if gates else None
    tt = V5.tt_gate(40, 20)
    if g_all:
        strong["gate"] = g_all
    weak["gate"] = (lambda *a: tt(*a) and g_all(*a)) if g_all else tt
    return [("trend", strong), ("weak", weak)]


# ---------------- 并行任务 ----------------
def job(args):
    pname, inst, combos = args
    days, end, _ = PERIODS[pname]
    unit = dict(L.live_insts())[inst]
    d = L.load(days, ((inst, unit),), "_live", end)[inst]
    feats = B.features({inst: d}, days, end).get(inst, {})
    m15, h1 = indicators(inst)
    hkey = lambda t: t // H * H
    un, Ts, p15, p1h, b5, fm = d
    q = {T: {**p15[T], "tt": feats.get(T), "ind": (m15.get(T), h1.get(hkey(T)))} for T in Ts
         if p15.get(T) is not None and m15.get(T) is not None and h1.get(hkey(T)) is not None}
    data = (un, Ts, q, p1h, b5, fm)
    out = []
    for label, names in combos:
        t = L.simulate(data, {"sigs": sigs_for([n for n in names if n != "v5（现在）"])})
        if len(t):
            out.append(pd.DataFrame({"ts": t.ts, "end": t.end, "ret": t.net / (t.entry * un * t.mult), "方案": label, "inst": inst}))
    return pname, pd.concat(out) if out else pd.DataFrame()


def stat(df, days):
    x = df.sort_values("ts").reset_index(drop=True)
    cv, mx = L.compound(x)
    mdd = float((1 - cv["eq"] / cv["eq"].cummax()).max())
    r = x.ret.values
    return {"复利": (cv["eq"].iloc[-1] - 1) * 100, "回撤": mdd * 100, "笔数": len(x), "胜率": (r > 0).mean() * 100,
            "每周": len(x) / days * 7, "去前5": (r.sum() - np.sort(r)[-5:].sum()) * 100}


def run(combos, workers=None):
    workers = workers or os.cpu_count()
    tasks = []
    for pname, (days, end, skip) in PERIODS.items():
        for inst, _ in L.live_insts():
            if inst in skip:
                continue
            k = max(1, len(combos) // 2)          # 每个品种拆成两组方案，任务数 ≈ 核数
            tasks += [(pname, inst, combos[i:i + k]) for i in range(0, len(combos), k)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(job, tasks))
    print(f"{len(tasks)} 个任务，{workers} 个进程并行，用时 {time.time() - t0:.0f}s", flush=True)
    rows = {}
    for pname, (days, _, _) in PERIODS.items():
        df = pd.concat([r for p, r in res if p == pname and len(r)])
        for label, _ in combos:
            rows.setdefault(label, {}).update({f"{pname} {k}": v for k, v in stat(df[df["方案"] == label], days).items()})
    return pd.DataFrame(rows).T


def show(tb):
    base = tb.iloc[0]
    out = pd.DataFrame(index=tb.index)
    for p in PERIODS:
        out[f"{p} 复利"] = tb[f"{p} 复利"].map(lambda v: f"{v:+.0f}%")
        out[f"{p} 回撤"] = tb[f"{p} 回撤"].map(lambda v: f"{v:.0f}%")
        out[f"{p} 胜率/每周"] = [f"{a:.0f}% / {b:.1f}" for a, b in zip(tb[f"{p} 胜率"], tb[f"{p} 每周"])]
    better = [all(r[f"{p} 复利"] > base[f"{p} 复利"] and r[f"{p} 回撤"] <= base[f"{p} 回撤"] + 1 for p in PERIODS) for _, r in tb.iterrows()]
    out["两段都更好"] = ["✅" if b else "" for b in better]
    out.iloc[0, -1] = "基准"
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 280); pd.set_option("display.unicode.east_asian_width", True)
    if len(sys.argv) > 1:          # 组合验证：python bt_v5_opt.py "E5 15m SuperTrend 向上" "X2 反向信号或 15m 跌破 EMA20"
        combos = [("v5（现在）", ["v5（现在）"])] + [(n, [n]) for n in sys.argv[1:]] + [("组合：" + " + ".join(n.split()[0] for n in sys.argv[1:]), sys.argv[1:])]
    else:
        combos = [(n, [n]) for n in NAMES]
    print(show(run(combos)).to_string())
