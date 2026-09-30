"""
v5 与 Christian 日线整理突破的结合方式（控制变量，每次只改一项），两段数据：最近 180 天（8 个品种）/ 2025 全年（6 个品种）。
组合按“每笔名义 = 开仓时已实现权益 × 1”复利；吃单 0.05%，不计滑点。

A 两个模块一起开：①谁先开谁占着（同一品种同时只有一笔，多轮模拟近似）②Christian 优先（它持仓时 v5 不在该品种开）
                  ③允许同一品种同时两笔 ④Christian 优先 + Christian 半仓
B 用 Christian 的日线趋势过滤 v5：⑤只在日线收盘 > EMA20 且 EMA10 > EMA20 时做 v5
C 用 Christian 的离场替换 v5 的离场：⑥不看反向信号，日线收盘跌破 EMA20 才走 ⑦跌破 EMA10 才走 ⑧反向信号或跌破 EMA20 先到先走

用法：python bt_v5_christian.py
"""
import sys

import numpy as np
import pandas as pd

import bt_top10 as T
import bt_ttrack as B
import bt_v5 as V5
import chan15_lab2 as L

TZ = "Asia/Shanghai"
L.SIGS["weak"] = L.sig_trend


def windows(tr):
    return [(int(a), int(b)) for a, b in zip(tr.entry_ts if "entry_ts" in tr else tr.ts, tr.exit_ts if "exit_ts" in tr else tr.end)]


def block_gate(wins):
    arr = sorted(wins)
    def g(t, side, info, hr):
        return not any(a <= t < b for a, b in arr)
    return g


def both(g1, g2):
    return lambda *a: g1(*a) and g2(*a)


def daily_trend_gate(D):
    e10, e20, end, c = ema(D.c, 10).values, ema(D.c, 20).values, D.end.values, D.c.values
    def g(t, side, info, hr):
        j = np.searchsorted(end, t, side="right") - 1     # t 时刻已收盘的最后一根日线
        return j >= 20 and c[j] > e20[j] and e10[j] > e20[j]
    return g


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def v5_sigs(gate=None, **kw):
    strong = {**V5.STRONG, **V5.CH, **kw}
    weak = {**V5.LOW, **V5.CH, **kw, "gate": V5.tt_gate(40, 20)}
    if gate:
        strong["gate"] = gate
        weak["gate"] = both(weak["gate"], gate)
    return [("trend", strong), ("weak", weak)]


def run_v5(data, per_inst):
    rows = []
    for inst, d in data.items():
        t = L.simulate(d, {"sigs": per_inst(inst)})
        if len(t):
            t["ret"] = t.net / (t.entry * d[0] * t.mult); t["inst"] = inst; t["mod"] = "v5"
            rows.append(t[["ts", "end", "ret", "inst", "mod"]])
    return pd.concat(rows) if rows else pd.DataFrame(columns=["ts", "end", "ret", "inst", "mod"])


def run_ch(ctx, start, stop, blocked=None, size=1.0):
    rows = []
    for inst, (D, H1, m5) in ctx.items():
        tr = T.m10_christian(D, H1, m5, start, (blocked or {}).get(inst, ()))
        tr = tr[(tr.entry_ts >= start) & (tr.entry_ts < stop)] if len(tr) else tr
        if len(tr):
            rows.append(pd.DataFrame({"ts": tr.entry_ts, "end": tr.exit_ts, "ret": tr.ret * size, "inst": inst, "mod": "Christian"}))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["ts", "end", "ret", "inst", "mod"])


def by_inst(tr):
    return {i: windows(g) for i, g in tr.groupby("inst")}


def stats(tr, days):
    x = tr.sort_values("ts").reset_index(drop=True)
    cv, mx = L.compound(x)
    mdd = float((1 - cv["eq"] / cv["eq"].cummax()).max())
    r = x.ret.values
    day = pd.to_datetime(x.ts, unit="ms", utc=True).dt.tz_convert(TZ).dt.floor("D")
    gaps = np.diff(np.sort(x.ts.values)) / 86_400_000
    return {"复利": f"{(cv['eq'].iloc[-1] - 1) * 100:+.0f}%", "回撤": f"{mdd:.0%}", "笔数": len(x),
            "其中 Christian": int((x["mod"] == "Christian").sum()), "胜率": f"{(r > 0).mean():.0%}",
            "每周": f"{len(x) / days * 7:.1f}", "有开仓天数": f"{day.nunique() / days:.0%}", "最长空窗": f"{gaps.max():.0f} 天",
            "去前5": f"{(r.sum() - np.sort(r)[-5:].sum()) * 100:+.0f}%", "最多同时": mx}


def period(days, end, skip):
    insts = tuple(x for x in L.live_insts() if x[0] not in skip)
    data = {i: d for i, d in L.load(days, insts, "_live", end).items() if i in dict(insts)}
    data = B.attach(data, B.features(data, days, end))
    t0 = min(d[1][0] for d in data.values()); t1 = max(d[1][-1] for d in data.values())
    ctx = {i: (T.load("1D", i), T.load("1H", i), T.load("5m", i)) for i in data}
    dexit = {}
    for i, (D, _, _) in ctx.items():
        e = {n: ema(D.c, n).values for n in (10, 20)}
        dexit[i] = {n: {int(t): (float(c), float(v)) for t, c, v in zip(D.end.values, D.c.values, e[n])} for n in (10, 20)}
    res = {}
    base = run_v5(data, lambda i: v5_sigs())
    res["v5（现在）"] = base
    ch = run_ch(ctx, t0, t1)
    # ① 谁先开谁占着：v5 自由 → Christian 避开 v5 → v5 避开 Christian
    ch1 = run_ch(ctx, t0, t1, by_inst(base))
    w = by_inst(ch1)
    res["① 两者都做·谁先开谁占着"] = pd.concat([run_v5(data, lambda i: v5_sigs(block_gate(w.get(i, [])))), ch1])
    wc = by_inst(ch)
    v5c = run_v5(data, lambda i: v5_sigs(block_gate(wc.get(i, []))))
    res["② 两者都做·Christian 优先"] = pd.concat([v5c, ch])
    res["③ 两者都做·同一品种可同时两笔"] = pd.concat([base, ch])
    res["④ Christian 优先 + Christian 半仓"] = pd.concat([v5c, run_ch(ctx, t0, t1, size=0.5)])
    res["⑤ v5 只在日线多头时做"] = run_v5(data, lambda i: v5_sigs(daily_trend_gate(ctx[i][0])))
    res["⑥ v5 离场改为日线跌破 EMA20"] = run_v5(data, lambda i: v5_sigs(opp=False, daily_exit=dexit[i][20]))
    res["⑦ v5 离场改为日线跌破 EMA10"] = run_v5(data, lambda i: v5_sigs(opp=False, daily_exit=dexit[i][10]))
    res["⑧ 反向信号或跌破 EMA20 先到先走"] = run_v5(data, lambda i: v5_sigs(daily_exit=dexit[i][20]))
    res["（参考）只做 Christian"] = ch
    return {k: stats(v, days) for k, v in res.items()}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 280); pd.set_option("display.unicode.east_asian_width", True)
    for pname, (days, end, skip) in V5.PERIODS.items():
        print(f"\n== {pname} ==\n" + pd.DataFrame(period(days, end, skip)).T.to_string(), flush=True)
