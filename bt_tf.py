"""
v5 换信号周期的对比（控制变量：只改缠论信号所在的周期，其余规则照 v5）：15m（现在）/ 5m / 1m。
- 1H / 4H 的大方向、区间套、1H 趋势效率、TradeTrack 1H / 4H 确认都不变（复用 15m 版本算好的）；
- 中枢宽度、不追高、止损缓冲都按信号周期自己的 ATR（和 v5 在 15m 上的做法一致）；“新”买卖点 = 最近 3 根信号 K 线内确认；
- 撮合：15m / 5m 信号用 5 分钟 K 线判定止损，1m 信号用 1 分钟 K 线；每笔名义 = 权益 × 1 复利，吃单 0.05%，不计滑点。
区间：15m / 5m 比较最近 180 天；1m 数据量大，只比较最近 30 天（三者同一段）。

用法：python bt_tf.py
"""
import gzip
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import bt_ttrack as B
import bt_v5 as V5
import chan15_lab2 as L

L.SIGS["weak"] = L.sig_trend
MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000}
M15, D = L.M15, L.D
_G = {}


def _init(d):
    _G.update(d)


def task(T):
    """信号周期上 T 时刻（K 线收盘）的缠论结构与 ATR 等，字段同 chan15_lab2.task15。"""
    import chan
    import ta
    ts, df, dur = _G["ts"], _G["df"], _G["dur"]
    j = np.searchsorted(ts, T - dur, side="right")
    s = df.iloc[max(0, j - 400):j].reset_index(drop=True)
    if len(s) < 300:
        return T, None
    ch = chan.analyze(s)
    n = len(s)
    rsi = ta.rsi(s.c)
    c = s.c.values[-33:]
    eff = abs(c[-1] - c[0]) / (abs(pd.Series(c).diff()).sum() or 1)
    return T, {"fresh": [(p["type"], int(s.ts.iloc[p["i"]]), float(p["price"])) for p in ch["points"] if p["i"] >= n - 3],
               "close": float(s.c.iloc[-1]), "open": float(s.o.iloc[-1]), "high": float(s.h.iloc[-1]), "low": float(s.l.iloc[-1]),
               "atr": float(ta.atr(s).iloc[-1]), "trend15": ch["trend"], "zs": ch.get("last_zs"),
               "rsi": float(rsi.iloc[-1]), "rsi_prev": float(rsi.iloc[-2]), "eff15": float(eff)}


def raw(inst, bar):
    h = pickle.load(gzip.open(L.OUT.parent / "cache" / f"{inst}_{bar}.pkl.gz", "rb"))
    ts = np.array(sorted(h), dtype="int64")
    return ts, pd.DataFrame([h[t][:6] for t in ts], columns=["ts", "o", "h", "l", "c", "volQuote"])


def precompute(inst, tf, t0, t1):
    f = L.OUT / f"tf_{tf}_{inst}_{t0}_{t1}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    ts, df = raw(inst, tf)
    Ts = list(range(t0, t1 + 1, MS[tf]))
    w = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(w, initializer=_init, initargs=({"ts": ts, "df": df, "dur": MS[tf]},)) as ex:
        p = dict(ex.map(task, Ts, chunksize=200))
    pickle.dump((Ts, p), open(f, "wb"))
    return Ts, p


def build(base, feats, inst, tf, t0, t1):
    """把 15m 版本的 p1h / 费用 / TradeTrack 特征接到新周期的信号上；1m 信号换用 1 分钟 K 线撮合。"""
    unit, _, _, p1h, b5, fm = base[inst]
    if tf == "15m":
        Ts = [t for t in base[inst][1] if t0 <= t <= t1]; p = base[inst][2]
    else:
        Ts, p = precompute(inst, tf, t0, t1)
    f = feats.get(inst, {})
    q = {T: {**p[T], "tt": f.get(T // M15 * M15)} for T in Ts if p.get(T) is not None}   # TradeTrack 取最近一根已收盘 15m 的值
    bars = b5
    if tf == "1m":
        ts1, d1 = raw(inst, "1m")
        bars = list(zip(ts1, d1.h.values, d1.l.values, d1.c.values))
    return (unit, Ts, q, p1h, bars, fm)


def stat(data, tf, days):
    rows = []
    for inst, d in data.items():
        t = L.simulate(d, {"sigs": V5.V["v5（两者都加，当前实盘）"]}, step=MS[tf])
        if len(t):
            t["ret"] = t.net / (t.entry * d[0] * t.mult); t["fee_ret"] = t.fee / (t.entry * d[0] * t.mult); rows.append(t)
    df = pd.concat(rows).sort_values("ts").reset_index(drop=True)
    cv, mx = L.compound(df); mdd = float((1 - cv["eq"] / cv["eq"].cummax()).max())
    r = df.ret.values
    day = pd.to_datetime(df.ts, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.floor("D")
    gaps = np.diff(np.sort(df.ts.values)) / D
    hold = (df.end - df.ts) / 3_600_000
    return {"复利": f"{(cv['eq'].iloc[-1] - 1) * 100:+.0f}%", "回撤": f"{mdd:.0%}", "笔数": len(df), "每天": f"{len(df) / days:.1f}",
            "有开仓天数": f"{day.nunique() / days:.0%}", "最长空窗": f"{gaps.max():.1f} 天", "胜率": f"{(r > 0).mean():.0%}",
            "均盈 / 均亏": f"{r[r > 0].mean() * 100:+.2f}% / {r[r <= 0].mean() * 100:+.2f}%",
            "手续费合计": f"{df.fee_ret.sum() * 100:.0f}%", "持仓中位": f"{hold.median():.1f} 小时",
            "去前5": f"{(r.sum() - np.sort(r)[-5:].sum()) * 100:+.0f}%", "最多同时": mx}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 260); pd.set_option("display.unicode.east_asian_width", True)
    insts = L.live_insts()
    base = {i: d for i, d in L.load(180, insts, "_live").items() if i in dict(insts)}
    feats = B.features(base, 180)
    t1 = min(d[1][-1] for d in base.values())
    for days, tfs in ((180, ("15m", "5m")), (30, ("15m", "5m", "1m"))):
        t0 = t1 - days * D
        res = {}
        for tf in tfs:
            t = time.time()
            data = {i: build(base, feats, i, tf, t0, t1) for i in base}
            res[f"{tf} 信号"] = stat(data, tf, days)
            print(f"  {days} 天 {tf}：{res[f'{tf} 信号']['笔数']} 笔（{time.time() - t:.0f}s）", flush=True)
        print(f"\n== 最近 {days} 天（{len(base)} 个品种）==\n" + pd.DataFrame(res).T.to_string(), flush=True)
