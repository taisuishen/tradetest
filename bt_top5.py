"""
投机实验室播放量前 5 的方法（docs/投机实验室_热门5.md）：单独回测，以及分别和 v5 结合（控制变量，每次只改一项）。
服务器 8 个品种、最近 180 天；只做多（实盘只做多）；每笔名义 = 权益 × 1 复利；吃单 0.05%，不计滑点；止损止盈用 5 分钟 K 线判定。
按品种分进程并行，默认用满全部 CPU 核。

单独回测：
  1 半木夏 MACD(13,34) 连续背离：零轴下方连续 2（或 3）个柱体波谷、后一个比前一个至少浅 30%、对应 K 线低点更低；
    柱子回升的第一根收盘入场，止损 = 这根最低价 − ATR13，止盈 2R（另测 1R）。15m / 1H。
  2 Kristjan：①日线突破整理区，按突破日收盘入场（要求 EMA10 > 20 > 50），止损在突破 K 线下方；
    ②横盘 60 ~ 120 天后 4 小时放巨量（≥ 日均量）突破，止损在最近 12 根 4 小时最低点；离场同 Christian（第 5 天平 1/3 并保本，
    跌破日线 EMA10 平 1/3，跌破 EMA20 全平）。参考：bt_top10 里 1 小时收盘入场的版本。
  3 维加斯隧道 3.0：隧道 EMA144/169 在 EMA576/676 上方且 EMA12 在隧道上方，价格回踩隧道（最低价 ≤ 隧道上沿 +0.1%）收盘入场，
    每段趋势最多进 2 次；硬止损 EMA676 下方 0.2 ATR；1R 平一半；收盘跌破隧道时放量（成交量震荡 > 144）就走，
    否则 24 小时内回不到隧道上方也走；EMA12 跌破隧道下沿时剩余全平。15m / 1H。
  4 Bit 浪浪：只在“夏天”（BTC 日线收盘 > EMA20 > EMA50 且 20 天涨幅 > 8%）、并且本品种日线 > EMA20 > EMA50 时做；
    15 分钟起稳拐点：高点 H0 → 回落 L1 → 反弹 H1（< H0）→ 再回落 L2（≥ L1，跌不下去）→ 收盘突破 H1 入场；止损 L2 下方 0.2 ATR，止盈 3R。
  5 三率全优：Normalized MACD（13/21，归一化 50，信号 WMA9）8 根内金叉 + RSI21 上穿其 SMA55（关键 K 线）+ 收盘 > MA13，收盘入场；
    止损近 10 根最低点（关键 K 线振幅 > 1.5 ATR 时改为关键 K 线最低价）；1R 平一半，其余 Normalized MACD 死叉离场。1H / 15m。
和 v5 结合：
  A 加一个模块：两者都做，同一品种同时只有一笔，谁先开谁占着（v5 自由 → 新方法避开 v5 → v5 避开新方法）
  B 开仓过滤：1H MACD(13,34) 柱 > 0 / 日线 EMA10 > 20 > 50 / 1H 或 15m 维加斯多头排列 / BTC 夏天 / 1H Normalized MACD 与 RSI 同时偏多
  C 离场：止盈 2R / 反向信号或日线跌破 EMA20 / 反向信号或 15m 跌破隧道下沿 / 反向信号或 1H Normalized MACD 死叉

用法：python bt_top5.py [2025]      默认最近 180 天；加 2025 跑 2025 全年（6 个品种）验证
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import bt_top10 as T10
import bt_ttrack as B
import bt_v5 as V5
import chan15_lab2 as L

L.SIGS["weak"] = L.sig_trend
M15, H, D = L.M15, L.H, L.D
MS = T10.MS


class BBook(T10.Book):
    """带“占用时段”的撮合：和其他模块共用一个品种仓位时，这些时段内不开新仓。"""

    def __init__(self, m5, blocked=()):
        super().__init__(m5)
        self.blocked = sorted(blocked)

    def open(self, ts, *a, **k):
        if any(x <= ts < y for x, y in self.blocked):
            return False
        super().open(ts, *a, **k)
        return True


def rsi(c, n):
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: (x * w).sum() / w.sum(), raw=True)


def nmacd(c):
    """Normalized MACD（TradingView 的 glaz 版本）：快 13、慢 21、归一化 50、信号 WMA9。返回 (线, 信号线)。"""
    sh, lon = T10.ema(c, 13), T10.ema(c, 21)
    ratio = np.minimum(sh, lon) / np.maximum(sh, lon)
    mac = pd.Series(np.where(sh > lon, 2 - ratio, ratio) - 1, index=c.index)
    lo, hi = mac.rolling(50).min(), mac.rolling(50).max()
    norm = (mac - lo) / (hi - lo + 1e-6) * 2 - 1
    return norm, wma(norm, 9)


def macd_hist(c, f=13, s=34, sig=9):
    m = T10.ema(c, f) - T10.ema(c, s)
    return m - T10.ema(m, sig)


def vegas(X):
    e = {n: T10.ema(X.c, n) for n in (12, 144, 169, 576, 676)}
    top, bot = np.maximum(e[144], e[169]), np.minimum(e[144], e[169])
    ok = (bot > np.maximum(e[576], e[676])) & (e[12] > top)
    ev = T10.ema(X.v, 14)
    vo = (X.v - ev) / ev * 100
    return e, top, bot, ok, vo


def trades_df(b):
    return pd.DataFrame(b.trades) if b.trades else pd.DataFrame(columns=["entry_ts", "exit_ts", "ret"])


def in_window(t, t0, t1):
    return t0 <= t < t1


# ---------------- 1 半木夏 MACD 连续背离 ----------------
def m1_bmx(X, m5, t0, t1, n=2, rr=2.0, d=0.3, blocked=()):
    hist = macd_hist(X.c).values
    a13 = T10.atr(X, 13).values
    lo, cl, end = X.l.values, X.c.values, X.end.values
    b = BBook(m5, blocked); tr = []
    for i in range(40, len(X)):
        if b.pos:
            b.walk(int(X.ts.iloc[i]), int(end[i])); continue
        if hist[i] >= 0:
            tr = []; continue
        if hist[i - 2] > hist[i - 1] < hist[i]:                 # i-1 是波谷，i 是柱子回升（实心转空心）的第一根
            tr.append((hist[i - 1], min(lo[i - 2], lo[i - 1])))
            if len(tr) >= n and in_window(int(end[i]), t0, t1):
                ok = all(abs(tr[-k][0]) < abs(tr[-k - 1][0]) * (1 - d) and tr[-k][1] < tr[-k - 1][1] for k in range(1, n))
                stop = lo[i] - a13[i]
                if ok and stop < cl[i]:
                    b.open(int(end[i]), cl[i], 1, stop, cl[i] + rr * (cl[i] - stop), tag="半木夏背离")
    return trades_df(b)


# ---------------- 2 Kristjan ----------------
def _christian_exits(b, D, i):
    """同 bt_top10.m10_christian 的离场：第 5 天平 1/3 并保本；跌破 EMA10 平 1/3；跌破 EMA20 全平。返回是否已平完。"""
    p, day = b.pos, D.iloc[i]
    p["days"] += 1
    if p["days"] == 5 and p["rem"] > 0.99:
        b.part(day.c, 1 / 3, int(day.end), "第5天平1/3"); p["stop"], p["be"] = p["entry"], True
    if b.pos and day.c < day.e10 and p["rem"] > 0.34 and not p.get("cut10"):
        b.part(day.c, 1 / 3, int(day.end), "跌破EMA10"); p["cut10"] = True
    if b.pos and day.c < day.e20:
        b.close(day.c, int(day.end), "跌破EMA20")
    return b.pos is None


def _base(D, i):
    for K in range(10, 41):
        w = D.iloc[i - K:i]
        Hh = w.h.max()
        prior_low = D.iloc[max(0, i - K - 66):i - K].l.min()
        half, third = K // 2, max(3, K // 3)
        if not (1.30 <= Hh / prior_low <= 2.0) or w.l.iloc[half:].min() <= w.l.iloc[:half].min():
            continue
        if (w.h.iloc[-third:].max() - w.l.iloc[-third:].min()) >= (w.h.iloc[:third].max() - w.l.iloc[:third].min()):
            continue
        return Hh
    return None


def m2_kris_daily(D, m5, t0, t1, blocked=()):
    D = D.copy(); D["e10"], D["e20"], D["e50"] = T10.ema(D.c, 10), T10.ema(D.c, 20), T10.ema(D.c, 50)
    b = BBook(m5, blocked)
    for i in range(110, len(D)):
        day = D.iloc[i]
        if b.pos:
            if not b.walk(int(day.ts), int(day.end)):
                _christian_exits(b, D, i)
            continue
        if not in_window(int(day.end), t0, t1) or not (day.e10 > day.e20 > day.e50):
            continue
        base = _base(D, i)
        if base is not None and day.c > base > D.c.iloc[i - 1]:
            b.open(int(day.end), day.c, 1, day.l * 0.999, tag="日线突破", days=0)
    return trades_df(b)


def m2_kris_accum(D, H4, m5, t0, t1, blocked=()):
    D = D.copy(); D["e10"], D["e20"] = T10.ema(D.c, 10), T10.ema(D.c, 20); D["vavg"] = D.v.rolling(20).mean().shift(1)
    dix = np.searchsorted(D.end.values, H4.end.values, side="right") - 1
    b = BBook(m5, blocked); last_day = None
    for j in range(20, len(H4)):
        bar, i = H4.iloc[j], dix[j]
        if b.pos:
            if b.walk(int(bar.ts), int(bar.end)):
                continue
            if i != last_day and i >= 0 and int(D.end.iloc[i]) <= int(bar.end):
                last_day = i
                _christian_exits(b, D, i)
            continue
        if i < 130 or not in_window(int(bar.end), t0, t1):
            continue
        for K in (60, 90, 120):
            w = D.iloc[i - K + 1:i + 1]
            hi, lo_ = w.h.iloc[:-1].max(), w.l.min()
            if hi / lo_ > 1.6:
                continue
            if bar.c > hi and bar.v >= D.vavg.iloc[i]:
                stop = H4.l.iloc[j - 11:j + 1].min()
                if stop < bar.c:
                    b.open(int(bar.end), bar.c, 1, stop, tag=f"横盘{K}天放量突破", days=0); last_day = i
                break
    return trades_df(b)


# ---------------- 3 维加斯隧道 3.0 ----------------
def m3_vegas(X, m5, t0, t1, blocked=()):
    e, top, bot, ok, vo = vegas(X)
    a = T10.atr(X).values
    b = BBook(m5, blocked); leg, cnt, prev_ok = 0, {}, False
    for i in range(700, len(X)):
        ts_, end_ = int(X.ts.iloc[i]), int(X.end.iloc[i])
        o = bool(ok.iloc[i])
        if o and not prev_ok:
            leg += 1
        prev_ok = o
        c, l = X.c.iloc[i], X.l.iloc[i]
        if b.pos:
            p = b.pos
            if b.walk(ts_, end_):
                continue
            if c < bot.iloc[i]:
                if vo.iloc[i] > 144:
                    b.close(c, end_, "放量跌破隧道"); continue
                p.setdefault("brk", end_)
                if end_ - p["brk"] >= 24 * H:
                    b.close(c, end_, "24小时回不到隧道"); continue
            elif c >= top.iloc[i]:
                p.pop("brk", None)
            if e[12].iloc[i] < bot.iloc[i]:
                b.close(c, end_, "EMA12跌破隧道")
            continue
        if not o or not in_window(end_, t0, t1) or cnt.get(leg, 0) >= 2:
            continue
        if l <= top.iloc[i] * 1.001 and c >= bot.iloc[i]:
            stop = e[676].iloc[i] - 0.2 * a[i]
            if stop < c:
                if b.open(end_, c, 1, stop, c + (c - stop), tag="回踩隧道", tp_frac=0.5):
                    cnt[leg] = cnt.get(leg, 0) + 1
    return trades_df(b)


# ---------------- 4 Bit 浪浪 起稳拐点 + 市场四季 ----------------
def summer(btcD):
    x = btcD.copy(); e20, e50 = T10.ema(x.c, 20), T10.ema(x.c, 50)
    s = (x.c > e20) & (e20 > e50) & (x.c / x.c.shift(20) - 1 > 0.08)
    return x.end.values, s.values


def m4_lang(M, Dd, btc, m5, t0, t1, blocked=()):
    se, sv = btc
    De, Dc = Dd.end.values, Dd.c.values
    De20, De50 = T10.ema(Dd.c, 20).values, T10.ema(Dd.c, 50).values
    a = T10.atr(M).values
    hh, ll, cc, end = M.h.values, M.l.values, M.c.values, M.end.values
    piv, k = [], 3
    b = BBook(m5, blocked)
    for i in range(2 * k, len(M)):
        j = i - k
        if hh[j] == hh[j - k:j + k + 1].max():
            piv.append(("H", j, hh[j]))
        elif ll[j] == ll[j - k:j + k + 1].min():
            piv.append(("L", j, ll[j]))
        if b.pos:
            b.walk(int(M.ts.iloc[i]), int(end[i])); continue
        if len(piv) < 4 or not in_window(int(end[i]), t0, t1):
            continue
        si = np.searchsorted(se, end[i], side="right") - 1; di = np.searchsorted(De, end[i], side="right") - 1
        if si < 0 or di < 50 or not sv[si] or not (Dc[di] > De20[di] > De50[di]):
            continue
        seq = piv[-4:]
        if [x[0] for x in seq] != ["H", "L", "H", "L"]:
            continue
        (_, _, H0), (_, _, L1), (_, jh1, H1), (_, jl2, L2) = seq
        if not (H1 < H0 and L2 >= L1 and i - jl2 <= 16 and cc[i] > H1 >= cc[i - 1]):
            continue
        stop = L2 - 0.2 * a[i]
        b.open(int(end[i]), cc[i], 1, stop, cc[i] + 3 * (cc[i] - stop), tag="起稳拐点")
    return trades_df(b)


# ---------------- 5 三率全优 ----------------
def m5_three(X, m5, t0, t1, blocked=()):
    norm, trig = nmacd(X.c)
    r, rs = rsi(X.c, 21), None
    rs = r.rolling(55).mean()
    ma13 = X.c.rolling(13).mean()
    a = T10.atr(X).values
    up = ((norm > trig) & (norm.shift(1) <= trig.shift(1))).values
    rup = ((r > rs) & (r.shift(1) <= rs.shift(1))).values
    dn = ((norm < trig) & (norm.shift(1) >= trig.shift(1))).values
    b = BBook(m5, blocked)
    for i in range(80, len(X)):
        ts_, end_ = int(X.ts.iloc[i]), int(X.end.iloc[i])
        if b.pos:
            if b.walk(ts_, end_):
                continue
            if dn[i]:
                b.close(X.c.iloc[i], end_, "NMACD死叉")
            continue
        if not in_window(end_, t0, t1) or not rup[i] or not up[max(0, i - 7):i + 1].any() or not norm.iloc[i] > trig.iloc[i] \
                or not X.c.iloc[i] > ma13.iloc[i]:
            continue
        c = X.c.iloc[i]
        stop = X.l.iloc[i] if (X.h.iloc[i] - X.l.iloc[i]) > 1.5 * a[i] else X.l.iloc[i - 9:i + 1].min()
        if stop < c:
            b.open(end_, c, 1, stop * 0.999, c + (c - stop), tag="三率全优", tp_frac=0.5)
    return trades_df(b)


# ---------------- 与 v5 结合用的状态（按 15m 收盘时刻 T 查）----------------
def ext_states(inst, btcD):
    M, H1, Dd = T10.load("15m", inst), T10.load("1H", inst), T10.load("1D", inst)
    _, _, bot15, ok15, _ = vegas(M)
    st15 = {int(t): (float(bb), bool(o)) for t, bb, o in zip(M.end.values, bot15.values, ok15.values)}
    _, _, _, ok1h, _ = vegas(H1)
    norm, trig = nmacd(H1.c)
    r = rsi(H1.c, 21); rs = r.rolling(55).mean()
    mh = macd_hist(H1.c)
    dn = (norm < trig) & (norm.shift(1) >= trig.shift(1))
    st1h = {int(t): (bool(o), bool(n > g), bool(x > y), bool(m > 0), bool(d_)) for t, o, n, g, x, y, m, d_
            in zip(H1.end.values, ok1h.values, norm.values, trig.values, r.values, rs.values, mh.values, dn.values)}
    e10, e20, e50 = T10.ema(Dd.c, 10).values, T10.ema(Dd.c, 20).values, T10.ema(Dd.c, 50).values
    dend = Dd.end.values
    se, sv = summer(btcD)
    def at(T):
        di = np.searchsorted(dend, T, side="right") - 1
        si = np.searchsorted(se, T, side="right") - 1
        return {"v15": st15.get(T), "h1": st1h.get(T // H * H), "dtrend": di >= 50 and e10[di] > e20[di] > e50[di],
                "summer": si >= 0 and bool(sv[si]), "d": (float(Dd.c.iloc[di]), float(e20[di])) if di >= 0 else None}
    return at


def gate(key):
    def g(t, side, info, hr):
        x = info.get("x")
        if not x:
            return False
        h1 = x["h1"]
        return {"B1": h1 and h1[3], "B2": x["dtrend"], "B3": h1 and h1[0], "B4": x["v15"] and x["v15"][1],
                "B5": x["summer"], "B6": h1 and h1[1] and h1[2]}[key]
    return g


def exit_fn(key):
    def f(info, pos, t):
        x = info.get("x")
        if not x:
            return None
        if key == "C3" and x["v15"] and info["close"] < x["v15"][0]:
            return ("close", "跌破隧道下沿")
        if key == "C4" and x["h1"] and x["h1"][4]:
            return ("close", "1H NMACD死叉")
        return None
    return f


def v5_sigs(extra_gate=None, **kw):
    strong, weak = {**V5.STRONG, **V5.CH, **kw}, {**V5.LOW, **V5.CH, **kw}
    tt = V5.tt_gate(40, 20)
    if extra_gate:
        strong["gate"] = extra_gate
        weak["gate"] = lambda *a: tt(*a) and extra_gate(*a)
    else:
        weak["gate"] = tt
    return [("trend", strong), ("weak", weak)]


def blk(wins):
    arr = sorted(wins)
    return lambda t, side, info, hr: not any(a <= t < b for a, b in arr)


def wins(df, a="entry_ts", b="exit_ts"):
    return [(int(x), int(y)) for x, y in zip(df[a], df[b])] if len(df) else []


# ---------------- 每个品种一个进程 ----------------
def job(args):
    inst, days, end = args if isinstance(args, tuple) else (args, 180, None)
    unit = dict(L.live_insts())[inst]
    base = L.load(days, ((inst, unit),), "_live", end)[inst]
    feats = B.features({inst: base}, days, end).get(inst, {})
    un, Ts, p15, p1h, b5, fm = base
    t0, t1 = Ts[0], Ts[-1]
    btcD = T10.load("1D", "BTC-USDT-SWAP")
    at = ext_states(inst, btcD)
    q = {T: {**p15[T], "tt": feats.get(T), "x": at(T)} for T in Ts if p15.get(T) is not None}
    data = (un, Ts, q, p1h, b5, fm)
    Dd = T10.load("1D", inst)
    daily_ex = {int(t): (float(c), float(e)) for t, c, e in zip(Dd.end.values, Dd.c.values, T10.ema(Dd.c, 20).values)}
    M, H1, H4, m5 = T10.load("15m", inst), T10.load("1H", inst), T10.load("4H", inst), T10.load("5m", inst)

    def v5(sigs):
        t = L.simulate(data, {"sigs": sigs})
        return pd.DataFrame({"ts": t.ts, "end": t.end, "ret": t.net / (t.entry * un * t.mult)}) if len(t) else pd.DataFrame(columns=["ts", "end", "ret"])

    def std(df):
        return pd.DataFrame({"ts": df.entry_ts, "end": df.exit_ts, "ret": df.ret}) if len(df) else pd.DataFrame(columns=["ts", "end", "ret"])

    methods = {
        "1 半木夏背离 15m（2 个波谷，2R）": lambda bl=(): m1_bmx(M, m5, t0, t1, blocked=bl),
        "1 半木夏背离 15m（3 个波谷，2R）": lambda bl=(): m1_bmx(M, m5, t0, t1, n=3, blocked=bl),
        "1 半木夏背离 15m（2 个波谷，1R）": lambda bl=(): m1_bmx(M, m5, t0, t1, rr=1.0, blocked=bl),
        "1 半木夏背离 1H（2 个波谷，2R）": lambda bl=(): m1_bmx(H1, m5, t0, t1, blocked=bl),
        "2 Kristjan 日线收盘突破": lambda bl=(): m2_kris_daily(Dd, m5, t0, t1, blocked=bl),
        "2 Kristjan 横盘放量突破（4H）": lambda bl=(): m2_kris_accum(Dd, H4, m5, t0, t1, blocked=bl),
        "2 参考：Christian 1H 收盘入场": lambda bl=(): T10.m10_christian(Dd, H1, m5, t0, bl).pipe(lambda x: x[x.entry_ts < t1] if len(x) else x),
        "3 维加斯隧道 15m": lambda bl=(): m3_vegas(M, m5, t0, t1, blocked=bl),
        "3 维加斯隧道 1H": lambda bl=(): m3_vegas(H1, m5, t0, t1, blocked=bl),
        "4 浪浪 起稳拐点 + 夏天": lambda bl=(): m4_lang(M, Dd, summer(btcD), m5, t0, t1, blocked=bl),
        "5 三率全优 1H": lambda bl=(): m5_three(H1, m5, t0, t1, blocked=bl),
        "5 三率全优 15m": lambda bl=(): m5_three(M, m5, t0, t1, blocked=bl),
    }
    out = {}
    base_v5 = v5(v5_sigs())
    out["v5（现在）"] = base_v5
    w5 = wins(base_v5, "ts", "end")
    for name, fn in methods.items():
        solo = fn()
        out[f"单独｜{name}"] = std(solo)
        mb = fn(w5)                                              # A：新方法避开 v5 持仓 → v5 再避开新方法
        v5b = v5(v5_sigs(blk(wins(mb))))
        out[f"A 加入 v5｜{name}"] = pd.concat([v5b, std(mb)])
    for key, label in (("B1", "B 过滤｜1H MACD(13,34) 柱 > 0（半木夏）"), ("B2", "B 过滤｜日线 EMA10>20>50（Kristjan）"),
                       ("B3", "B 过滤｜1H 维加斯多头排列"), ("B4", "B 过滤｜15m 维加斯多头排列"),
                       ("B5", "B 过滤｜BTC 夏天（浪浪）"), ("B6", "B 过滤｜1H NMACD 与 RSI 偏多（三率全优）")):
        out[label] = v5(v5_sigs(gate(key)))
    out["C 离场｜止盈 2R（半木夏）"] = v5(v5_sigs(tp_rr=2.0))
    out["C 离场｜反向信号或日线跌破 EMA20（Kristjan）"] = v5(v5_sigs(daily_exit=daily_ex))
    out["C 离场｜反向信号或 15m 跌破隧道下沿（维加斯）"] = v5(v5_sigs(exit_fn=exit_fn("C3")))
    out["C 离场｜反向信号或 1H NMACD 死叉（三率全优）"] = v5(v5_sigs(exit_fn=exit_fn("C4")))
    return inst, out


def stat(df, days=180):
    x = df.sort_values("ts").reset_index(drop=True)
    if not len(x):
        return {"复利": "-", "回撤": "-", "笔数": 0}
    cv, mx = L.compound(x)
    mdd = float((1 - cv["eq"] / cv["eq"].cummax()).max())
    r = x.ret.values
    return {"复利": f"{(cv['eq'].iloc[-1] - 1) * 100:+.0f}%", "回撤": f"{mdd:.0%}", "笔数": len(x), "每周": f"{len(x) / days * 7:.1f}",
            "胜率": f"{(r > 0).mean():.0%}", "去前5": f"{(r.sum() - np.sort(r)[-5:].sum()) * 100:+.0f}%", "最多同时": mx,
            "_ret": (cv["eq"].iloc[-1] - 1) * 100, "_dd": mdd * 100}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 260); pd.set_option("display.unicode.east_asian_width", True)
    periods = [("最近 180 天", 180, None, ())]
    if "2025" in sys.argv:
        periods = [("2025 全年", 365, V5.END25, ("ZEC-USDT-SWAP", "HYPE-USDT-SWAP"))]
    for pname, days, end, skip in periods:
        insts = [i for i, _ in L.live_insts() if i not in skip]
        t = time.time()
        with ProcessPoolExecutor(os.cpu_count()) as ex:
            res = dict(ex.map(job, [(i, days, end) for i in insts]))
        print(f"{len(insts)} 个品种并行，用时 {time.time() - t:.0f}s")
        names = list(next(iter(res.values())))
        rows = {n: stat(pd.concat([res[i][n] for i in insts if len(res[i][n])] or [pd.DataFrame(columns=["ts", "end", "ret"])]), days) for n in names}
        tb = pd.DataFrame(rows).T
        b = tb.loc["v5（现在）"]
        tb["比 v5"] = ["基准" if n == "v5（现在）" else ("✅ 利润↑回撤不增" if r["笔数"] and r["_ret"] > b["_ret"] and r["_dd"] <= b["_dd"] + 0.5 else
                      ("利润↑但回撤↑" if r["笔数"] and r["_ret"] > b["_ret"] else ("回撤↓但利润↓" if r["笔数"] and r["_dd"] < b["_dd"] else "")))
                      for n, r in tb.iterrows()]
        print(f"\n== {pname}，{len(insts)} 个品种，每笔名义 = 权益 × 1 复利 ==\n" + tb.drop(columns=["_ret", "_dd"]).to_string())
