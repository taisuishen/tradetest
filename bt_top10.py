"""
投机实验室《全网 100 多种交易方法中测出的最强的 10 个》逐个回测（只看 ETH-USDT-SWAP，最近 2 年）。
视频：https://www.youtube.com/watch?v=T1CawPmNG-0 （方法说明见 docs/投机实验室_top10.md）

统一口径：每笔名义 = 当时权益 × 1（复利，一次只持一个仓位）；吃单 0.05%（开平各一次，分批平仓每批各收），不计滑点；
止损 / 止盈用 5 分钟 K 线逐根判定（同一根同时触及按先止损，偏保守）；按收盘价的离场规则在该周期收盘时执行。
视频没讲清楚的细节按注释里的假设实现，属于“近似”，结论只代表这些具体规则。

用法：python bt_top10.py [天数=730]
"""
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INST = "ETH-USDT-SWAP"
FEE = 0.0005
CACHE = Path(__file__).parent / "cache"
MS = {"5m": 300_000, "15m": 900_000, "1H": 3_600_000, "4H": 14_400_000, "1D": 86_400_000}


# ---------------- 数据与指标 ----------------
def load(bar, inst=INST):
    h = pickle.load(gzip.open(CACHE / f"{inst}_{bar}.pkl.gz", "rb"))
    ts = sorted(h)
    d = pd.DataFrame([h[t][:6] for t in ts], columns=["ts", "o", "h", "l", "c", "v"])
    d["end"] = d.ts + MS[bar]                      # 收盘时刻
    return d.reset_index(drop=True)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(d, n=14):
    pc = d.c.shift()
    tr = pd.concat([d.h - d.l, (d.h - pc).abs(), (d.l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd_hist(c):
    m = ema(c, 12) - ema(c, 26)
    return m - ema(m, 9)


def stoch(d, n, smooth):
    lo, hi = d.l.rolling(n).min(), d.h.rolling(n).max()
    k = (d.c - lo) / (hi - lo).replace(0, np.nan) * 100
    return k.rolling(smooth).mean()


# ---------------- 撮合 ----------------
class Book:
    """一次只持一个仓位；支持分批平仓与移动止损。"""

    def __init__(self, m5):
        self.t5, self.h5, self.l5, self.o5 = m5.ts.values, m5.h.values, m5.l.values, m5.o.values
        self.trades, self.pos = [], None

    def open(self, ts, px, side, stop, tp=None, tag="", **extra):
        self.pos = {"ts": ts, "entry": px, "side": side, "stop": stop, "tp": tp, "rem": 1.0, "real": 0.0, "fee": FEE,
                    "tag": tag, **extra}

    def part(self, px, frac, ts, why):
        """平掉 frac（占原始仓位的比例）。"""
        p = self.pos
        frac = min(frac, p["rem"])
        p["real"] += (px - p["entry"]) / p["entry"] * p["side"] * frac
        p["fee"] += FEE * frac * px / p["entry"]
        p["rem"] -= frac
        if p["rem"] <= 1e-9:
            self.close(px, ts, why, already=True)

    def close(self, px, ts, why, already=False):
        p = self.pos
        if not already:
            p["real"] += (px - p["entry"]) / p["entry"] * p["side"] * p["rem"]
            p["fee"] += FEE * p["rem"] * px / p["entry"]
        self.trades.append({"entry_ts": p["ts"], "exit_ts": ts, "side": p["side"], "entry": p["entry"], "exit": px,
                            "ret": p["real"] - p["fee"], "why": why, "tag": p["tag"]})
        self.pos = None

    def walk(self, t0, t1):
        """在 [t0, t1) 的 5 分钟 K 线里检查止损 / 止盈；触发就平仓并返回 True。"""
        p = self.pos
        if not p:
            return False
        i = np.searchsorted(self.t5, t0, side="left")
        j = np.searchsorted(self.t5, t1, side="left")
        s = p["side"]
        for k in range(i, j):
            h, l, o = self.h5[k], self.l5[k], self.o5[k]
            if (l <= p["stop"]) if s > 0 else (h >= p["stop"]):
                fill = min(o, p["stop"]) if s > 0 else max(o, p["stop"])   # 跳空穿过止损按开盘价
                self.close(fill, int(self.t5[k]) + 300_000, "止损" if not p.get("be") else "保本")
                return True
            if p["tp"] is not None and ((h >= p["tp"]) if s > 0 else (l <= p["tp"])):
                tp_frac = p.get("tp_frac", 1.0)
                if tp_frac >= p["rem"] - 1e-9:
                    self.close(p["tp"], int(self.t5[k]) + 300_000, "止盈")
                    return True
                self.part(p["tp"], tp_frac, int(self.t5[k]) + 300_000, "止盈")
                p["tp"] = None
                if p.get("tp_then_be"):
                    p["stop"], p["be"] = p["entry"], True
        return False


def stats(tr, days):
    if not len(tr):
        return {"笔数": 0}
    r = tr.ret.values
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    w, l = r[r > 0], r[r <= 0]
    return {"笔数": len(r), "每月": f"{len(r) / days * 30:.1f}", "胜率": f"{(r > 0).mean():.0%}",
            "均盈": f"{w.mean() * 100:+.2f}%" if len(w) else "-", "均亏": f"{l.mean() * 100:+.2f}%" if len(l) else "-",
            "盈亏因子": f"{w.sum() / -l.sum():.2f}" if len(l) and l.sum() else "∞",
            "复利": f"{(eq[-1] - 1) * 100:+.0f}%", "最大回撤": f"{(1 - eq / peak).max() * 100:.0f}%",
            "去前5（不复利）": f"{(r.sum() - np.sort(r)[-5:].sum()) * 100:+.0f}%"}


# ---------------- 各方法 ----------------
def m10_christian(D, H1, m5, start, blocked=()):
    """第 10 位 Christian（Qullamaggie）突破：
    ① 整理前 1~3 个月涨幅 30%~100%；② 整理 2 周~2 个月（10~40 根日线），低点抬高、区间收窄；
    ③ 1 小时收盘突破整理区高点就买；④ 止损在当日最低价，且不超过 1 倍日线 ATR（超过就按 1 倍 ATR）；
    ⑤ 第 5 天收盘平 1/3、止损移到开仓价；日线收盘跌破 EMA10 再平 1/3，跌破 EMA20 平剩下的。
    blocked：[(开始, 结束)] 这些时段内不开新仓（和其他模块共用一个品种仓位时用）。"""
    D = D.copy(); D["atr"] = atr(D); D["e10"], D["e20"], D["e50"] = ema(D.c, 10), ema(D.c, 20), ema(D.c, 50)
    b = Book(m5)
    for i in range(80, len(D)):
        day = D.iloc[i]
        if day.ts < start:
            continue
        p = b.pos
        if p:                                            # 持仓：先在当天的 5 分钟 K 线里看止损，再按收盘规则
            if b.walk(day.ts, day.end):
                continue
            p["days"] += 1
            if p["days"] == 5 and p["rem"] > 0.99:
                b.part(day.c, 1 / 3, day.end, "第5天平1/3"); p["stop"], p["be"] = p["entry"], True
            if b.pos and day.c < day.e10 and p["rem"] > 0.34 and not p.get("cut10"):
                b.part(day.c, 1 / 3, day.end, "跌破EMA10"); p["cut10"] = True
            if b.pos and day.c < day.e20:
                b.close(day.c, day.end, "跌破EMA20")
            continue
        base = None                                      # 用前一天为止的日线找整理区
        for K in range(10, 41):
            w = D.iloc[i - K:i]
            H = w.h.max()
            prior_low = D.iloc[max(0, i - K - 66):i - K].l.min()
            half = K // 2
            if not (1.30 <= H / prior_low <= 2.0):
                continue
            if w.l.iloc[half:].min() <= w.l.iloc[:half].min():          # 低点抬高
                continue
            third = max(3, K // 3)
            if (w.h.iloc[-third:].max() - w.l.iloc[-third:].min()) >= (w.h.iloc[:third].max() - w.l.iloc[:third].min()):
                continue                                                  # 区间收窄
            base = H; break
        if base is None:
            continue
        hs = H1[(H1.ts >= day.ts) & (H1.ts < day.end)]
        for _, bar in hs.iterrows():                    # 当天 1 小时收盘突破整理区高点
            if bar.c > base:
                if any(a0 <= int(bar.end) < a1 for a0, a1 in blocked):
                    break                                # 这个品种正被别的模块占着
                day_low = hs[hs.ts <= bar.ts].l.min()
                stop = max(day_low, bar.c - D.iloc[i - 1].atr)
                b.open(int(bar.end), bar.c, 1, stop, tag="突破", days=0)
                b.walk(int(bar.end), day.end)
                break
    return pd.DataFrame(b.trades)


def m9_langlang(D, H1, M15, m5, start):
    """第 9 位 bit 浪浪（近似）：只做阶段 1 / 2 / 4 的顺势多单。
    大级别：日线收盘 > EMA20 > EMA50（上升趋势）；上方空间：离 90 天最高点至少 2 倍日线 ATR（或已创新高）；
    小级别入场：15 分钟出现“起稳拐点”——回调形成更高的低点后，收盘突破两个低点之间的高点；
    止损：更高的那个低点下方 0.2 倍 15 分钟 ATR；止盈：到 90 天最高点（已创新高则 3R），或止损。"""
    D = D.copy(); D["e20"], D["e50"], D["atr"] = ema(D.c, 20), ema(D.c, 50), atr(D); D["hi90"] = D.h.rolling(90).max()
    M = M15.copy(); M["atr"] = atr(M)
    n = len(M); hh, ll, cc = M.h.values, M.l.values, M.c.values
    piv_lo, piv_hi = [], []
    b = Book(m5)
    di = np.searchsorted(D.end.values, M.end.values, side="right") - 1      # 每根 15m 收盘时已收盘的最后一根日线
    k = 3
    for i in range(k * 2, n):
        t = int(M.ts.iloc[i])
        j = i - k                                        # 分型要等右边 k 根走完才确认
        if hh[j] == hh[j - k:j + k + 1].max():
            piv_hi.append((j, hh[j]))
        if ll[j] == ll[j - k:j + k + 1].min():
            piv_lo.append((j, ll[j]))
        if t < start:
            continue
        if b.pos:
            b.walk(t, t + MS["15m"]); continue
        if di[i] < 90 or len(piv_lo) < 2 or not piv_hi:
            continue
        d = D.iloc[di[i]]
        if not (d.c > d.e20 > d.e50):
            continue
        (j1, l1), (j2, l2) = piv_lo[-2], piv_lo[-1]
        mids = [p for p in piv_hi if j1 < p[0] < j2]
        if l2 <= l1 or not mids or i - j2 > 12:
            continue
        trig = max(p[1] for p in mids)
        if not (cc[i] > trig >= cc[i - 1]):
            continue
        room = d.hi90 - cc[i]
        if 0 < room < 2 * d.atr:
            continue
        stop = l2 - 0.2 * M.atr.iloc[i]
        tp = d.hi90 if room > 0 else cc[i] + 3 * (cc[i] - stop)
        b.open(int(M.end.iloc[i]), cc[i], 1, stop, tp, tag="起稳拐点")
    return pd.DataFrame(b.trades)


def m8_minervini(D, m5, start, loose=False):
    """第 8 位 Mark Minervini SEPA + VCP：
    趋势模板：收盘 > SMA50 > SMA150 > SMA200、SMA200 比 22 天前高、比 52 周低点高 ≥25%、离 52 周高点 ≤25%；
    VCP：最近 10 / 20 / 40 天的振幅依次收窄，且 10 天均量 < 50 天均量（量缩）；loose=1 只要求最近 10 天比前 10 天收窄、不要求量缩；
    loose=2 只要趋势模板 + 收盘突破 10 天高点（不要求 VCP、不要求放量）；
    入场：日线收盘放量（> 1.5 倍 50 天均量）突破最近 10 天高点；止损：突破日最低价（最多 8%）；离场：收盘跌破 SMA50。"""
    D = D.copy()
    for n in (50, 150, 200):
        D[f"s{n}"] = D.c.rolling(n).mean()
    D["lo52"], D["hi52"] = D.l.rolling(365).min(), D.h.rolling(365).max()
    D["v50"], D["v10"] = D.v.rolling(50).mean(), D.v.rolling(10).mean()
    b = Book(m5)
    for i in range(365, len(D)):
        d = D.iloc[i]
        if d.ts < start:
            continue
        if b.pos:
            if b.walk(d.ts, d.end):
                continue
            if d.c < d.s50:
                b.close(d.c, d.end, "跌破SMA50")
            continue
        if not (d.c > d.s50 > d.s150 > d.s200 and d.s200 > D.s200.iloc[i - 22] and d.c >= 1.25 * d.lo52 and d.c >= 0.75 * d.hi52):
            continue
        w = D.iloc[i - 40:i]
        rng = lambda x: (x.h.max() - x.l.min()) / x.c.iloc[-1]
        if loose == 1:
            if not rng(w.iloc[-10:]) < rng(w.iloc[-20:-10]):
                continue
        elif loose == 2:
            pass
        elif not (rng(w.iloc[-10:]) < rng(w.iloc[-20:-10]) < rng(w.iloc[:20])) or not (D.v10.iloc[i - 1] < D.v50.iloc[i - 1]):
            continue
        pivot = w.iloc[-10:].h.max()
        if d.c > pivot and (loose == 2 or d.v > 1.5 * D.v50.iloc[i - 1]):
            b.open(int(d.end), d.c, 1, max(d.l, d.c * 0.92), tag="VCP突破")
    return pd.DataFrame(b.trades)


def m7_rolling(H4, m5, start):
    """第 7 位 浮盈滚仓（按视频的阶梯表）：
    入场（视频第三种）：从 90 天最低点反弹 ≥20%，且此前 30 根 4 小时收盘都在维加斯通道（EMA144/169）附近以下，
    然后 4 小时放量（> 1.5 倍 20 根均量）收盘站上通道上沿 → 用 1 份本金、逐仓 50 倍开多；
    本金（保证金 + 浮盈）每翻一倍就把利润加进仓位、杠杆按 50→30→20→15→10→5 降下来，第一次加仓后止损移到开仓价；
    亏完保证金（≈ 1/杠杆 − 0.5% 维持保证金）即爆仓；到 100 倍或 4 小时收盘跌破通道下沿就全部离场。
    每次尝试的结果按“本金的倍数”记（亏光 = −100%）。"""
    H = H4.copy(); H["a"], H["b"] = ema(H.c, 144), ema(H.c, 169); H["v20"] = H.v.rolling(20).mean()
    H["lo90"] = H.l.rolling(540).min()
    t5, h5, l5, c5 = m5.ts.values, m5.h.values, m5.l.values, m5.c.values
    LEV = [50, 30, 20, 15, 10, 5]
    res, i = [], 600
    while i < len(H):
        d = H.iloc[i]
        top = max(d.a, d.b)
        prev = H.iloc[i - 30:i]
        ok = (d.ts >= start and d.c > top and (prev.c <= prev[["a", "b"]].max(axis=1) * 1.01).all()
              and d.v > 1.5 * H.v20.iloc[i - 1] and d.c >= 1.2 * d.lo90)
        if not ok:
            i += 1; continue
        entry, stake, lvl = d.c, 1.0, 0
        base_eq = stake
        eq, notional, avg = stake, stake * LEV[0], entry
        eq -= notional * FEE
        stop, out, why, t_end = None, None, None, None
        k = np.searchsorted(t5, int(d.end))
        while k < len(t5):
            lo = l5[k]
            upl_lo = notional * (lo - avg) / avg
            if stop is not None and lo <= stop:
                eq += notional * (stop - avg) / avg - notional * FEE; out, why = eq, "保本止损"; t_end = t5[k]; break
            if upl_lo <= -(eq - notional * 0.005):
                out, why = 0.0, "爆仓"; t_end = t5[k]; break
            hi = h5[k]
            val = eq + notional * (hi - avg) / avg
            if val >= 2 * base_eq and lvl < len(LEV) - 1:   # 本金翻倍：把浮盈加进去、降杠杆
                eq = val; lvl += 1; base_eq = eq
                new_notional = eq * LEV[lvl]
                eq -= abs(new_notional - notional) * FEE
                notional, avg = new_notional, hi
                if stop is None:
                    stop = entry
            if val >= 100 * stake:
                out, why = val - notional * FEE, "100倍止盈"; t_end = t5[k]; break
            if (t5[k] + 300_000) % MS["4H"] == 0:        # 4 小时收盘：跌破通道下沿就走
                j = np.searchsorted(H.ts.values, t5[k] + 300_000 - MS["4H"])
                if j < len(H) and H.c.iloc[j] < min(H.a.iloc[j], H.b.iloc[j]):
                    eq += notional * (c5[k] - avg) / avg - notional * FEE; out, why = max(eq, 0.0), "跌破通道"; t_end = t5[k]; break
            k += 1
        if out is None:
            break
        res.append({"entry_ts": int(d.end), "exit_ts": int(t_end), "side": 1, "entry": entry, "exit": np.nan,
                    "ret": out / stake - 1, "why": why, "tag": f"最高降到 {LEV[lvl]} 倍"})
        i = np.searchsorted(H.ts.values, t_end) + 1
    return pd.DataFrame(res)


def m6_macd3(X, m5, start, tf, side=1):
    """第 6 位 半神 MACD 三段背离（参数 12/26/9）：
    底背离（做多）：价格连续 3 段低点一个比一个低，同时对应的 3 段绿柱（负值）一段比一段短（每段之间要有红柱隔开）；
    第 3 段绿柱开始缩短的那根 K 线收盘入场；止损：第 3 段的最低价；
    下一根收盘绿柱没有继续缩短就立即离场；有效期最多 50 根 K 线（到期离场）。顶背离（做空）对称。"""
    X = X.copy(); X["hist"] = macd_hist(X.c) * side
    b = Book(m5); segs, cur = [], None
    hist, lo, hi, cl = X["hist"].values, X.l.values, X.h.values, X.c.values
    for i in range(35, len(X)):
        t0, t1 = int(X.ts.iloc[i]), int(X.end.iloc[i])
        if b.pos:
            p = b.pos
            if b.walk(t0, t1):
                continue
            p["bars"] += 1
            if p["bars"] == 1 and not (hist[i] > hist[i - 1]):
                b.close(cl[i], t1, "柱子没继续缩短"); continue
            if p["bars"] >= 50:
                b.close(cl[i], t1, "到期"); continue
        neg = hist[i] < 0
        price = lo[i] if side > 0 else hi[i]
        if neg:
            if cur is None:
                cur = {"min": hist[i], "px": price}
            else:
                cur["min"] = min(cur["min"], hist[i])
                cur["px"] = min(cur["px"], price) if side > 0 else max(cur["px"], price)
        elif cur is not None:
            segs.append(cur); cur = None
        if b.pos or cur is None or len(segs) < 2 or t0 < start:
            continue
        s1, s2 = segs[-2], segs[-1]
        deeper = (lambda a, c: a < c) if side > 0 else (lambda a, c: a > c)
        if (deeper(s2["px"], s1["px"]) and deeper(cur["px"], s2["px"]) and s1["min"] < s2["min"] < cur["min"]
                and hist[i] > hist[i - 1] and hist[i - 1] <= cur["min"] + 1e-12 and not cur.get("used")):
            cur["used"] = True
            b.open(t1, cl[i], side, cur["px"], tag=f"{tf} 三段{'底' if side > 0 else '顶'}背离", bars=0)
    return pd.DataFrame(b.trades)


def m5_flight(X, m5, start, tf, two=False):
    """第 5 位 韩国 Flight 逆势抄底（做多）：
    急跌中出现“天量大阴线”——实体 ≥ 2 倍 ATR、成交量 ≥ 3 倍 20 根均量、并且跌破最近 3 天的最低点（打到关键支撑）；
    two=True 要求连续两根这样的 K 线；下一根开盘买入，止损 2%，止盈 4%（盈亏比 2），最多持有 12 根 K 线。"""
    X = X.copy(); X["atr"] = atr(X); X["v20"] = X.v.rolling(20).mean()
    look = int(3 * MS["1D"] / MS[tf])
    X["low3d"] = X.l.rolling(look).min().shift(1)
    big = ((X.o - X.c) >= 2 * X.atr.shift(1)) & (X.v >= 3 * X.v20.shift(1)) & (X.l < X.low3d)
    sig = big & big.shift(1).fillna(False) if two else big
    b = Book(m5)
    for i in range(30, len(X) - 1):
        t0, t1 = int(X.ts.iloc[i]), int(X.end.iloc[i])
        if b.pos:
            b.pos["bars"] += 1
            if b.walk(t0, t1):
                continue
            if b.pos["bars"] >= 12:
                b.close(X.c.iloc[i], t1, "到期")
            continue
        if t0 >= start and sig.iloc[i]:
            px = X.o.iloc[i + 1]
            b.open(int(X.ts.iloc[i + 1]), px, 1, px * 0.98, px * 1.04, tag="天量阴线", bars=-1)
    return pd.DataFrame(b.trades)


def m4_quad(X, m5, start, tf, mode):
    """第 4 位 四重随机指标轮动（Stochastic 9-3、14-3、40-4、60-10，只做多）：
    mode=1 背离：四条都跌破 20 → 反弹（9-3 回升 ≥10）→ 价格再创新低但 9-3 没创新低（底背离）→ 9-3 拐头向上收盘入场；
           止损：背离低点下方 0.2 倍 ATR；止盈：1 倍风险（盈亏比 1:1）。
    mode=2 牛旗：EMA20 > EMA50 > EMA200、价格回踩到 EMA20、9-3 ≤ 20、60-10 ≥ 85 → 收盘入场；
           止损：EMA50 下方 0.2 倍 ATR；9-3 上穿 80 时平一半并把止损移到开仓价，剩下的收盘跌破 EMA50 离场。"""
    X = X.copy(); X["atr"] = atr(X)
    for n, s in ((9, 3), (14, 3), (40, 4), (60, 10)):
        X[f"k{n}"] = stoch(X, n, s)
    X["e20"], X["e50"], X["e200"] = ema(X.c, 20), ema(X.c, 50), ema(X.c, 200)
    b = Book(m5); st = None
    k9 = X.k9.values
    for i in range(210, len(X)):
        r = X.iloc[i]; t0, t1 = int(r.ts), int(r.end)
        if b.pos:
            p = b.pos
            if b.walk(t0, t1):
                continue
            if mode == 2:
                if not p.get("half") and k9[i] > 80 >= k9[i - 1]:
                    b.part(r.c, 0.5, t1, "9-3上穿80平一半"); p["half"] = True; p["stop"], p["be"] = p["entry"], True
                if b.pos and r.c < r.e50:
                    b.close(r.c, t1, "跌破EMA50")
            continue
        if mode == 1:
            if max(r.k9, r.k14, r.k40, r.k60) < 20:
                st = {"L1": r.l, "K1": r.k9, "i": i, "bounced": False} if st is None or st.get("div") else \
                     {**st, "L1": min(st["L1"], r.l), "K1": min(st["K1"], r.k9)}
                continue
            if st is None or i - st["i"] > 40:
                st = None; continue
            if not st["bounced"] and r.k9 >= st["K1"] + 10:
                st["bounced"] = True
            if st["bounced"] and r.l < st["L1"]:
                st["L2"], st["K2"] = min(st.get("L2", r.l), r.l), min(st.get("K2", r.k9), r.k9)
            if st.get("L2") and st["K2"] > st["K1"] and k9[i] > k9[i - 1] and t0 >= start:
                stop = st["L2"] - 0.2 * r.atr
                if stop < r.c:
                    b.open(t1, r.c, 1, stop, r.c + (r.c - stop), tag="四重背离")
                st = None
        else:
            if (t0 >= start and r.e20 > r.e50 > r.e200 and r.l <= r.e20 < r.h + r.atr and r.k9 <= 20 and r.k60 >= 85
                    and r.c > r.e50):
                b.open(t1, r.c, 1, r.e50 - 0.2 * r.atr, tag="牛旗")
    return pd.DataFrame(b.trades)


def m3_larry(D, m5, start, exit_mode, side=1):
    """第 3 位 Larry Williams 外包线（日线）：
    做多：今天最高 > 昨天最高、最低 < 昨天最低（外包），收盘 < 昨天最低，实体 ≥ 昨天实体 2 倍 → 明天开盘买入；做空对称。
    止损：外包线最低价下方 0.2 倍 ATR14；exit_mode：hl=外包线最高价止盈 / rr2=2 倍风险止盈 / fpo=第一个盈利的开盘价离场。"""
    D = D.copy(); D["atr"] = atr(D); D["body"] = (D.c - D.o).abs()
    b = Book(m5)
    for i in range(20, len(D) - 1):
        d, y = D.iloc[i], D.iloc[i - 1]
        if b.pos:
            p = b.pos
            if exit_mode == "fpo" and (d.o - p["entry"]) * p["side"] > 0 and d.ts > p["ts"]:
                b.close(d.o, int(d.ts), "第一个盈利的开盘");
            elif b.walk(d.ts, d.end):
                pass
        if b.pos or d.ts < start:
            continue
        outside = d.h > y.h and d.l < y.l and d.body >= 2 * y.body
        if not outside or not ((d.c < y.l) if side > 0 else (d.c > y.h)):
            continue
        n = D.iloc[i + 1]
        stop = d.l - 0.2 * d.atr if side > 0 else d.h + 0.2 * d.atr
        tp = (d.h if side > 0 else d.l) if exit_mode == "hl" else (n.o + side * 2 * abs(n.o - stop)) if exit_mode == "rr2" else None
        if (stop - n.o) * side >= 0 or (tp is not None and (tp - n.o) * side <= 0):
            continue
        b.open(int(n.ts), n.o, side, stop, tp, tag=f"外包线 {exit_mode}")
    return pd.DataFrame(b.trades)


def m2_alex(D, H1, m5, start, run_pct=0.15):
    """第 2 位 Alex Temiz 大涨后做空（只做空）：
    连续 ≥3 天收阳、累计涨幅 ≥ run_pct（抛物线）；之后某天价格跌破前一日收盘价就做空（1 小时 K 线判定，按该价成交）；
    止损：当天收盘重新站上前一日收盘价就平仓，另设硬止损在当天最高价之上；持有最多 5 天，每天收盘检查。"""
    D = D.copy(); D["up"] = D.c > D.o
    b = Book(m5)
    for i in range(10, len(D)):
        d = D.iloc[i]
        if b.pos:
            p = b.pos
            if b.walk(d.ts, d.end):
                continue
            p["days"] += 1
            if d.c > D.c.iloc[i - 1]:
                b.close(d.c, d.end, "收盘站回前一日收盘价")
            elif p["days"] >= 5:
                b.close(d.c, d.end, "持有 5 天到期")
            continue
        if d.ts < start:
            continue
        n = 0
        while n < i and D.up.iloc[i - 1 - n]:
            n += 1
        if n < 3 or D.c.iloc[i - 1] / D.o.iloc[i - n] - 1 < run_pct:
            continue
        ref = D.c.iloc[i - 1]
        hs = H1[(H1.ts >= d.ts) & (H1.ts < d.end)]
        for _, bar in hs.iterrows():
            if bar.l < ref:
                hi = hs[hs.ts <= bar.ts].h.max()
                b.open(int(bar.ts), min(ref, bar.o), -1, hi * 1.002, tag=f"连涨{n}天后破位", days=0)
                b.walk(int(bar.ts), d.end)
                if b.pos and d.c > ref:
                    b.close(d.c, d.end, "收盘站回前一日收盘价")
                break
    return pd.DataFrame(b.trades)


def v5_eth(days):
    """对照：我们的 v5 策略只跑 ETH（每笔名义 = 权益 × 1）。"""
    import bt_ttrack as B
    import bt_v5
    import chan15_lab2 as L
    data = {INST: L.load(days, ((INST, 1.0),), "_eth")[INST]}
    data = B.attach(data, B.features(data, days))
    t = L.simulate(data[INST], {"sigs": bt_v5.V["v5（两者都加，当前实盘）"]})
    t["ret"] = t.net / (t.entry * t.mult)
    return t.rename(columns={"ts": "entry_ts"})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 260); pd.set_option("display.unicode.east_asian_width", True)
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 730
    D, H4, H1, M15, m5 = load("1D"), load("4H"), load("1H"), load("15m"), load("5m")
    end = int(m5.ts.iloc[-1]); start = end - days * MS["1D"]
    rows = {}
    def add(name, tr):
        rows[name] = stats(tr, days); print(f"  {name}：{rows[name].get('笔数', 0)} 笔", flush=True)
        return tr
    add("10 Christian 整理突破", m10_christian(D, H1, m5, start))
    add("9 bit浪浪 起稳拐点（近似）", m9_langlang(D, H1, M15, m5, start))
    add("8 Minervini VCP", m8_minervini(D, m5, start))
    add("8 Minervini VCP（放宽）", m8_minervini(D, m5, start, loose=1))
    add("8 Minervini 只要趋势模板+突破", m8_minervini(D, m5, start, loose=2))
    roll = m7_rolling(H4, m5, start)
    rows["7 浮盈滚仓（按本金倍数）"] = ({"笔数": len(roll), "胜率": f"{(roll.ret > 0).mean():.0%}", "均盈": f"{roll[roll.ret > 0].ret.mean() * 100:+.0f}%" if (roll.ret > 0).any() else "-",
                                     "复利": f"合计 {roll.ret.sum():+.1f} 份本金", "最大回撤": f"爆仓 {(roll.why == '爆仓').sum()} 次"} if len(roll) else {"笔数": 0})
    print(f"  7 浮盈滚仓：{len(roll)} 次尝试", flush=True)
    for tf, X in (("1D", D), ("4H", H4)):
        add(f"6 MACD 三段底背离 {tf}", m6_macd3(X, m5, start, tf, 1))
        add(f"6 MACD 三段顶背离 {tf}（空）", m6_macd3(X, m5, start, tf, -1))
    for tf, X in (("15m", M15), ("1H", H1)):
        add(f"5 Flight 天量阴线抄底 {tf}", m5_flight(X, m5, start, tf))
        add(f"5 Flight 连续两根 {tf}", m5_flight(X, m5, start, tf, two=True))
        add(f"4 四重随机·背离 {tf}", m4_quad(X, m5, start, tf, 1))
        add(f"4 四重随机·牛旗 {tf}", m4_quad(X, m5, start, tf, 2))
    for em in ("hl", "rr2", "fpo"):
        add(f"3 Larry 外包线 多 {em}", m3_larry(D, m5, start, em, 1))
        add(f"3 Larry 外包线 空 {em}", m3_larry(D, m5, start, em, -1))
    for rp in (0.10, 0.15):
        add(f"2 Alex 连涨后做空 ≥{rp:.0%}（空）", m2_alex(D, H1, m5, start, rp))
    try:
        add("对照：我们的 v5（ETH）", v5_eth(days))
    except Exception as e:
        print("  v5 对照失败：", e)
    print(f"\n== ETH 最近 {days} 天（{pd.Timestamp(start, unit='ms'):%Y-%m-%d} ~ {pd.Timestamp(end, unit='ms'):%Y-%m-%d}），每笔名义 = 权益 × 1，吃单 0.05% ==")
    print(pd.DataFrame(rows).T.fillna("").to_string())
