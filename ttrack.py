"""
TradeTrack App（Android，Kotlin）的多周期指标评分与支撑压力算法的 Python 移植，口径与 App 一致，用于回测。
原实现：tradeTrack/app/src/main/java/com/example/tradetrack/ta/{Indicators,TechAnalysis}.kt

8 项指标各给 多 / 空 / 中 信号并加权，合成周期综合评分 -100（强空）~ 100（强多）：
  EMA 20/50/200 排列（权重 2）、MACD 柱方向（1.5）、RSI(14)（1，超买超卖只算中性）、KDJ（0.75）、
  布林带位置（1）、ADX/DMI（1.5）、OBV 量能（1）、SuperTrend(10,3)（1.5）
支撑压力：摆动高低点（按成交量和时间远近加权）+ 前一根高低点、EMA50/200、区间极值 → 以 0.35×ATR 聚类 →
  按“强度 ÷ 距离衰减”选上下各 3 档，强度归一化为 1~5 星。回测里没有历史盘口，所以不含挂单墙。
"""
import math

import numpy as np

WEIGHTS = {"EMA": 2.0, "MACD": 1.5, "RSI": 1.0, "KDJ": 0.75, "BOLL": 1.0, "ADX": 1.5, "OBV": 1.0, "ST": 1.5}
FRACTAL = {"1H": 3, "4H": 3, "1D": 3, "1W": 2}


def sma(v, n):
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        c = np.cumsum(np.insert(v, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def ema(v, n):
    """以前 n 个值的 SMA 作种子。"""
    out = np.full(len(v), np.nan)
    if len(v) < n:
        return out
    k = 2.0 / (n + 1)
    prev = float(np.mean(v[:n])); out[n - 1] = prev
    for i in range(n, len(v)):
        prev = v[i] * k + prev * (1 - k); out[i] = prev
    return out


def ema_skip_nan(v, n):
    ok = np.where(~np.isnan(v))[0]
    out = np.full(len(v), np.nan)
    if len(ok):
        out[ok[0]:] = ema(v[ok[0]:], n)
    return out


def rma(v, n, start=0):
    """Wilder 平滑。"""
    out = np.full(len(v), np.nan)
    if len(v) - start < n:
        return out
    prev = float(np.mean(v[start:start + n])); out[start + n - 1] = prev
    for i in range(start + n, len(v)):
        prev = (prev * (n - 1) + v[i]) / n; out[i] = prev
    return out


def true_range(h, l, c):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    ag, al = rma(np.maximum(d, 0), n, 1), rma(np.maximum(-d, 0), n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + ag / al)
    r[(al == 0) & ~np.isnan(ag)] = 100.0
    return r


def kdj(h, l, c, n=9, m1=3, m2=3):
    k = np.full(len(c), np.nan); d = k.copy(); j = k.copy()
    pk = pd_ = 50.0
    for i in range(n - 1, len(c)):
        hh, ll = h[i - n + 1:i + 1].max(), l[i - n + 1:i + 1].min()
        rsv = 50.0 if hh == ll else (c[i] - ll) / (hh - ll) * 100
        pk = ((m1 - 1) * pk + rsv) / m1; pd_ = ((m2 - 1) * pd_ + pk) / m2
        k[i], d[i], j[i] = pk, pd_, 3 * pk - 2 * pd_
    return k, d, j


def adx(h, l, c, n=14):
    up = np.diff(h, prepend=h[0]); dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm[0] = mdm[0] = 0.0
    atr_, p, m = rma(true_range(h, l, c), n, 1), rma(pdm, n, 1), rma(mdm, n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi, mdi = 100 * p / atr_, 100 * m / atr_
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    ok = np.where(~np.isnan(dx))[0]
    a = rma(dx, n, ok[0]) if len(ok) else np.full(len(c), np.nan)
    return a, pdi, mdi


def obv(c, v):
    s = np.sign(np.diff(c, prepend=c[0]))
    return np.cumsum(s * v)


def supertrend(h, l, c, n=10, mult=3.0):
    atr_ = rma(true_range(h, l, c), n)
    line = np.full(len(c), np.nan); up = np.ones(len(c), bool)
    fu = fl = math.nan
    for i in range(len(c)):
        if math.isnan(atr_[i]):
            continue
        hl2 = (h[i] + l[i]) / 2; bu, bl = hl2 + mult * atr_[i], hl2 - mult * atr_[i]
        pc = c[i - 1] if i > 0 else c[i]
        fu = bu if (math.isnan(fu) or bu < fu or pc > fu) else fu
        fl = bl if (math.isnan(fl) or bl > fl or pc < fl) else fl
        prev_up = up[i - 1] if i > 0 and not math.isnan(line[i - 1]) else True
        up[i] = c[i] >= fl if prev_up else c[i] > fu
        line[i] = fl if up[i] else fu
    return line, up


def last_valid(a):
    ok = a[~np.isnan(a)]
    return float(ok[-1]) if len(ok) else math.nan


def valid_at(a, back):
    ok = a[~np.isnan(a)]
    return float(ok[-1 - back]) if len(ok) > back else math.nan


def analyze(o, h, l, c, v, price, tf):
    """一个周期的 8 项指标评分与支撑压力。数组按时间升序，只含已收盘 K 线。返回 None 表示数据不足（少于 30 根）。"""
    n = len(c)
    if n < 30:
        return None
    last = price if price > 0 else c[-1]
    sig = {}
    e20, e50, e200 = last_valid(ema(c, 20)), last_valid(ema(c, 50)), last_valid(ema(c, 200))
    has_long = not math.isnan(e200)
    bull = last > e20 > e50 and (not has_long or e50 > e200)
    bear = last < e20 < e50 and (not has_long or e50 < e200)
    sig["EMA"] = 1 if bull else -1 if bear else (1 if not math.isnan(e50) and last > e50 and e20 > e50 else
                                                 -1 if not math.isnan(e50) and last < e50 and e20 < e50 else 0)
    macd = ema(c, 12) - ema(c, 26); hist = macd - ema_skip_nan(macd, 9)
    hh, hp = valid_at(hist, 0), valid_at(hist, 1)
    sig["MACD"] = 1 if hh > 0 and hh >= hp else -1 if hh < 0 and hh <= hp else 0
    r = last_valid(rsi(c))
    sig["RSI"] = 0 if (r >= 70 or r <= 30) else 1 if r >= 55 else -1 if r <= 45 else 0
    k, d, j = kdj(h, l, c)
    kk, dd, jj = last_valid(k), last_valid(d), last_valid(j)
    sig["KDJ"] = 0 if (jj > 100 or jj < 0) else 1 if kk > dd else -1 if kk < dd else 0
    mid = sma(c, 20); sd = np.array([np.nan] * 19 + [c[i - 19:i + 1].std() for i in range(19, n)]) if n >= 20 else np.full(n, np.nan)
    upb, lob, mi = last_valid(mid + 2 * sd), last_valid(mid - 2 * sd), last_valid(mid)
    pb = (last - lob) / (upb - lob) if upb > lob else 0.5
    sig["BOLL"] = 1 if pb > 1 else -1 if pb < 0 else 1 if last > mi else -1 if last < mi else 0
    a, pdi, mdi = adx(h, l, c)
    av, pv, mv = last_valid(a), last_valid(pdi), last_valid(mdi)
    sig["ADX"] = 1 if av >= 20 and pv > mv else -1 if av >= 20 and mv > pv else 0
    ob = obv(c, v); oe = last_valid(ema(ob, 20)); slope = ob[-1] - ob[max(0, n - 6)]
    sig["OBV"] = 1 if ob[-1] > oe and slope > 0 else -1 if ob[-1] < oe and slope < 0 else 0
    _, stu = supertrend(h, l, c)
    sig["ST"] = 1 if stu[-1] else -1
    score = int(sum(WEIGHTS[x] * s for x, s in sig.items()) / sum(WEIGHTS.values()) * 100)
    atr_ = last_valid(rma(true_range(h, l, c), 14))
    extras = [(h[-2], 1.2), (l[-2], 1.2), (e200, 1.5), (e50, 0.8), (h.max(), 1.5), (l.min(), 1.5)]
    sup, res = levels(h, l, v, last, atr_, FRACTAL[tf], extras)
    return {"score": score, "sig": sig, "atr": atr_, "sup": sup, "res": res, "adx": av, "rsi": r}


def levels(h, l, v, price, atr_, k, extras):
    """返回 (支撑, 压力)：各最多 3 个 (价格, 强度 1~5)，按离现价由近到远。"""
    n = len(h)
    avg_v = float(np.mean(v)) or 1.0
    pts = []
    for i in range(k, n - k):
        rec = i / n; w = 1.0 + min(max(v[i] / avg_v, 0.3), 3.0) * 0.5 + rec
        if h[i] >= h[i - k:i + k + 1].max():
            pts.append((h[i], w))
        if l[i] <= l[i - k:i + k + 1].min():
            pts.append((l[i], w))
    pts += [(p, w) for p, w in extras if p > 0 and not math.isnan(p)]
    if not pts:
        return [], []
    tol = max(0.0 if math.isnan(atr_) else atr_ * 0.35, price * 0.0015)
    clusters = []
    for p, w in sorted(pts):
        if clusters and p - clusters[-1][0] <= tol:
            cp, cw = clusters[-1]; nw = cw + w
            clusters[-1] = ((cp * cw + p * w) / nw, nw)
        else:
            clusters.append((p, w))
    maxw = max(w for _, w in clusters)
    rng = price * 0.3 if math.isnan(atr_) else atr_ * 15
    eps = price * 0.0005
    unit = price * 0.02 if (math.isnan(atr_) or atr_ <= 0) else atr_ * 4
    rank = lambda cl: cl[1] / (1 + abs(cl[0] - price) / unit)
    lv = lambda cl: (cl[0], min(max(math.ceil(cl[1] / maxw * 5), 1), 5))
    sup = sorted(sorted([x for x in clusters if x[0] < price - eps and price - x[0] <= rng], key=rank, reverse=True)[:3],
                 key=lambda x: -x[0])
    res = sorted(sorted([x for x in clusters if x[0] > price + eps and x[0] - price <= rng], key=rank, reverse=True)[:3],
                 key=lambda x: x[0])
    return [lv(x) for x in sup], [lv(x) for x in res]
