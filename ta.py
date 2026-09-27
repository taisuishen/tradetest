"""课程中用到的技术指标、结构与形态识别。每个函数都注明对应讲次。"""
import numpy as np
import pandas as pd


# ---------------- 基础指标 ----------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def sma(s, n):
    return s.rolling(n).mean()


def atr(df, n=14):
    pc = df.c.shift()
    tr = pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd(c, f=12, s=26, sig=9):  # 第3讲
    dif = ema(c, f) - ema(c, s)
    dea = ema(dif, sig)
    return dif, dea, 2 * (dif - dea)


def rsi(c, n=14):  # 第14讲
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def kdj(df, n=9):  # 第13讲
    lo, hi = df.l.rolling(n).min(), df.h.rolling(n).max()
    rsv = ((df.c - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def boll(c, n=20, m=2):  # 第15讲
    mid = sma(c, n)
    sd = c.rolling(n).std(ddof=0)
    return mid + m * sd, mid, mid - m * sd


def cci(df, n=14):  # 第38讲
    tp = (df.h + df.l + df.c) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md)


def williams_r(df, n=10):  # 第39讲：(H10-C)/(H10-L10)*100，数值越小越强
    hi, lo = df.h.rolling(n).max(), df.l.rolling(n).min()
    return (hi - df.c) / (hi - lo).replace(0, np.nan) * 100


def bbi(c):  # 第35讲
    return (sma(c, 3) + sma(c, 6) + sma(c, 12) + sma(c, 24)) / 4


def pbx(c):  # 第36讲 瀑布线
    return {m: (ema(c, m) + sma(c, 2 * m) + sma(c, 4 * m)) / 3 for m in (4, 6, 9, 13, 18, 24)}


def adx(df, n=14):  # 技术基础 DMI
    up, dn = df.h.diff(), -df.l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    ndi = 100 * pd.Series(ndm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return pdi, ndi, dx.ewm(alpha=1 / n, adjust=False).mean()


def asi(df):  # 第34讲（按 Wilder 原式）
    cp, op, lp = df.c.shift(), df.o.shift(), df.l.shift()
    A, B = (df.h - cp).abs(), (df.l - cp).abs()
    C, D = (df.h - lp).abs(), (cp - op).abs()
    R = np.where((A >= B) & (A >= C), A - 0.5 * B + 0.25 * D,
                 np.where((B >= A) & (B >= C), B - 0.5 * A + 0.25 * D, C + 0.25 * D))
    R = pd.Series(R, index=df.index).replace(0, np.nan)
    X = (df.c - cp) + 0.5 * (df.c - df.o) + 0.25 * (cp - op)
    K = pd.concat([A, B], axis=1).max(axis=1)
    T = (df.h - df.l).median() * 3
    si = (50 * X / R * K / T).fillna(0)
    return si.cumsum()


def ichimoku(df, a=9, b=26, c=52):  # 第32讲
    ten = (df.h.rolling(a).max() + df.l.rolling(a).min()) / 2
    kij = (df.h.rolling(b).max() + df.l.rolling(b).min()) / 2
    span_a_raw = (ten + kij) / 2
    span_b_raw = (df.h.rolling(c).max() + df.l.rolling(c).min()) / 2
    return ten, kij, span_a_raw.shift(b), span_b_raw.shift(b), span_a_raw, span_b_raw


# ---------------- 结构：ATR 自适应 ZigZag（道氏高低点） ----------------
def zigzag(df, a, mult=2.0):
    h, l = df.h.values, df.l.values
    av = a.bfill().values
    piv, trend = [], 0
    hi, hi_i, lo, lo_i = h[0], 0, l[0], 0
    for i in range(1, len(df)):
        thr = mult * av[i]
        if trend == 0:
            if h[i] > hi: hi, hi_i = h[i], i
            if l[i] < lo: lo, lo_i = l[i], i
            if hi - l[i] > thr and hi_i < i:
                piv.append((hi_i, hi, "H")); trend = -1; lo, lo_i = l[i], i
            elif h[i] - lo > thr and lo_i < i:
                piv.append((lo_i, lo, "L")); trend = 1; hi, hi_i = h[i], i
        elif trend == 1:
            if h[i] > hi: hi, hi_i = h[i], i
            elif hi - l[i] > thr:
                piv.append((hi_i, hi, "H")); trend = -1; lo, lo_i = l[i], i
        else:
            if l[i] < lo: lo, lo_i = l[i], i
            elif h[i] - lo > thr:
                piv.append((lo_i, lo, "L")); trend = 1; hi, hi_i = h[i], i
    tentative = (hi_i, hi, "H") if trend == 1 else (lo_i, lo, "L")
    return piv, tentative


def structure(piv, close):
    """第1讲/道氏：HH+HL 上升，LH+LL 下降。"""
    H = [p for p in piv if p[2] == "H"]
    L = [p for p in piv if p[2] == "L"]
    if len(H) < 2 or len(L) < 2:
        return {"state": "数据不足", "vote": 0}
    hh, hl = H[-1][1] > H[-2][1], L[-1][1] > L[-2][1]
    if hh and hl:
        state, vote = "上升（HH+HL）", 1
        if close < L[-1][1]:
            state, vote = "上升结构被破坏（跌破最近低点）", -0.5
    elif not hh and not hl:
        state, vote = "下降（LH+LL）", -1
        if close > H[-1][1]:
            state, vote = "下降结构被破坏（突破最近高点）", 0.5
    else:
        state, vote = ("高点抬高、低点降低（扩散）" if hh else "高点降低、低点抬高（收敛）"), 0
    return {"state": state, "vote": vote,
            "last_highs": [round(p[1], 6) for p in H[-3:]],
            "last_lows": [round(p[1], 6) for p in L[-3:]]}


def line_at(p1, p2, x):
    (x1, y1), (x2, y2) = p1, p2
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1) if x2 != x1 else y2


def trendlines(piv, n_last):
    """第1讲：连最近两个低点得支撑线，连最近两个高点得压力线，外推到当前 K 线。"""
    H = [p for p in piv if p[2] == "H"][-2:]
    L = [p for p in piv if p[2] == "L"][-2:]
    out = {}
    if len(L) == 2:
        out["support_line"] = line_at((L[0][0], L[0][1]), (L[1][0], L[1][1]), n_last)
        out["support_slope"] = "上行" if L[1][1] > L[0][1] else "下行"
    if len(H) == 2:
        out["resist_line"] = line_at((H[0][0], H[0][1]), (H[1][0], H[1][1]), n_last)
        out["resist_slope"] = "上行" if H[1][1] > H[0][1] else "下行"
    return out


# ---------------- 形态 ----------------
def chart_patterns(piv, close, a_now):
    """第5-12讲：双顶底、头肩、三角形、楔形、矩形、扩散。基于最近的 zigzag 转折点。"""
    res = []
    H = [p for p in piv if p[2] == "H"]
    L = [p for p in piv if p[2] == "L"]
    tol = 0.6 * a_now
    # 双顶 / 双底
    if len(H) >= 2 and len(L) >= 1 and abs(H[-1][1] - H[-2][1]) < tol and L[-1][0] > H[-2][0]:
        neck = L[-1][1]
        res.append({"name": "双顶(M头)", "neckline": neck,
                    "status": "已跌破颈线（确认）" if close < neck else "未破颈线（提醒）",
                    "target": neck - (max(H[-1][1], H[-2][1]) - neck)})
    if len(L) >= 2 and len(H) >= 1 and abs(L[-1][1] - L[-2][1]) < tol and H[-1][0] > L[-2][0]:
        neck = H[-1][1]
        res.append({"name": "双底(W底)", "neckline": neck,
                    "status": "已突破颈线（确认）" if close > neck else "未破颈线（提醒）",
                    "target": neck + (neck - min(L[-1][1], L[-2][1]))})
    # 头肩
    if len(H) >= 3 and len(L) >= 2:
        l, m, r = H[-3][1], H[-2][1], H[-1][1]
        if m > l and m > r and abs(l - r) < 1.5 * a_now and m - max(l, r) > 0.8 * a_now:
            neck = (L[-2][1] + L[-1][1]) / 2
            res.append({"name": "头肩顶", "neckline": neck,
                        "status": "已跌破颈线（确认）" if close < neck else "右肩已现，未破颈线（提醒）",
                        "target": neck - (m - neck)})
    if len(L) >= 3 and len(H) >= 2:
        l, m, r = L[-3][1], L[-2][1], L[-1][1]
        if m < l and m < r and abs(l - r) < 1.5 * a_now and min(l, r) - m > 0.8 * a_now:
            neck = (H[-2][1] + H[-1][1]) / 2
            res.append({"name": "头肩底", "neckline": neck,
                        "status": "已突破颈线（确认）" if close > neck else "右肩已现，未破颈线（提醒）",
                        "target": neck + (neck - m)})
    # 收敛 / 平行形态：最近 3 个高点与 3 个低点的拟合斜率（单位：ATR/根）
    if len(H) >= 3 and len(L) >= 3:
        hx, hy = np.array([p[0] for p in H[-3:]]), np.array([p[1] for p in H[-3:]])
        lx, ly = np.array([p[0] for p in L[-3:]]), np.array([p[1] for p in L[-3:]])
        sh, sl = np.polyfit(hx, hy, 1)[0] / a_now, np.polyfit(lx, ly, 1)[0] / a_now
        eps = 0.02
        flat_h, flat_l = abs(sh) < eps, abs(sl) < eps
        name = None
        if flat_h and flat_l: name = "矩形（中继）"
        elif flat_h and sl > eps: name = "上升三角形"
        elif sh < -eps and flat_l: name = "下降三角形"
        elif sh < -eps and sl > eps: name = "对称三角形"
        elif sh > eps and sl < -eps: name = "扩散三角形（喇叭形，常为反转）"
        elif sh > eps and sl > eps: name = "上升楔形（看跌）" if sl > sh * 1.2 else "上升通道"
        elif sh < -eps and sl < -eps: name = "下降楔形（看涨）" if sh < sl * 1.2 else "下降通道"
        if name:
            n_last = max(H[-1][0], L[-1][0])
            up = np.polyval(np.polyfit(hx, hy, 1), n_last)
            dn = np.polyval(np.polyfit(lx, ly, 1), n_last)
            res.append({"name": name, "upper_now": float(up), "lower_now": float(dn),
                        "slope_high_atr_per_bar": round(sh, 3), "slope_low_atr_per_bar": round(sl, 3)})
    return res


def candle_patterns(df):
    """第18讲 K 线线态（最近 3 根已收盘 K 线）。"""
    out = []
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    i = len(df) - 1
    body = abs(c[i] - o[i]); rng = h[i] - l[i] or 1e-12
    up_w, lo_w = h[i] - max(c[i], o[i]), min(c[i], o[i]) - l[i]
    if body <= 0.1 * rng: out.append("十字星（收敛/犹豫）")
    if lo_w >= 2 * body and up_w <= 0.3 * rng and body > 0: out.append("锤子线/下影线探底")
    if up_w >= 2 * body and lo_w <= 0.3 * rng and body > 0: out.append("射击之星/上影线冲高回落")
    pb = abs(c[i - 1] - o[i - 1])
    if c[i] > o[i] and c[i - 1] < o[i - 1] and c[i] >= o[i - 1] and o[i] <= c[i - 1] and body > pb:
        out.append("看涨吞没")
    if c[i] < o[i] and c[i - 1] > o[i - 1] and c[i] <= o[i - 1] and o[i] >= c[i - 1] and body > pb:
        out.append("看跌吞没")
    b2 = abs(c[i - 2] - o[i - 2]); b1 = abs(c[i - 1] - o[i - 1])
    if c[i - 2] < o[i - 2] and b1 < 0.4 * b2 and c[i] > o[i] and c[i] > (o[i - 2] + c[i - 2]) / 2:
        out.append("早晨之星")
    if c[i - 2] > o[i - 2] and b1 < 0.4 * b2 and c[i] < o[i] and c[i] < (o[i - 2] + c[i - 2]) / 2:
        out.append("黄昏之星")
    avg_body = np.mean(np.abs(c[-21:-1] - o[-21:-1]))
    if body > 2 * avg_body: out.append("大实体K线（发散）" + ("阳" if c[i] > o[i] else "阴"))
    return out


def divergence(piv, tent, series, name, price_close_trend):
    """第3/13/14/38讲 背离：价格创新高/低，而指标没有。含正在形成的（未确认）背离。"""
    out = []
    H = [p for p in piv if p[2] == "H"]
    L = [p for p in piv if p[2] == "L"]
    v = series.values
    def val(i): return v[max(0, i - 1):i + 2].max(), v[max(0, i - 1):i + 2].min()
    if len(H) >= 2 and H[-1][1] > H[-2][1] and val(H[-1][0])[0] < val(H[-2][0])[0]:
        out.append(f"{name}顶背离（已确认高点）")
    if len(L) >= 2 and L[-1][1] < L[-2][1] and val(L[-1][0])[1] > val(L[-2][0])[1]:
        out.append(f"{name}底背离（已确认低点）")
    if tent[2] == "H" and H and tent[1] > H[-1][1] and val(tent[0])[0] < val(H[-1][0])[0]:
        out.append(f"{name}顶背离（正在形成）")
    if tent[2] == "L" and L and tent[1] < L[-1][1] and val(tent[0])[1] > val(L[-1][0])[1]:
        out.append(f"{name}底背离（正在形成）")
    return out


def pct_rank(s, window=120):
    w = s.dropna().iloc[-window:]
    return float((w < w.iloc[-1]).mean() * 100) if len(w) else np.nan


# ---------------- 单周期完整分析 ----------------
def analyze_tf(df, tf, zz_mult=2.0):
    c = df.c
    a = atr(df)
    A = float(a.iloc[-1])
    close = float(c.iloc[-1])
    r = {"tf": tf, "close": close, "atr": A, "atr_pct": A / close * 100,
         "last_bar_time": str(df.time.iloc[-1])}

    # 均线组（第2讲斐波那契排列）
    fib = {n: ema(c, n) for n in (5, 8, 13, 21, 34, 55)}
    vals = [float(fib[n].iloc[-1]) for n in (5, 8, 13, 21, 34, 55)]
    spread = (max(vals) - min(vals)) / A
    if all(vals[i] > vals[i + 1] for i in range(5)): ma_state, ma_vote = "多头排列", 1
    elif all(vals[i] < vals[i + 1] for i in range(5)): ma_state, ma_vote = "空头排列", -1
    else: ma_state, ma_vote = "交织", 0
    prev = [float(fib[n].iloc[-6]) for n in (5, 8, 13, 21, 34, 55)]
    prev_spread = (max(prev) - min(prev)) / A
    rhythm = "汇聚" if spread < 1.0 else ("发散中" if spread > prev_spread * 1.1 else ("收敛中" if spread < prev_spread * 0.9 else "平行"))
    r["ema_group"] = {"state": ma_state, "rhythm": rhythm, "spread_atr": round(spread, 2),
                      **{f"ema{n}": float(fib[n].iloc[-1]) for n in fib}}

    # 三均线（第41讲）
    e5, e13, e34 = vals[0], vals[2], vals[4]
    r["three_ema"] = "做多（5上穿13与34）" if e5 > e13 and e5 > e34 else (
        "做空（5下穿13与34）" if e5 < e13 and e5 < e34 else "休息（5夹在13与34之间）")

    # 葛兰威尔乖离（第24讲），用 ATR 单位衡量
    e20 = float(ema(c, 20).iloc[-1])
    r["bias_ema20_atr"] = round((close - e20) / A, 2)
    r["bias_ema20_pct"] = round((close / e20 - 1) * 100, 2)

    # 顾比 GMMA（第26讲）
    S = [ema(c, n) for n in (3, 5, 8, 10, 12, 15)]
    Lg = [ema(c, n) for n in (30, 35, 40, 45, 50, 60)]
    s_now = [float(x.iloc[-1]) for x in S]; l_now = [float(x.iloc[-1]) for x in Lg]
    l_prev = [float(x.iloc[-6]) for x in Lg]
    s_w, l_w = (max(s_now) - min(s_now)) / A, (max(l_now) - min(l_now)) / A
    l_w_prev = (max(l_prev) - min(l_prev)) / A
    if min(s_now) > max(l_now): g_state, g_vote = "短期组在长期组上方（多头）", 1
    elif max(s_now) < min(l_now): g_state, g_vote = "短期组在长期组下方（空头）", -1
    else: g_state, g_vote = "短期组渗透/交织于长期组", 0
    sep = (min(s_now) - max(l_now)) / A if g_vote >= 0 else (min(l_now) - max(s_now)) / A
    r["gmma"] = {"state": g_state, "short_width_atr": round(s_w, 2), "long_width_atr": round(l_w, 2),
                 "long_group": "扩散" if l_w > l_w_prev * 1.05 else ("收缩" if l_w < l_w_prev * 0.95 else "稳定"),
                 "separation_atr": round(sep, 2)}

    # 维加斯（第31讲）
    v144, v169 = ema(c, 144), ema(c, 169)
    vt, vb = max(v144.iloc[-1], v169.iloc[-1]), min(v144.iloc[-1], v169.iloc[-1])
    slope = (v144.iloc[-1] - v144.iloc[-11]) / A
    if close > vt and slope > 0: vg_state, vg_vote = "价格在通道上方且通道上行", 1
    elif close < vb and slope < 0: vg_state, vg_vote = "价格在通道下方且通道下行", -1
    elif vb <= close <= vt: vg_state, vg_vote = "价格在通道内部", 0
    else: vg_state, vg_vote = "价格与通道方向不一致", 0
    r["vegas"] = {"ema144": float(v144.iloc[-1]), "ema169": float(v169.iloc[-1]), "state": vg_state,
                  "slope_10bars_atr": round(float(slope), 2), "width_atr": round(float((vt - vb) / A), 2)}

    # 一目均衡（第32讲）
    ten, kij, sa, sb, sa_raw, sb_raw = ichimoku(df)
    ct, cb = max(sa.iloc[-1], sb.iloc[-1]), min(sa.iloc[-1], sb.iloc[-1])
    chikou_up = close > c.iloc[-27]
    bull = [close > ct, close > kij.iloc[-1], ten.iloc[-1] > kij.iloc[-1], chikou_up]
    bear = [close < cb, close < kij.iloc[-1], ten.iloc[-1] < kij.iloc[-1], not chikou_up]
    ich_vote = 1 if sum(bull) >= 3 else (-1 if sum(bear) >= 3 else 0)
    r["ichimoku"] = {"tenkan": float(ten.iloc[-1]), "kijun": float(kij.iloc[-1]),
                     "cloud_top": float(ct), "cloud_bottom": float(cb),
                     "price_vs_cloud": "云上" if close > ct else ("云下" if close < cb else "云中"),
                     "bull_conditions": int(sum(bull)), "bear_conditions": int(sum(bear)),
                     "future_cloud": "阳云(A>B)" if sa_raw.iloc[-1] > sb_raw.iloc[-1] else "阴云(A<B)",
                     "cloud_thickness_atr": round(float((ct - cb) / A), 2)}

    # MACD（第3/4讲）
    dif, dea, hist = macd(c)
    m_vote = 1 if dif.iloc[-1] > 0 and dea.iloc[-1] > 0 else (-1 if dif.iloc[-1] < 0 and dea.iloc[-1] < 0 else 0)
    cross = "金叉" if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2] else (
        "死叉" if dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2] else ("DIF在DEA上" if dif.iloc[-1] > dea.iloc[-1] else "DIF在DEA下"))
    r["macd"] = {"dif": float(dif.iloc[-1]), "dea": float(dea.iloc[-1]), "hist": float(hist.iloc[-1]),
                 "hist_trend": "柱线放大" if abs(hist.iloc[-1]) > abs(hist.iloc[-2]) else "柱线缩短",
                 "hist_rising": bool(hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3]),
                 "hist_falling": bool(hist.iloc[-1] < hist.iloc[-2] < hist.iloc[-3]),
                 "zero_axis": "零轴上方" if m_vote == 1 else ("零轴下方" if m_vote == -1 else "零轴附近交错"),
                 "cross": cross}

    # BBI / PBX（第35/36讲）
    b = float(bbi(c).iloc[-1])
    px = {m: float(s.iloc[-1]) for m, s in pbx(c).items()}
    p_vote = 1 if close > max(px.values()) else (-1 if close < min(px.values()) else 0)
    pv = list(px.values())
    p_order = "多头排列" if all(pv[i] > pv[i + 1] for i in range(5)) else ("空头排列" if all(pv[i] < pv[i + 1] for i in range(5)) else "交织")
    r["bbi"] = {"value": b, "state": "价格在BBI上方（多头）" if close > b else "价格在BBI下方（空头）"}
    r["pbx"] = {"state": "价格在全部瀑布线上方" if p_vote == 1 else ("价格在全部瀑布线下方" if p_vote == -1 else "价格在瀑布线内部（阵地争夺，不宜开仓）"),
                "order": p_order, "lines": px}

    # 震荡类指标（第13/14/38/39讲）
    k, d, j = kdj(df)
    rs = rsi(c)
    cc = cci(df)
    wr = williams_r(df)
    r["oscillators"] = {"rsi14": float(rs.iloc[-1]), "rsi6": float(rsi(c, 6).iloc[-1]),
                        "k": float(k.iloc[-1]), "d": float(d.iloc[-1]), "j": float(j.iloc[-1]),
                        "kd_cross": "K上穿D" if k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2] else (
                            "K下穿D" if k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2] else ""),
                        "cci14": float(cc.iloc[-1]), "cci_prev": float(cc.iloc[-2]),
                        "wr10": float(wr.iloc[-1])}

    # 布林（第15讲）
    up, mid, lo = boll(c)
    bw = (up - lo) / mid
    r["boll"] = {"upper": float(up.iloc[-1]), "mid": float(mid.iloc[-1]), "lower": float(lo.iloc[-1]),
                 "pct_b": float((close - lo.iloc[-1]) / (up.iloc[-1] - lo.iloc[-1])),
                 "width_pct_rank_120": round(pct_rank(bw), 1),
                 "width_change": "开口" if bw.iloc[-1] > bw.iloc[-4] * 1.08 else ("收口" if bw.iloc[-1] < bw.iloc[-4] * 0.92 else "平稳"),
                 "position": "中轨与上轨之间（强）" if close > mid.iloc[-1] else "中轨与下轨之间（弱）"}

    # ADX
    pdi, ndi, ad = adx(df)
    r["adx"] = {"adx": float(ad.iloc[-1]), "adx_prev5": float(ad.iloc[-6]), "+di": float(pdi.iloc[-1]), "-di": float(ndi.iloc[-1])}

    # 结构、趋势线、形态、背离
    piv, tent = zigzag(df, a, zz_mult)
    st = structure(piv, close)
    r["structure"] = st
    r["pivots_recent"] = [{"time": str(df.time.iloc[i]), "price": p, "type": t} for i, p, t in piv[-6:]]
    r["tentative_extreme"] = {"time": str(df.time.iloc[tent[0]]), "price": tent[1], "type": tent[2]}
    tl = trendlines(piv, len(df) - 1)
    r["trendlines"] = {k2: (float(v) if isinstance(v, (float, np.floating)) else v) for k2, v in tl.items()}
    r["chart_patterns"] = chart_patterns(piv, close, A)
    r["candle_patterns"] = candle_patterns(df)
    dv = []
    for s, nm in ((dif, "MACD"), (rs, "RSI"), (d, "KD"), (cc, "CCI")):
        dv += divergence(piv, tent, s, nm, st["vote"])
    r["divergences"] = dv

    # ASI 领先突破（第34讲）
    asv = asi(df)
    H = [p for p in piv if p[2] == "H"]; L = [p for p in piv if p[2] == "L"]
    lead = []
    if H and asv.iloc[-1] > asv.iloc[H[-1][0]] and close < H[-1][1]:
        lead.append("ASI已先于价格突破前高（看涨领先）")
    if L and asv.iloc[-1] < asv.iloc[L[-1][0]] and close > L[-1][1]:
        lead.append("ASI已先于价格跌破前低（看跌领先）")
    r["asi_lead"] = lead

    # 速率分形（第22讲）：当前腿与上一腿的速度（ATR/根）
    legs = []
    pts = piv[-2:] + [tent]
    for (i1, p1, _), (i2, p2, _) in zip(pts[:-1], pts[1:]):
        bars = max(1, i2 - i1)
        legs.append({"from": round(p1, 6), "to": round(p2, 6), "bars": bars,
                     "move_atr": round((p2 - p1) / A, 2), "speed_atr_per_bar": round((p2 - p1) / A / bars, 3)})
    cur_bars = max(1, len(df) - 1 - tent[0])
    legs.append({"from": round(tent[1], 6), "to": close, "bars": cur_bars,
                 "move_atr": round((close - tent[1]) / A, 2), "speed_atr_per_bar": round((close - tent[1]) / A / cur_bars, 3),
                 "note": "从最近极值到当前"})
    r["speed_legs"] = legs

    # 斐波那契（第16/30讲）：最近一段完整波段
    if len(piv) >= 2:
        (ia, pa, ta_), (ib, pb_, tb) = piv[-2], piv[-1]
        if tent[0] > ib and abs(tent[1] - pb_) > abs(pb_ - pa) * 0.9:  # 当前腿已比上一腿长，用 B→当前极值
            (ia, pa, ta_), (ib, pb_, tb) = piv[-1], tent
        rng = pb_ - pa
        retr = {str(x): pb_ - x * rng for x in (0.236, 0.382, 0.5, 0.618, 0.764)}
        ext = {str(x): pa + x * rng for x in (1.382, 1.618, 2.0)}
        r["fib"] = {"swing_from": pa, "swing_to": pb_, "direction": "上涨波段" if rng > 0 else "下跌波段",
                    "retracement": retr, "extension_nonstandard": ext}
        if tent[0] > ib and piv[-1][0] == ib:  # 标准扩展 A-B-C
            C = tent[1]
            r["fib"]["extension_standard_from_C"] = {"C": C, **{str(x): C + x * rng for x in (0.618, 1.0, 1.618)}}

    # 趋势投票
    votes = {"结构": st["vote"], "均线组": ma_vote, "顾比": g_vote, "维加斯": vg_vote,
             "一目": ich_vote, "MACD零轴": m_vote, "BBI": 1 if close > b else -1, "瀑布线": p_vote}
    r["trend_votes"] = votes
    r["trend_score"] = round(sum(votes.values()) / len(votes) * 3, 2)

    # 行情性质（第27讲）
    adxv = r["adx"]["adx"]; bwr = r["boll"]["width_pct_rank_120"]; ts_ = r["trend_score"]
    if abs(ts_) >= 1.5 and adxv >= 22: regime = "单边"
    elif bwr <= 20 and adxv < 22: regime = "收缩"
    elif bwr >= 80 and abs(ts_) < 1: regime = "发散"
    else: regime = "区间"
    r["regime"] = regime
    return r, piv, a
