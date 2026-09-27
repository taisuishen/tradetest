"""
缠论（简化实现，基于笔中枢）：
  K 线包含处理 → 顶底分型 → 笔（两分型之间至少隔 3 根合并 K 线）→ 笔中枢（连续 3 笔重叠区间 [ZD, ZG]）
  → 走势（相邻两个中枢上移 / 下移）→ 背驰（离开中枢的笔 MACD 面积小于进入中枢的笔）→ 三类买卖点

返回的时间点都是“信号确认的 K 线下标”，调用方据此判断信号是否足够新。
"""
import numpy as np

import ta


def merge_inclusion(h, l):
    """包含处理：向上时取高高、低高，向下时取高低、低低。返回 [{h, l, end(原始下标), hi_i, lo_i}]。"""
    m = []
    for i in range(len(h)):
        if m:
            last = m[-1]
            if (h[i] <= last["h"] and l[i] >= last["l"]) or (h[i] >= last["h"] and l[i] <= last["l"]):
                up = True if len(m) < 2 else last["h"] > m[-2]["h"]
                if up:
                    if h[i] > last["h"]: last["hi_i"] = i
                    last["h"] = max(h[i], last["h"]); last["l"] = max(l[i], last["l"])
                else:
                    if l[i] < last["l"]: last["lo_i"] = i
                    last["h"] = min(h[i], last["h"]); last["l"] = min(l[i], last["l"])
                last["end"] = i
                continue
        m.append({"h": h[i], "l": l[i], "end": i, "hi_i": i, "lo_i": i})
    return m


def strokes(m):
    """分型 → 笔。返回端点列表 [(合并下标k, 'top'/'bot', 价格, 原始下标)]，最后一个端点尚未被反向分型确认。"""
    fr = []
    for k in range(1, len(m) - 1):
        if m[k]["h"] > m[k - 1]["h"] and m[k]["h"] > m[k + 1]["h"]:
            fr.append((k, "top", m[k]["h"], m[k]["hi_i"]))
        elif m[k]["l"] < m[k - 1]["l"] and m[k]["l"] < m[k + 1]["l"]:
            fr.append((k, "bot", m[k]["l"], m[k]["lo_i"]))
    pts = []
    for f in fr:
        if not pts:
            pts.append(f); continue
        last = pts[-1]
        if f[1] == last[1]:
            if (f[1] == "top" and f[2] > last[2]) or (f[1] == "bot" and f[2] < last[2]):
                pts[-1] = f
        elif f[0] - last[0] >= 4 and ((f[1] == "top" and f[2] > last[2]) or (f[1] == "bot" and f[2] < last[2])):
            pts.append(f)
    return pts


def analyze(df, recent_bars=24):
    h, l, c = df.h.values, df.l.values, df.c.values
    n = len(df)
    m = merge_inclusion(h, l)
    pts = strokes(m)
    _, _, hist = ta.macd(df.c)
    hist = hist.values
    bis = []
    for a, b in zip(pts[:-1], pts[1:]):
        up = b[1] == "top"
        i0, i1 = a[3], b[3]
        seg = hist[i0:i1 + 1]
        area = float(seg[seg > 0].sum()) if up else float(-seg[seg < 0].sum())
        bis.append({"up": up, "high": max(a[2], b[2]), "low": min(a[2], b[2]), "start": a[3], "end": b[3],
                    "end_k": b[0], "area": area, "from": a[2], "to": b[2]})
    # 笔中枢
    zs, i = [], 0
    while i + 2 < len(bis):
        three = bis[i:i + 3]
        zg, zd = min(b["high"] for b in three), max(b["low"] for b in three)
        if zg > zd:
            j = i + 3
            while j < len(bis) and bis[j]["low"] <= zg and bis[j]["high"] >= zd:
                j += 1
            zs.append({"zg": zg, "zd": zd, "gg": max(b["high"] for b in bis[i:j]), "dd": min(b["low"] for b in bis[i:j]),
                       "first": i, "last": j - 1})
            i = j
        else:
            i += 1
    out = {"bis": bis, "zs": zs, "trend": "盘整", "points": [], "stroke_points": [p[2] for p in pts[-10:]]}
    if len(zs) >= 2:
        a, b = zs[-2], zs[-1]
        out["trend"] = "上涨" if b["zd"] > a["zg"] else ("下跌" if b["zg"] < a["zd"] else "盘整")
    if not zs:
        return out
    # 三类买卖点：中枢最后一笔向上离开（高于 ZG），紧接着的回抽笔完全不回中枢（低点仍在 ZG 之上）→ 三买；三卖反之。
    # 中枢的延伸规则是“与 [ZD, ZG] 有重叠就并入”，所以回抽笔就是中枢之后第一根不重叠的笔。
    for Zk in zs:
        if Zk["last"] + 1 < len(bis):
            leave, back = bis[Zk["last"]], bis[Zk["last"] + 1]
            if leave["up"] and not back["up"] and back["low"] > Zk["zg"]:
                out["points"].append({"type": "三买", "price": back["low"], "i": back["end"]})
            if not leave["up"] and back["up"] and back["high"] < Zk["zd"]:
                out["points"].append({"type": "三卖", "price": back["high"], "i": back["end"]})
    Z = zs[-1]
    after = bis[Z["last"] + 1:]
    # 一类买卖点：趋势中离开最后中枢的笔创新高/新低，但 MACD 面积小于进入中枢的同向笔（背驰）
    enter = bis[Z["first"] - 1] if Z["first"] >= 1 else None
    for k, bb in enumerate(after):
        if enter is None or bb["up"] != enter["up"]:
            continue
        if not bb["up"] and out["trend"] == "下跌" and bb["low"] < Z["dd"] and bb["area"] < enter["area"]:
            out["points"].append({"type": "一买", "price": bb["low"], "i": bb["end"]})
            nxt = after[k + 2] if k + 2 < len(after) else None
            if nxt and not nxt["up"] and nxt["low"] > bb["low"]:
                out["points"].append({"type": "二买", "price": nxt["low"], "i": nxt["end"]})
        if bb["up"] and out["trend"] == "上涨" and bb["high"] > Z["gg"] and bb["area"] < enter["area"]:
            out["points"].append({"type": "一卖", "price": bb["high"], "i": bb["end"]})
            nxt = after[k + 2] if k + 2 < len(after) else None
            if nxt and nxt["up"] and nxt["high"] < bb["high"]:
                out["points"].append({"type": "二卖", "price": nxt["high"], "i": nxt["end"]})
        # 盘整背驰（《缠论精解》：背驰分趋势背驰与盘整背驰）：只有一个中枢时，离开段创新高/新低但力度弱于进入段
        if out["trend"] == "盘整":
            if not bb["up"] and bb["low"] < Z["dd"] and bb["area"] < enter["area"]:
                out["points"].append({"type": "盘背买", "price": bb["low"], "i": bb["end"]})
            if bb["up"] and bb["high"] > Z["gg"] and bb["area"] < enter["area"]:
                out["points"].append({"type": "盘背卖", "price": bb["high"], "i": bb["end"]})
    # 端点在下标 i 处，需要后面再走出 1 根确认分型
    out["points"] = [p for p in out["points"] if p["i"] < n - 1]
    out["recent"] = [p["type"] for p in out["points"] if p["i"] >= n - recent_bars]
    out["last_zs"] = {"zg": Z["zg"], "zd": Z["zd"]}
    out["zs_levels"] = [(z["zg"], z["zd"]) for z in zs[-2:]]
    return out


def bias(ch):
    """+1 偏多 / -1 偏空 / 0 中性：走势 + 近期买卖点。"""
    r = set(ch.get("recent", []))
    if r & {"一卖", "三卖"}:
        return -1
    if r & {"一买", "三买"}:
        return 1
    if ch["trend"] == "上涨" or "二买" in r:
        return 1
    if ch["trend"] == "下跌" or "二卖" in r:
        return -1
    return 0
