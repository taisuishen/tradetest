"""
OKX 永续合约分析器：按《方法论.md》的五步流程，结合 K 线、订单簿和合约数据生成报告。

用法：
    python okx_analyzer.py                  # 默认 BTC-USDT-SWAP
    python okx_analyzer.py ETH-USDT-SWAP
    python okx_analyzer.py SOL-USDT-SWAP --no-llm   # 只出数据，不调用大模型
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import okx_client as ox
import ta

ROOT = Path(__file__).parent
_cfg_file = ROOT / "config.json" if (ROOT / "config.json").exists() else ROOT / "config.example.json"
CFG = json.loads(_cfg_file.read_text(encoding="utf-8"))
if os.environ.get("LLM_API_KEY"):  # 服务器上可用环境变量提供 Key，避免写进文件
    CFG["llm"]["api_key"] = os.environ["LLM_API_KEY"]
TZ = timezone(timedelta(hours=8))
TFS = [("1D", 300, 4), ("4H", 500, 3), ("1H", 500, 2), ("15m", 300, 1)]  # 周期, K线数, 权重


def log(*a):
    print(*a, flush=True)


# ---------------- 订单簿 ----------------
def orderbook_analysis(inst, ct_val, price, n_snap, interval):
    """多次采样订单簿：计算各档位深度、不平衡，并识别持续存在的挂单墙（过滤闪现的诱单）。"""
    bucket = price * 0.0001  # 0.01% 价格分桶
    bands = (0.1, 0.25, 0.5, 0.9)
    snaps, wall_hits = [], {}
    for k in range(n_snap):
        b = ox.books_full(inst)
        asks = np.array([[float(x[0]), float(x[1])] for x in b["asks"]])
        bids = np.array([[float(x[0]), float(x[1])] for x in b["bids"]])
        mid = (asks[0, 0] + bids[0, 0]) / 2
        snap = {"mid": mid, "spread": asks[0, 0] - bids[0, 0],
                "coverage_pct": round(min(asks[-1, 0] / mid - 1, 1 - bids[-1, 0] / mid) * 100, 2)}
        for bd in bands:
            a_sz = asks[asks[:, 0] <= mid * (1 + bd / 100), 1].sum() * ct_val
            b_sz = bids[bids[:, 0] >= mid * (1 - bd / 100), 1].sum() * ct_val
            snap[f"±{bd}%"] = {"bid_coin": b_sz, "ask_coin": a_sz,
                               "imbalance": (b_sz - a_sz) / (b_sz + a_sz) if b_sz + a_sz else 0}
        for side, arr in (("bid", bids), ("ask", asks)):
            keys = np.floor(arr[:, 0] / bucket).astype(int)
            s = pd.Series(arr[:, 1] * ct_val).groupby(keys).sum()
            thr = max(s.median() * 4, s.quantile(0.95))
            for key, v in s[s >= thr].nlargest(10).items():
                wall_hits.setdefault((side, key), []).append(v)
        snaps.append(snap)
        if k < n_snap - 1:
            time.sleep(interval)
    last = snaps[-1]
    walls = []
    for (side, key), vs in wall_hits.items():
        if len(vs) >= max(2, n_snap - 1):  # 至少在 n-1 次采样中都出现
            p = (key + 0.5) * bucket
            walls.append({"side": "买单墙(支撑)" if side == "bid" else "卖单墙(阻力)",
                          "price": round(p, 6), "size_coin": round(float(np.mean(vs)), 4),
                          "size_usd": round(float(np.mean(vs)) * p), "dist_pct": round((p / last["mid"] - 1) * 100, 3),
                          "persist": f"{len(vs)}/{n_snap}"})
    walls.sort(key=lambda w: -w["size_usd"])
    flashy = sum(1 for vs in wall_hits.values() if len(vs) <= 1)
    imb_series = {bd: [s[f"±{bd}%"]["imbalance"] for s in snaps] for bd in bands}
    return {
        "mid": last["mid"], "spread": last["spread"], "coverage_pct": last["coverage_pct"],
        "depth": {f"±{bd}%": {"bid_coin": round(last[f"±{bd}%"]["bid_coin"], 3),
                              "ask_coin": round(last[f"±{bd}%"]["ask_coin"], 3),
                              "bid_usd": round(last[f"±{bd}%"]["bid_coin"] * last["mid"]),
                              "ask_usd": round(last[f"±{bd}%"]["ask_coin"] * last["mid"]),
                              "imbalance_last": round(last[f"±{bd}%"]["imbalance"], 3),
                              "imbalance_avg": round(float(np.mean(imb_series[bd])), 3),
                              "imbalance_std": round(float(np.std(imb_series[bd])), 3)} for bd in bands},
        "persistent_walls": walls[:12],
        "flash_walls_count": flashy,
        "snapshots": n_snap, "interval_sec": interval,
    }


# ---------------- 合约数据 ----------------
def derivatives_analysis(inst, ct_val, h1):
    base = inst.split("-")[0]
    out = {}
    # 资金费率
    f = ox.funding(inst)
    fh = ox.funding_history(inst, 90)
    rates = np.array([float(x["realizedRate"] or x["fundingRate"]) for x in fh])
    cur = float(f["fundingRate"])
    out["funding"] = {"current": cur, "current_pct": cur * 100,
                      "next_funding_time": datetime.fromtimestamp(int(f["fundingTime"]) / 1000, TZ).strftime("%m-%d %H:%M"),
                      "avg_30d_pct": float(rates.mean() * 100), "max_30d_pct": float(rates.max() * 100),
                      "min_30d_pct": float(rates.min() * 100),
                      "pct_rank_30d": float((rates < cur).mean() * 100),
                      "last_9_pct": [round(x * 100, 4) for x in rates[:9]],
                      "annualized_pct": cur * 3 * 365 * 100}
    # 持仓量与价格四象限
    oi = ox.open_interest(inst)
    oih = pd.DataFrame(ox.oi_history(inst, "1H", 100), columns=["ts", "oi", "oiCcy", "oiUsd"]).astype(float)
    oih = oih.sort_values("ts").reset_index(drop=True)
    px = h1.set_index("ts").c
    def chg(hours):
        o_now, o_then = oih.oiCcy.iloc[-1], oih.oiCcy.iloc[-1 - hours]
        t_then = oih.ts.iloc[-1 - hours]
        p_then = px[px.index <= t_then].iloc[-1] if (px.index <= t_then).any() else px.iloc[0]
        p_now = px.iloc[-1]
        dp, do = (p_now / p_then - 1) * 100, (o_now / o_then - 1) * 100
        quad = ("价涨OI增：新多入场（上涨健康）" if dp > 0 and do > 0 else
                "价涨OI减：空头回补（上涨偏弱）" if dp > 0 else
                "价跌OI增：新空入场（下跌健康）" if do > 0 else "价跌OI减：多头离场（下跌或近衰竭）")
        return {"price_chg_pct": round(dp, 2), "oi_chg_pct": round(do, 2), "quadrant": quad}
    out["open_interest"] = {"oi_coin": float(oi["oiCcy"]), "oi_usd": float(oi["oiUsd"]),
                            "4h": chg(4), "24h": chg(24), "72h": chg(72),
                            "oi_pct_rank_100h": float((oih.oiCcy < oih.oiCcy.iloc[-1]).mean() * 100)}
    # 多空比
    def ratio(fn, name):
        d = pd.DataFrame(fn(inst, "1H", 100), columns=["ts", "r"]).astype(float).sort_values("ts")
        return {"now": round(d.r.iloc[-1], 3), "24h_ago": round(d.r.iloc[-25], 3),
                "pct_rank_100h": round(float((d.r < d.r.iloc[-1]).mean() * 100), 1),
                "max_100h": round(d.r.max(), 3), "min_100h": round(d.r.min(), 3)}
    out["long_short"] = {"all_accounts": ratio(ox.ls_ratio_all, "全部账户多空人数比"),
                         "top_trader_accounts": ratio(ox.ls_ratio_top_account, "大户多空人数比"),
                         "top_trader_positions": ratio(ox.ls_ratio_top_position, "大户多空持仓比")}
    # 主动买卖量 / CVD
    def taker(period, limit):
        d = pd.DataFrame(ox.taker_volume(inst, period, limit), columns=["ts", "sell", "buy"]).astype(float).sort_values("ts")
        d["net"] = d.buy - d.sell
        return d
    t1 = taker("1H", 100)
    t5 = taker("5m", 100)
    def win(d, n):
        w = d.iloc[-n:]
        return {"buy": round(w.buy.sum() * ct_val, 2), "sell": round(w.sell.sum() * ct_val, 2),
                "net_coin": round(w.net.sum() * ct_val, 2), "buy_ratio": round(w.buy.sum() / (w.buy.sum() + w.sell.sum()), 3)}
    cvd = (t1.net.cumsum() * ct_val)
    # CVD 与价格背离：近 24h 价格新高/新低而 CVD 未跟
    p24 = h1.c.iloc[-24:]
    cvd24 = cvd.iloc[-24:]
    cvd_div = []
    if p24.iloc[-1] >= p24.max() * 0.999 and cvd24.iloc[-1] < cvd24.max() - 0.25 * (cvd24.max() - cvd24.min()):
        cvd_div.append("价格接近24h新高，但CVD未创新高（量价顶背离）")
    if p24.iloc[-1] <= p24.min() * 1.001 and cvd24.iloc[-1] > cvd24.min() + 0.25 * (cvd24.max() - cvd24.min()):
        cvd_div.append("价格接近24h新低，但CVD未创新低（量价底背离）")
    out["taker_flow"] = {"unit": base, "last_1h": win(t1, 1), "last_4h": win(t1, 4), "last_24h": win(t1, 24),
                         "last_72h": win(t1, 72), "last_30m_5m": win(t5, 6), "last_2h_5m": win(t5, 24),
                         "cvd_divergence": cvd_div}
    # 逐笔成交（最近 500 笔）
    tr = pd.DataFrame(ox.trades(inst, 500))
    tr["sz"] = tr.sz.astype(float) * ct_val; tr["px"] = tr.px.astype(float); tr["ts"] = tr.ts.astype("int64")
    big = tr[tr.sz >= tr.sz.quantile(0.98)]
    span = (tr.ts.max() - tr.ts.min()) / 1000
    out["recent_trades"] = {"count": len(tr), "span_sec": round(span, 1),
                            "buy_coin": round(tr[tr.side == "buy"].sz.sum(), 3),
                            "sell_coin": round(tr[tr.side == "sell"].sz.sum(), 3),
                            "big_trade_threshold_coin": round(float(tr.sz.quantile(0.98)), 4),
                            "big_buy_coin": round(big[big.side == "buy"].sz.sum(), 3),
                            "big_sell_coin": round(big[big.side == "sell"].sz.sum(), 3)}
    # 基差
    mk = float(ox.mark_price(inst)["markPx"])
    idx = float(ox.index_ticker("-".join(inst.split("-")[:2]))["idxPx"])
    out["basis"] = {"mark": mk, "index": idx, "premium_pct": round((mk / idx - 1) * 100, 4)}
    # 爆仓
    liq = ox.liquidations("-".join(inst.split("-")[:2]), 100)
    now_ms = time.time() * 1000
    agg = {}
    for x in liq:
        age_h = (now_ms - int(x["ts"])) / 3.6e6
        usd = float(x["sz"]) * ct_val * float(x["bkPx"])
        for w in (1, 4, 24):
            if age_h <= w:
                key = f"{w}h_{'空头爆仓' if x['posSide'] == 'short' else '多头爆仓'}_usd"
                agg[key] = agg.get(key, 0) + usd
    out["liquidations"] = {k: round(v) for k, v in sorted(agg.items())}
    out["liquidations"]["sample_note"] = f"接口仅返回最近 {len(liq)} 条爆仓单"
    return out


# ---------------- 成交密集区（第17讲） ----------------
def volume_profile(df, price):
    lo, hi = df.l.min(), df.h.max()
    step = price * 0.002
    edges = np.arange(lo, hi + step, step)
    vol = np.zeros(len(edges))
    for l, h, v in zip(df.l.values, df.h.values, df.volQuote.values):
        i0, i1 = int((l - lo) / step), int((h - lo) / step)
        vol[i0:i1 + 1] += v / (i1 - i0 + 1)
    s = pd.Series(vol, index=edges + step / 2)
    sm = s.rolling(3, center=True, min_periods=1).mean()
    peaks = [(p, v) for i, (p, v) in enumerate(sm.items())
             if 0 < i < len(sm) - 1 and v >= sm.iloc[i - 1] and v >= sm.iloc[i + 1] and v > sm.mean() * 1.3]
    peaks.sort(key=lambda x: -x[1])
    return {"poc": float(sm.idxmax()), "hvn": [float(p) for p, _ in peaks[:8]]}


# ---------------- 关键位置汇总（第17讲 位置分析） ----------------
def collect_levels(price, tfr, pivs, dfs, vp, ob):
    L = []
    def add(p, src, w):
        if p and np.isfinite(p) and abs(p / price - 1) < 0.15:
            L.append((float(p), src, w))
    wmap = {"1D": 3, "4H": 2, "1H": 1, "15m": 0.5}
    for tf in ("1D", "4H", "1H"):
        df = dfs[tf]
        for i, p, t in pivs[tf][-8:]:
            add(p, f"{tf}{'前高' if t == 'H' else '前低'}", wmap[tf])
    add(vp["poc"], "成交量最密集价(POC)", 3)
    for p in vp["hvn"]:
        add(p, "成交密集区", 1.5)
    for tf, w in (("1D", 2), ("4H", 1.5)):
        fb = tfr[tf].get("fib")
        if fb:
            for k, v in fb["retracement"].items():
                add(v, f"{tf}斐波回调{k}", w if k in ("0.382", "0.5", "0.618") else w / 2)
            for k, v in fb["extension_nonstandard"].items():
                add(v, f"{tf}斐波扩展{k}", w / 2)
            for k, v in fb.get("extension_standard_from_C", {}).items():
                if k != "C":
                    add(v, f"{tf}标准扩展{k}", w / 2)
    for tf, w in (("1D", 2), ("4H", 1.5), ("1H", 0.8)):
        r = tfr[tf]
        add(r["vegas"]["ema144"], f"{tf}维加斯EMA144", w)
        add(r["vegas"]["ema169"], f"{tf}维加斯EMA169", w)
        add(r["ema_group"]["ema55"], f"{tf}EMA55", w * 0.8)
        add(r["ichimoku"]["kijun"], f"{tf}基准线", w * 0.8)
        add(r["ichimoku"]["cloud_top"], f"{tf}云上沿", w * 0.7)
        add(r["ichimoku"]["cloud_bottom"], f"{tf}云下沿", w * 0.7)
        add(r["boll"]["upper"], f"{tf}布林上轨", w * 0.6)
        add(r["boll"]["lower"], f"{tf}布林下轨", w * 0.6)
        add(r["boll"]["mid"], f"{tf}布林中轨", w * 0.5)
        for k in ("support_line", "resist_line"):
            if k in r["trendlines"]:
                add(r["trendlines"][k], f"{tf}{'支撑趋势线' if k == 'support_line' else '压力趋势线'}", w * 0.7)
    for w_ in ob["persistent_walls"]:
        add(w_["price"], f"订单簿{w_['side']}{w_['size_usd'] / 1e6:.1f}M$", 1.5 if w_["size_usd"] > 5e6 else 1)
    mag = 10 ** int(np.floor(np.log10(price)))
    for m in (mag / 2, mag / 10):
        for k in range(-3, 4):
            add((np.floor(price / m) + k) * m, "整数关口", 0.5 if m == mag / 10 else 1)
    # 聚类：现价上下分开聚，单个簇的宽度不超过 0.6 ATR(4H)，避免链式合并成大区间
    a4 = tfr["4H"]["atr"]
    tol, max_span = 0.25 * a4, 0.6 * a4
    clusters = []
    for group in (sorted(x for x in L if x[0] < price), sorted(x for x in L if x[0] >= price)):
        start = None
        for p, src, w in group:
            if start is not None and p - clusters[-1]["hi"] <= tol and p - start <= max_span:
                c = clusters[-1]
                c["items"].append((p, src, w)); c["hi"] = p
            else:
                clusters.append({"items": [(p, src, w)], "hi": p}); start = p
    out = []
    for c in clusters:
        ws = sum(w for _, _, w in c["items"])
        pc = sum(p * w for p, _, w in c["items"]) / ws
        out.append({"price": round(pc, 6), "range": [round(c["items"][0][0], 6), round(c["hi"], 6)],
                    "strength": round(ws, 1), "sources": sorted({s for _, s, _ in c["items"]}),
                    "dist_pct": round((pc / price - 1) * 100, 2),
                    "dist_atr4h": round((pc - price) / tfr["4H"]["atr"], 2)})
    sup = sorted([x for x in out if x["price"] < price and x["strength"] >= 2], key=lambda x: -x["price"])[:7]
    res = sorted([x for x in out if x["price"] > price and x["strength"] >= 2], key=lambda x: x["price"])[:7]
    return sup, res


def breakout_check(h1, levels):
    """第29讲假突破识别：近 6 根 1H K 线对关键位的突破是否被收盘和下一根确认。"""
    out = []
    c, h, l = h1.c.values, h1.h.values, h1.l.values
    for lv in levels:
        p = lv["price"]
        for i in range(len(c) - 6, len(c)):
            if h[i] > p and c[i] < p and c[i - 1] < p:
                out.append(f"{str(h1.time.iloc[i])[5:16]} 上影刺破 {p:.6g} 后收回（疑似向上假突破）")
            if l[i] < p and c[i] > p and c[i - 1] > p:
                out.append(f"{str(h1.time.iloc[i])[5:16]} 下影刺破 {p:.6g} 后收回（疑似向下假突破）")
            if c[i] > p and c[i - 1] < p:
                conf = i + 1 < len(c) and c[i + 1] > p
                out.append(f"{str(h1.time.iloc[i])[5:16]} 收盘向上突破 {p:.6g}，" + ("下一根收盘确认" if conf else ("等待下一根确认" if i + 1 == len(c) else "下一根未确认（警惕假突破）")))
            if c[i] < p and c[i - 1] > p:
                conf = i + 1 < len(c) and c[i + 1] < p
                out.append(f"{str(h1.time.iloc[i])[5:16]} 收盘向下跌破 {p:.6g}，" + ("下一根收盘确认" if conf else ("等待下一根确认" if i + 1 == len(c) else "下一根未确认（警惕假突破）")))
    return out[-10:]


# ---------------- 综合评分与条件式计划 ----------------
def synthesize(price, tfr, sup, res, der, ob):
    w = {tf: wt for tf, _, wt in TFS}
    bias = sum(tfr[tf]["trend_score"] * w[tf] for tf in w) / sum(w.values())
    signs = [np.sign(tfr[tf]["trend_score"]) if abs(tfr[tf]["trend_score"]) >= 0.75 else 0 for tf in w]
    resonance = "多周期共振向上" if all(s > 0 for s in signs) else ("多周期共振向下" if all(s < 0 for s in signs) else "多周期发散")
    a4 = tfr["4H"]["atr"]
    # 计划锚点用强度 ≥5 的关键位（多个来源重合），弱位置只作参考
    key_s = [x for x in sup if x["strength"] >= 5]
    key_r = [x for x in res if x["strength"] >= 5]
    ns, nr = (key_s[0] if key_s else (sup[0] if sup else None)), (key_r[0] if key_r else (res[0] if res else None))
    sup = [x for x in sup if not ns or x["price"] <= ns["price"]]
    res = [x for x in res if not nr or x["price"] >= nr["price"]]
    if ns and nr:
        ds, dr = price - ns["price"], nr["price"] - price
        if ds <= 0.5 * a4: position = f"见位区：贴近支撑 {ns['price']:.6g}"
        elif dr <= 0.5 * a4: position = f"见位区：贴近阻力 {nr['price']:.6g}"
        else: position = f"间位区：距支撑 {ds / a4:.1f} ATR、距阻力 {dr / a4:.1f} ATR（课程不提倡在此追单）"
    else:
        position = "位置数据不足"
    # 合约确认分
    conf = 0; notes = []
    q = der["open_interest"]["24h"]["quadrant"]
    if "新多" in q: conf += 1
    if "新空" in q: conf -= 1
    notes.append(f"OI四象限(24h)：{q}")
    tf24 = der["taker_flow"]["last_24h"]["buy_ratio"]
    conf += 0.5 if tf24 > 0.52 else (-0.5 if tf24 < 0.48 else 0)
    fr = der["funding"]["pct_rank_30d"]
    if fr > 90: notes.append("资金费率处于30天高位，多头拥挤"); conf -= 0.5
    if fr < 10: notes.append("资金费率处于30天低位，空头拥挤"); conf += 0.5
    la = der["long_short"]["all_accounts"]["pct_rank_100h"]; tp = der["long_short"]["top_trader_positions"]["now"]
    if la > 85 and tp < 1: notes.append("散户多空人数比偏高而大户持仓偏空，逆向看空信号"); conf -= 0.5
    if la < 15 and tp > 1: notes.append("散户偏空而大户持仓偏多，逆向看多信号"); conf += 0.5
    imb = ob["depth"]["±0.5%"]["imbalance_avg"]
    conf += 0.5 if imb > 0.15 else (-0.5 if imb < -0.15 else 0)
    # 草拟计划（交给大模型复核）
    # 草拟计划：止损放在结构外侧（若紧挨着还有同向位置，就放到那一层外面），目标至少离入场 1 ATR(4H)
    plans = []
    buf = 0.3 * a4
    def rr(entry, stop, tgt): return round(abs(tgt - entry) / abs(entry - stop), 2) if entry != stop else None
    def plan(kind, entry, stop, tg, trigger):
        tg = tg[:3]
        if tg:
            plans.append({"type": kind, "entry": entry, "stop": stop, "targets": tg,
                          "risk_pct": round(abs(entry - stop) / entry * 100, 2),
                          "rr_first": rr(entry, stop, tg[0]), "trigger": trigger})
    if ns:
        stop = ns["range"][0] - buf
        if len(sup) > 1 and sup[1]["range"][1] > stop - 0.5 * a4:
            stop = sup[1]["range"][0] - buf
        plan("做多·见位（回踩支撑确认后）", ns["range"][1], stop,
             [x["price"] for x in res if x["price"] >= ns["range"][1] + a4],
             "回踩支撑区后，1H 出现看涨 K 线线态且 KD 低位金叉，或收回支撑之上")
        plan("做空·破位（跌破支撑后回抽）", ns["range"][0], ns["range"][1] + buf,
             [x["price"] for x in sup[1:] if x["price"] <= ns["range"][0] - a4],
             "1H 收盘跌破支撑区且下一根确认，主动卖量与 OI 同步放大")
    if nr:
        stop = nr["range"][1] + buf
        if len(res) > 1 and res[1]["range"][0] < stop + 0.5 * a4:
            stop = res[1]["range"][1] + buf
        plan("做空·见位（反弹阻力受压）", nr["range"][0], stop,
             [x["price"] for x in sup if x["price"] <= nr["range"][0] - a4],
             "反弹至阻力区后，1H 出现看跌 K 线线态且 KD 高位死叉")
        plan("做多·破位（突破阻力后回踩）", nr["range"][1], nr["range"][0] - buf,
             [x["price"] for x in res[1:] if x["price"] >= nr["range"][1] + a4],
             "1H 收盘站上阻力区且下一根确认，主动买量与 OI 同步放大")
    return {"weighted_trend_bias": round(bias, 2), "resonance": resonance,
            "regime_4h": tfr["4H"]["regime"], "regime_1d": tfr["1D"]["regime"], "position": position,
            "derivatives_confirm_score": conf, "derivatives_notes": notes, "draft_plans": plans}


# ---------------- 大模型报告 ----------------
def round_obj(o, nd=6):
    if isinstance(o, float):
        if not np.isfinite(o): return None
        return float(f"{o:.{nd}g}")
    if isinstance(o, dict): return {k: round_obj(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_obj(v, nd) for v in o]
    if isinstance(o, (np.floating,)): return round_obj(float(o), nd)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


def llm_report(facts):
    from openai import OpenAI
    c = CFG["llm"]
    client = OpenAI(base_url=c["base_url"], api_key=c["api_key"], timeout=600)
    method = (ROOT / "方法论.md").read_text(encoding="utf-8")
    system = (
        "你是一名严格遵守下述《交易分析方法论》的合约技术分析师。你的任务是基于用户提供的实时数据（JSON），"
        "按方法论的执行清单与五步流程写出中文分析报告。\n"
        "硬性要求：\n"
        "1. 只能使用 JSON 中给出的数值，不得编造价格、指标或新闻；数据不足时明确说“数据不足”。\n"
        "2. 顺序：数据事件 → 多周期趋势（1D→4H→1H→15m，说明是否共振）→ 行情性质与占优策略 → 位置分析（列出上下关键位及其来源、重合度，判断见位/破位/顶位/间位）"
        "→ 形态与速率 → 指标确认（背离、超买超卖；指标只作确认）→ 合约数据确认（OI 四象限、CVD、资金费率、多空比、基差、爆仓、订单簿挂单墙及其持续性）→ 结论。\n"
        "3. 结论必须是条件式情景（若……则……），至少给出多、空、观望三种情景，每种写清触发条件、入场区、止损（放在结构外侧）、目标位、盈亏比；"
        "盈亏比低于 2 的情景要标注“盈亏比不足，按方法论放弃”。可以参考 JSON 中的 draft_plans，但要按方法论复核并修正。\n"
        "4. 遵守方法论的风控原则：不猜顶底、不逆大势、间位不追、收缩/发散行情空仓、试探仓+加仓+跟进止损、不摊平、不锁仓。\n"
        "5. 用 Markdown，关键价位用表格。不要写报告大标题和时间抬头（程序会加），直接从“## 0. 数据事件”开始。"
        "结尾给出“一句话结论”，并注明本报告是基于公开行情数据的技术分析，不构成投资建议。\n"
        "6. 报告正文之后，另起一行输出一个 ```json 代码块，内容为你最终给出的情景列表，程序会用它精确复算盈亏比：\n"
        '[{"name":"情景名称","side":"long或short或none","entry_low":数值,"entry_high":数值,"stop":数值,"targets":[数值,...]}]\n'
        "观望情景 side 填 none，价格字段填 null。\n"
        "7. 控制篇幅，正文约 3000–5000 字，思考不要过长。\n\n"
        "=====《交易分析方法论》=====\n" + method)
    user = "以下是实时数据（JSON，价格单位 USDT，时间为北京时间）：\n```json\n" + json.dumps(facts, ensure_ascii=False) + "\n```"
    r = client.chat.completions.create(model=c["model"], temperature=c["temperature"], max_tokens=c["max_tokens"],
                                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    text = r.choices[0].message.content
    if not text:
        raise RuntimeError(f"大模型没有返回正文（finish_reason={r.choices[0].finish_reason}），可调大 config.json 里的 max_tokens")
    if r.choices[0].finish_reason == "length":
        text += "\n\n> ⚠️ 大模型输出达到 max_tokens 上限被截断，可调大 config.json 里的 max_tokens。"
    return text, r.usage


def verify_scenarios(text):
    """从报告末尾的 json 代码块取出情景，程序精确复算盈亏比（入场取区间中点）。"""
    m = re.search(r"```json\s*(\[.*?\])\s*```", text, re.S)
    if not m:
        return text, "\n\n> 未找到情景 JSON，跳过程序复核。"
    body = text[:m.start()].rstrip() + text[m.end():]
    try:
        sc = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        return body, f"\n\n> 情景 JSON 解析失败（{e}），跳过程序复核。"
    rows = ["\n\n## 程序复核：情景盈亏比\n",
            "> 入场按区间中点计算；盈亏比 = |目标 − 入场| / |入场 − 止损|。低于 2 的按方法论放弃。\n",
            "| 情景 | 方向 | 入场(中点) | 止损 | 风险% | 目标 → 盈亏比 | 判定 |", "|---|---|---|---|---|---|---|"]
    for s in sc:
        if s.get("side") not in ("long", "short") or s.get("stop") is None:
            rows.append(f"| {s.get('name')} | 观望 | — | — | — | — | — |")
            continue
        lo, hi = s.get("entry_low"), s.get("entry_high")
        e = (lo + hi) / 2 if lo is not None and hi is not None else (lo or hi)
        st = s["stop"]
        wrong = (s["side"] == "long" and st >= e) or (s["side"] == "short" and st <= e)
        risk = abs(e - st)
        rr = [(t, abs(t - e) / risk) for t in s.get("targets", []) if risk]
        best = max((x for _, x in rr), default=0)
        verdict = "止损方向错误" if wrong else ("合格" if best >= 2 else "盈亏比不足，放弃")
        rows.append(f"| {s['name']} | {'多' if s['side'] == 'long' else '空'} | {e:,.6g} | {st:,.6g} | {risk / e * 100:.2f}% | "
                    + "；".join(f"{t:,.6g} → {x:.2f}" for t, x in rr) + f" | {verdict} |")
    return body, "\n".join(rows)


# ---------------- 数据附录 ----------------
def appendix(facts):
    f = lambda v: f"{v:,.6g}" if isinstance(v, (int, float)) else str(v)
    lines = ["\n\n---\n\n## 附录：程序计算的原始数据\n",
             "### 多周期趋势投票\n", "| 周期 | 趋势分(−3~+3) | 结构 | 均线组 | 顾比 | 维加斯 | 一目(多/空条件) | MACD | 行情性质 | ADX |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for tf, r in facts["timeframes"].items():
        lines.append(f"| {tf} | {r['trend_score']} | {r['structure']['state']} | {r['ema_group']['state']}/{r['ema_group']['rhythm']} | "
                     f"{r['gmma']['state']} | {r['vegas']['state']} | {r['ichimoku']['price_vs_cloud']} {r['ichimoku']['bull_conditions']}/{r['ichimoku']['bear_conditions']} | "
                     f"{r['macd']['zero_axis']} {r['macd']['cross']} | {r['regime']} | {r['adx']['adx']:.1f} |")
    lines += ["\n### 关键位置（聚类后）\n", "| 类型 | 价格 | 距离 | 强度 | 来源 |", "|---|---|---|---|---|"]
    for name, arr in (("阻力", facts["levels"]["resistance"][::-1]), ("支撑", facts["levels"]["support"])):
        for x in arr:
            lines.append(f"| {name} | {f(x['price'])} | {x['dist_pct']}% / {x['dist_atr4h']} ATR | {x['strength']} | {'、'.join(x['sources'][:6])} |")
    ob = facts["orderbook"]
    lines += ["\n### 订单簿深度（最后一次采样）\n", "| 范围 | 买盘(币) | 卖盘(币) | 不平衡(均值) |", "|---|---|---|---|"]
    for k, v in ob["depth"].items():
        lines.append(f"| {k} | {f(v['bid_coin'])} | {f(v['ask_coin'])} | {v['imbalance_avg']} |")
    lines += ["\n持续挂单墙：\n", "| 方向 | 价格 | 距离 | 数量(币) | 金额(USD) | 持续 |", "|---|---|---|---|---|---|"]
    for w_ in ob["persistent_walls"]:
        lines.append(f"| {w_['side']} | {f(w_['price'])} | {w_['dist_pct']}% | {f(w_['size_coin'])} | {w_['size_usd']:,.0f} | {w_['persist']} |")
    d = facts["derivatives"]
    lines += ["\n### 合约数据\n",
              f"- 资金费率：{d['funding']['current_pct']:.4f}%（30 天分位 {d['funding']['pct_rank_30d']:.0f}%，30 天均值 {d['funding']['avg_30d_pct']:.4f}%）",
              f"- 持仓量：{d['open_interest']['oi_coin']:,.0f} 币 / {d['open_interest']['oi_usd'] / 1e8:.2f} 亿 USD；4h {d['open_interest']['4h']}；24h {d['open_interest']['24h']}",
              f"- 主动买入占比：1h {d['taker_flow']['last_1h']['buy_ratio']}，4h {d['taker_flow']['last_4h']['buy_ratio']}，24h {d['taker_flow']['last_24h']['buy_ratio']}；{('；'.join(d['taker_flow']['cvd_divergence']) or '无CVD背离')}",
              f"- 多空比：全部账户 {d['long_short']['all_accounts']['now']}，大户人数 {d['long_short']['top_trader_accounts']['now']}，大户持仓 {d['long_short']['top_trader_positions']['now']}",
              f"- 基差：{d['basis']['premium_pct']}%；爆仓：{d['liquidations']}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inst", nargs="?", default=CFG["default_inst"])
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    inst = args.inst.upper()
    now = datetime.now(TZ)
    log(f"[{now:%H:%M:%S}] 分析 {inst}")

    ins = ox.instrument(inst)
    ct_val = float(ins["ctVal"])
    tk = ox.ticker(inst)
    price = float(tk["last"])

    dfs, tfr, pivs = {}, {}, {}
    for tf, n, _ in TFS:
        dfs[tf] = ox.candles(inst, tf, n)
        tfr[tf], pivs[tf], _ = ta.analyze_tf(dfs[tf], tf, zz_mult=2.0 if tf != "1D" else 1.5)
        log(f"  {tf}: {len(dfs[tf])} 根K线, 趋势分 {tfr[tf]['trend_score']}, {tfr[tf]['structure']['state']}, {tfr[tf]['regime']}")

    log("  订单簿多次采样中…")
    ob = orderbook_analysis(inst, ct_val, price, CFG["orderbook_snapshots"], CFG["orderbook_interval_sec"])
    log("  合约数据…")
    der = derivatives_analysis(inst, ct_val, dfs["1H"])
    vp = volume_profile(dfs["1H"], price)
    sup, res = collect_levels(price, tfr, pivs, dfs, vp, ob)
    bo = breakout_check(dfs["1H"], sup[:3] + res[:3])
    syn = synthesize(price, tfr, sup, res, der, ob)

    tk2 = ox.ticker(inst)
    facts = round_obj({
        "instrument": inst, "contract_value": f"1张={ct_val}{ins['ctValCcy']}",
        "analysis_time_bj": now.strftime("%Y-%m-%d %H:%M"),
        "ticker": {"last": float(tk2["last"]), "high24h": float(tk2["high24h"]), "low24h": float(tk2["low24h"]),
                   "open24h": float(tk2["open24h"]), "chg24h_pct": (float(tk2["last"]) / float(tk2["open24h"]) - 1) * 100,
                   "vol24h_coin": float(tk2["volCcy24h"])},
        "note": "所有K线指标只用已收盘K线计算（收盘原则）；订单簿仅覆盖盘口附近 coverage_pct 范围",
        "macro_events": CFG.get("events") or "未提供（可在 config.json 的 events 里手动填写近期 FOMC/CPI/非农 等事件）",
        "timeframes": tfr,
        "volume_profile_1h_21d": vp,
        "levels": {"support": sup, "resistance": res},
        "breakout_checks_1h": bo,
        "orderbook": ob,
        "derivatives": der,
        "synthesis": syn,
    })

    out_dir = ROOT / "reports"; out_dir.mkdir(exist_ok=True)
    stem = f"{inst}_{now:%Y%m%d_%H%M}"
    (out_dir / f"{stem}.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  综合：趋势偏向 {syn['weighted_trend_bias']}，{syn['resonance']}，4H {syn['regime_4h']}，{syn['position']}")

    header = f"# {inst} 合约分析报告\n\n> 生成时间：{now:%Y-%m-%d %H:%M}（北京时间）｜最新价 {facts['ticker']['last']}｜数据源：OKX 公共行情｜分析框架：《方法论.md》\n\n"
    if args.no_llm:
        body = "（未调用大模型，仅输出程序计算结果）"
    else:
        log(f"  调用 {CFG['llm']['model']} 生成报告…")
        t0 = time.time()
        body, usage = llm_report(facts)
        log(f"  完成，用时 {time.time() - t0:.0f}s，tokens={usage.total_tokens if usage else '?'}")
        body, check = verify_scenarios(body)
        body += check
    md = header + body + appendix(facts)
    path = out_dir / f"{stem}.md"
    path.write_text(md, encoding="utf-8")
    log(f"报告：{path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
