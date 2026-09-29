"""
TradeTrack 多周期指标短线策略回测（激进短线）：每根 15m 收盘看一次 TradeTrack 的 8 项指标评分与支撑压力（ttrack.py）。

入场（多单；空单对称）：1H 评分 ≥ a 且 4H 评分 ≥ b（可选 15m 评分 ≥ c），且这一根刚满足（上一根还不满足），
      上方最近的 1H 压力位离现价至少 room 倍 ATR(1H)
止损：最近的 1H 支撑下方 0.2 ATR(1H)；没有支撑或太远（> 3 ATR）就用 1.5 ATR(1H)
止盈：r1 最近的 1H 压力位（提前 0.1 ATR）/ r2 第二个压力位 / 数字 = 几倍风险
离场：止损、止盈，或 1H 评分转向（多单 < flip）
仓位：每笔名义 = 开仓时权益 × 1（逐仓 10 倍、保证金 10%），复利；吃单 0.05%，止盈按挂单 0.02%，不计滑点

用法：python bt_ttrack.py [天数=30]      分别列出最近 7 天、30 天和 N 天
"""
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import chan15_lab2 as L
import ttrack

M15, H, D = L.M15, L.H, L.D
BAR = {"15m": M15, "1H": H, "4H": 4 * H}
_G = {}


def _init(arrs):
    _G.update(arrs)


def _feat(args):
    """某合约在 15m 收盘时刻 T 的 15m / 1H / 4H 指标评分与支撑压力（只用 T 之前已收盘的 K 线，最近 300 根）。"""
    inst, T = args
    out = {}
    for tf, dur in BAR.items():
        ts, o, h, l, c, v = _G[(inst, tf)]
        j = np.searchsorted(ts, T - dur, side="right")        # ts + dur <= T 的已收盘 K 线
        i = max(0, j - 300)
        if j - i < 60:
            return T, None
        price = _G[(inst, "15m")][4][np.searchsorted(_G[(inst, "15m")][0], T - M15, side="right") - 1]
        r = ttrack.analyze(o[i:j], h[i:j], l[i:j], c[i:j], v[i:j], price, "1H" if tf == "15m" else tf)
        if r is None:
            return T, None
        out[tf] = {"score": r["score"], "atr": r["atr"], "sup": r["sup"], "res": r["res"]}
    return T, out


def features(data, days):
    """给最近 days 天的每根 15m 收盘算 TradeTrack 特征，按合约缓存到 bt/。"""
    import kcache
    res = {}
    for inst, d in data.items():
        Ts = d[1][-days * 96:]
        f = L.OUT / f"tt_{inst}_{Ts[0]}_{Ts[-1]}.pkl"
        if f.exists():
            res[inst] = pickle.load(open(f, "rb")); continue
        t0 = time.time()
        arrs = {}
        for tf in BAR:
            k = kcache.candles(inst, tf, days * 96 * M15 // BAR[tf] + 320, progress=False, per_sec=9)
            arrs[(inst, tf)] = (k.ts.values.astype("int64"), k.o.values, k.h.values, k.l.values, k.c.values, k.volQuote.values)
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1), initializer=_init, initargs=(arrs,)) as ex:
            res[inst] = dict(ex.map(_feat, [(inst, T) for T in Ts], chunksize=32))
        pickle.dump(res[inst], open(f, "wb"))
        print(f"{inst} TradeTrack 特征 {len(Ts)} 根，{time.time() - t0:.0f}s", flush=True)
    return res


def sig_tt(info, hr, v, used):
    """TradeTrack 短线入场。"""
    f, fp = info.get("tt"), info.get("tt_prev")
    if not f or not fp:
        return None
    a, b, c = v.get("a", 40), v.get("b", 20), v.get("c")
    def ok(x, side):
        return (x["1H"]["score"] * side >= a and x["4H"]["score"] * side >= b
                and (c is None or x["15m"]["score"] * side >= c))
    for side in v.get("sides", (1, -1)):
        if not ok(f, side) or ok(fp, side):
            continue
        h1, px = f["1H"], info["close"]
        atr = h1["atr"]
        ahead = h1["res"] if side > 0 else h1["sup"]          # 前方的压力（多）/ 支撑（空）
        behind = h1["sup"] if side > 0 else h1["res"]
        if ahead and abs(ahead[0][0] - px) < v.get("room", 1.0) * atr:
            continue                                            # 离前方的压力 / 支撑太近，空间不够
        stop = behind[0][0] - side * 0.2 * atr if behind else None
        if stop is None or abs(px - stop) > 3 * atr or (stop - px) * side >= 0:
            stop = px - side * 1.5 * atr
        o = {"side": side, "tag": "TT多" if side > 0 else "TT空", "stop": stop}
        tp = v.get("tp", "r1")
        if tp in ("r1", "r2") and ahead:
            k = 0 if tp == "r1" or len(ahead) < 2 else 1
            o["tp"] = ahead[k][0] - side * 0.1 * atr
        elif isinstance(tp, (int, float)):
            o["tp"] = px + side * tp * abs(px - stop)
        return o
    return None


L.SIGS["tt"] = sig_tt


def attach(data, feats):
    """把特征挂到每根 15m 的 info 上（tt：当前，tt_prev：上一根），返回新的 data。"""
    out = {}
    for inst, (unit, Ts, p15, p1h, b5, fm) in data.items():
        f = feats.get(inst, {})
        q = {}
        for T in Ts:
            info = p15.get(T)
            if info is not None:
                q[T] = {**info, "tt": f.get(T), "tt_prev": f.get(T - M15)}
        out[inst] = (unit, Ts, q, p1h, b5, fm)
    return out


def run(data, v, t_from):
    rows = []
    for inst, d in data.items():
        t = L.simulate(d, v, t_from=t_from)
        if len(t):
            t["inst"] = inst.split("-")[0]
            t["ret"] = t.net / (t.entry * d[0] * t.mult)
            t["fee_ret"] = t.fee / (t.entry * d[0] * t.mult)
            rows.append(t)
    if not rows:
        return {"笔数": 0}
    df = pd.concat(rows).sort_values("ts").reset_index(drop=True)
    cv, mx = L.compound(df)
    mdd = float((1 - cv["eq"] / cv["eq"].cummax()).max())
    return {"笔数": len(df), "胜率": f"{(df.net > 0).mean():.0%}", "复利": f"{cv['eq'].iloc[-1] - 1:+.1%}", "最大回撤": f"{mdd:.1%}",
            "手续费合计": f"{df.fee_ret.sum():.1%}",
            "去掉最赚 5 笔": f"{df.ret.sum() - df.ret.nlargest(5).sum():+.1%}",
            "止盈/止损/转向": f"{(df.why == '止盈').sum()}/{(df.why == '止损').sum()}/{(df.why == '转向').sum()}"}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250); pd.set_option("display.unicode.east_asian_width", True)
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    insts = L.live_insts()
    data = {i: d for i, d in L.load(180, insts, "_live").items() if i in dict(insts)}
    data = attach(data, features(data, days))
    end = min(d[1][-1] for d in data.values())
    chan = ("trend", {"eff_min": 0.2, "zs_w_max": 1.5, "types": {"三买"}})
    tt = lambda **k: ("tt", {"opp": False, "tt_exit": 0, **k})
    V = [("对照：缠论 v4（只做多）", {"sigs": [chan]}),
         ("TT 多空 1H≥40 4H≥20 止盈 r1", {"sigs": [tt()]}),
         ("TT 只做多", {"sigs": [tt(sides=(1,))]}),
         ("TT 止盈 r2", {"sigs": [tt(tp="r2")]}),
         ("TT 止盈 2R", {"sigs": [tt(tp=2)]}),
         ("TT 更严 1H≥60 4H≥40", {"sigs": [tt(a=60, b=40)]}),
         ("TT 更松 1H≥20 4H≥0", {"sigs": [tt(a=20, b=0)]}),
         ("TT 加 15m≥40", {"sigs": [tt(c=40)]}),
         ("TT 不按评分转向离场", {"sigs": [tt(tt_exit=None)]}),
         ("缠论 v4 + TT 短线（缠论优先）", {"sigs": [chan, tt()]}),
         ("缠论 v4 + TT 只做多", {"sigs": [chan, tt(sides=(1,))]})]
    wins = sorted({7, 30, days})
    for w in wins:
        rows = [{"方案": name, **run(data, v, end - w * D)} for name, v in V]
        print(f"\n== 最近 {w} 天 ==\n" + pd.DataFrame(rows).to_string(index=False))
