"""
策略 v5 / v6 两段回测：v4 / + 不追高 / + 低效率三买（TradeTrack 确认）/ 两者都加（= v5）/ v5 + 维加斯隧道离场（= v6 实盘）。
每笔名义 = 开仓时已实现权益 × 1（逐仓 10 倍、保证金 10%），复利；吃单 0.05%，不计滑点。
两段：最近 180 天（8 个品种）、2025 全年（6 个品种，ZEC / HYPE 当年上线太晚）。

低效率三买：1H 趋势效率在 10–20%（v4 要求 ≥20% 所以原本被挡掉），且 TradeTrack 1H 评分 ≥40、4H 评分 ≥20 才做
不追高：现价离三买点不超过 1.75 倍 ATR(15m)
v6 离场：反向信号，或 15m 收盘跌破维加斯隧道下沿 min(EMA144, EMA169)，先到先走（来源见 bt_top5.py / docs/投机实验室_热门5.md）

用法：python bt_v5.py      首次运行要算 TradeTrack 特征（约 3 分钟，缓存在 bt/）
"""
import gzip
import pickle
import sys

import pandas as pd

import bt_ttrack as B
import chan15_lab2 as L

L.SIGS["weak"] = L.sig_trend        # 同一套缠论规则，换个模块名，方便单独统计低效率信号的单子
END25 = int(pd.Timestamp("2026-01-01", tz="Asia/Shanghai").timestamp() * 1000)
PERIODS = {"最近 180 天": (180, None, set()), "2025 全年": (365, END25, {"ZEC-USDT-SWAP", "HYPE-USDT-SWAP"})}
STRONG = {"eff_min": 0.2, "zs_w_max": 1.5, "types": {"三买"}}
LOW = {**STRONG, "eff_min": 0.10, "eff_max": 0.20}
CH = {"max_chase_atr": 1.75}


def tt_gate(s1, s4):
    def g(T, side, info, hr):
        f = info.get("tt")
        return bool(f) and f["1H"]["score"] * side >= s1 and f["4H"]["score"] * side >= s4
    return g


V = {"v4": [("trend", STRONG)],
     "v4 + 不追高 1.75 ATR": [("trend", {**STRONG, **CH})],
     "v4 + 低效率三买（TT 确认）": [("trend", STRONG), ("weak", {**LOW, "gate": tt_gate(40, 20)})],
     "v5（两者都加）": [("trend", {**STRONG, **CH}), ("weak", {**LOW, **CH, "gate": tt_gate(40, 20)})]}
V["v6（v5 + 跌破维加斯隧道下沿离场，当前实盘）"] = [(m, {**pv, "tunnel_exit": True}) for m, pv in V["v5（两者都加）"]]
LIVE = V["v6（v5 + 跌破维加斯隧道下沿离场，当前实盘）"]


def vegas_tunnel(inst):
    """15m 维加斯隧道 {收盘时刻 T: (下沿, 上沿)}，EMA144 / EMA169 用全部缓存历史计算（实盘取 1000 根，初值影响可忽略）。"""
    h = pickle.load(gzip.open(L.OUT.parent / "cache" / f"{inst}_15m.pkl.gz", "rb"))
    ts = sorted(h)
    c = pd.Series([h[t][4] for t in ts], dtype=float)
    a, b = c.ewm(span=144, adjust=False).mean().values, c.ewm(span=169, adjust=False).mean().values
    return {int(t) + L.M15: (float(min(x, y)), float(max(x, y))) for t, x, y in zip(ts, a, b)}


def attach_vt(data):
    """把 15m 维加斯隧道挂到每根 15m 的 info["vt"] 上（v6 离场用），返回新的 data。"""
    out = {}
    for inst, (unit, Ts, p15, p1h, b5, fm) in data.items():
        vt = vegas_tunnel(inst)
        out[inst] = (unit, Ts, {T: {**x, "vt": vt.get(T)} for T, x in p15.items() if x is not None}, p1h, b5, fm)
    return out


def stat(data, sigs, days):
    rows = []
    for d in data.values():
        t = L.simulate(d, {"sigs": sigs})
        if len(t):
            t["ret"] = t.net / (t.entry * d[0] * t.mult); rows.append(t)
    df = pd.concat(rows).sort_values("ts").reset_index(drop=True)
    cv, mx = L.compound(df); mdd = float((1 - cv["eq"] / cv["eq"].cummax()).max())
    m = cv.groupby("月")["eq"].last(); mr = m / m.shift(1, fill_value=1.0) - 1
    return {"复利": f"{(cv['eq'].iloc[-1] - 1) * 100:+.0f}%", "回撤": f"{mdd:.0%}", "笔数": len(df), "每周": f"{len(df) / days * 7:.1f}",
            "胜率": f"{(df.net > 0).mean():.0%}", "去前5（不复利）": f"{(df.ret.sum() - df.ret.nlargest(5).sum()) * 100:+.0f}%",
            "亏损月": f"{(mr < 0).sum()}/{len(mr)}", "最差月": f"{mr.min():+.0%}", "最多同时": mx}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 260); pd.set_option("display.unicode.east_asian_width", True)
    insts = L.live_insts()
    for pname, (days, end, skip) in PERIODS.items():
        data = {i: d for i, d in L.load(days, tuple(x for x in insts if x[0] not in skip), "_live", end).items() if i in dict(insts)}
        data = attach_vt(B.attach(data, B.features(data, days, end)))
        print(f"\n== {pname} ==\n" + pd.DataFrame({k: stat(data, s, days) for k, s in V.items()}).T.to_string())
