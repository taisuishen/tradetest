"""
试探小仓回测：主策略因“1H 趋势效率不足（震荡）”放弃的三买 / 三卖信号，改开 0.1 倍小仓，
按手续费倍数设止盈 / 止损（1 分钟 K 线判定，同一根同时触及按先止损）。与主策略各占一个仓位、互不影响。

一倍手续费 = 开平一次的交易手续费（成交金额 × 吃单费率 × 2），约等于价格波动 0.1%。
方案 A：止盈 净 +1 倍（价格 +2 倍）；止损 价格 −1 倍（净 −2 倍）
方案 B：止盈 净 +1 倍（价格 +2 倍）；止损 净 −1 倍（止损放在开仓价）
方案 C：对称，价格 ±1 倍（止盈净 0，止损净 −2 倍）

用法：python bt_scout.py [天数=120]
"""
import sys
import time
from datetime import datetime

import pandas as pd

import chan15_lab as lab
import fees
import kcache
from bt_sizing import LIVE

M15, H = 900_000, 3_600_000
PLANS = {"A 止盈净+1倍 / 止损价格-1倍": (2, 1), "B 止盈净+1倍 / 止损净-1倍（止损=开仓价）": (2, 0), "C 对称 价格±1倍": (1, 1)}


def rejected_signals(Ts, p15, p1h, v):
    """主策略里通过了其他所有过滤、只因 1H 趋势效率不足被放弃的信号：[(时刻, 买卖点, 方向, 入场价)]"""
    out, used = [], set()
    for T in Ts:
        info = p15.get(T)
        if not info:
            continue
        hr = p1h.get(T // H * H)
        for t, ts_, px in reversed(info["fresh"]):
            if t not in v["types"] or (t, ts_) in used:
                continue
            used.add((t, ts_))
            side = 1 if t in lab.LONG else -1
            if not hr or (v["trend_filter"] and hr["trend"] * side <= -1.0) or (v["nest"] and hr["bias"] * side < 0):
                continue
            stop = px - side * v["stop_buf"] * info["atr"]
            if (stop - info["close"]) * side >= 0:
                continue
            if hr["eff1h"] < v["eff1h_min"]:                      # 只取“因为震荡被放弃”的
                out.append((T, t, side, info["close"], hr["eff1h"]))
            break
    return out


def run_scouts(sigs, m1, unit, taker, funding, tp_k, sl_k, mult=0.1):
    qty = unit * mult
    trades, busy_until, j = [], 0, 0
    for T, t, side, entry, eff in sigs:
        if T < busy_until:           # 上一笔小仓还没结束，不叠加
            continue
        fee1 = 2 * entry * qty * taker                          # 一倍手续费（开平一次）
        d = fee1 / qty                                           # 对应的价格距离
        tp, sl = entry + side * tp_k * d, entry - side * sl_k * d
        while j < len(m1) and m1[j][0] < T:
            j += 1
        k, res = j, None
        while k < len(m1):
            ts_, h, l = m1[k]
            if (l <= sl) if side > 0 else (h >= sl):
                res = (sl, ts_ + 60_000, "止损"); break
            if (h >= tp) if side > 0 else (l <= tp):
                res = (tp, ts_ + 60_000, "止盈"); break
            k += 1
        if not res:
            break
        px, te, why = res
        gross = (px - entry) * side * qty
        fee = fees.trade_fee(entry, px, qty, taker)
        fund = fees.funding_cost(side, qty, entry, T, te, funding)
        trades.append({"ts": T, "type": t, "entry": entry, "exit": px, "result": why, "minutes": (te - T) / 60000,
                       "gross": gross, "fee": fee, "funding": fund, "net": gross - fee - fund, "eff": eff})
        busy_until = te
    return pd.DataFrame(trades)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    since = int(time.time() * 1000) - (days + 1) * 86_400_000
    rows = []
    for inst, unit in (("ETH-USDT-SWAP", 1.0), ("BTC-USDT-SWAP", 0.03)):
        Ts, p15, p1h, _ = lab.precompute(inst, days)
        sigs = rejected_signals(Ts, p15, p1h, LIVE)
        t0 = time.time()
        k1 = kcache.candles(inst, "1m", days * 1440 + 60)      # 并行拉取 + 本地缓存
        m1 = list(zip(k1.ts.astype("int64"), k1.h, k1.l))
        taker = fees.rates(inst)["taker"]
        funding = fees.funding_history(inst, since)
        print(f"{inst}：因震荡被放弃的信号 {len(sigs)} 个；1 分钟 K 线 {len(m1)} 根（{time.time() - t0:.0f}s）", flush=True)
        for name, (tp_k, sl_k) in PLANS.items():
            df = run_scouts(sigs, m1, unit, taker, funding, tp_k, sl_k)
            if not len(df):
                rows.append({"合约": inst[:3], "方案": name, "笔数": 0}); continue
            eq = df.net.cumsum(); mdd = float((eq.cummax().clip(lower=0) - eq).max())
            rows.append({"合约": inst[:3], "方案": name, "笔数": len(df), "胜率%": round((df.result == "止盈").mean() * 100, 1),
                         "毛利U": round(df.gross.sum(), 2), "手续费U": round(df.fee.sum(), 2), "资金费U": round(df.funding.sum(), 3),
                         "净利U": round(df.net.sum(), 2), "最大回撤U": round(mdd, 2), "平均持仓分钟": round(df.minutes.mean(), 1),
                         "盈亏平衡胜率%": round((sl_k + 1) / (tp_k + sl_k) * 100, 1)})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print("\n两个合约合计：\n" + res.groupby("方案", sort=False)[["笔数", "净利U", "手续费U"]].sum().round(2).to_string())
