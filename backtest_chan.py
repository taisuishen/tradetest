"""
回测纯缠论单仓策略（chan_strategy.py），1H 逐根推进，止盈止损用 5 分钟 K 线判定（同根先判止损）。
用法：python backtest_chan.py [天数=60] [合约]
"""
import json
import sys
from datetime import datetime

import bt_parallel as bp
import okx_client as ox
import paper_trader as pt
import chan_strategy as cs
from backtest import summary

import pandas as pd

H = 3_600_000


def run(inst, unit, days, cfg, workers=None):
    df1h = ox.candles(inst, "1H", 400 + days * 24 + 2)
    df4h = ox.candles(inst, "4H", 300 + days * 6 + 2)
    df1d = ox.candles(inst, "1D", 300 + days + 2)
    df5 = ox.candles(inst, "5m", days * 288 + 24)
    b5 = list(zip(df5.ts.astype("int64"), df5.h, df5.l))
    start = int(df1h.ts.iloc[-1]) + H - days * 24 * H
    Ts = list(range(start, int(df1h.ts.iloc[-1]) + H + 1, H))
    pre = bp.precompute(bp.chan_task, Ts, {"df1h": df1h, "df4h": df4h, "df1d": df1d, "cfg": cfg, "unit": unit}, workers)
    j, pos, used, trades, cool = 0, None, set(), [], 0
    fee = cfg["fee_per_unit"]

    def book(px, why, ts_):
        gross = (px - pos["entry"]) * unit * (1 if pos["side"] == "long" else -1)
        trades.append({"inst": inst.split("-")[0], "side": pos["side"], "point": pos["point"],
                       "start": datetime.fromtimestamp(pos["ts"] / 1000).strftime("%m-%d %H:%M"),
                       "end": datetime.fromtimestamp(ts_ / 1000).strftime("%m-%d %H:%M"),
                       "layers": 1, "entry": round(pos["entry"], 2), "exit": round(px, 2), "result": why,
                       "rr_plan": round(pos["rr"], 2), "fee": fee, "net": round(gross - fee, 2)})
        return gross - fee

    for T in Ts:
        info = pre[T]
        if info is None:
            continue
        if pos:  # 持仓中出现反向买卖点 → 按 1H 收盘价离场
            want = cs.SHORT_PTS if pos["side"] == "long" else cs.LONG_PTS
            opp = next((t for t, ts_ in info["pts"] if t in want and ts_ >= pos["ts"]), None)
            if opp:
                net = book(info["close"], f"反向{opp}离场", T)
                cool = T + (cfg["cooldown_after_loss_min"] if net <= 0 else cfg["cooldown_after_win_min"]) * 60_000
                pos = None
        if pos is None and T >= cool:
            res = info["decide"]
            if len(res) == 3 and res[2] in used:      # 同一个买卖点只做一次
                res = (None, "已交易过")
            if len(res) == 3:
                used.add(res[2])
            if res[0]:
                pos = {**res[0], "ts": T}
        if pos is None:
            continue
        while j < len(b5) and b5[j][0] < T:
            j += 1
        k = j
        while k < len(b5) and b5[k][0] < T + H:
            long = pos["side"] == "long"
            ts_, h, l = b5[k]
            if (l <= pos["stop"]) if long else (h >= pos["stop"]):
                book(pos["stop"], "止损", ts_ + 300_000); pos = None
                cool = ts_ + 300_000 + cfg["cooldown_after_loss_min"] * 60_000
                break
            if (h >= pos["target"]) if long else (l <= pos["target"]):
                book(pos["target"], "止盈", ts_ + 300_000); pos = None
                cool = ts_ + 300_000 + cfg["cooldown_after_win_min"] * 60_000
                break
            k += 1
    return pd.DataFrame(trades), pos


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cfg = pt.load_cfg()
    insts = dict(cfg["instruments"])
    if len(sys.argv) > 2 and sys.argv[2] != "all":
        insts = {sys.argv[2]: insts.get(sys.argv[2], {"unit": 1.0})}
    if len(sys.argv) > 3:
        cfg.update(json.loads(sys.argv[3])); print("参数覆盖：", sys.argv[3])
    for inst, ic in insts.items():
        df, open_pos = run(inst, ic["unit"], days, cfg)
        if len(df):
            summary(df, f"纯缠论 {inst}（每笔 1 倍={ic['unit']}）")
            print(df.groupby("point").net.agg(笔数="count", 胜率=lambda x: round((x > 0).mean() * 100, 1), 净利="sum").to_string())
            print(df.to_string(index=False))
        else:
            print(f"纯缠论 {inst}：无成交")
        if open_pos:
            print(f"   回测结束时仍持仓：{open_pos['side']} {open_pos['point']} @{open_pos['entry']:.6g}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
