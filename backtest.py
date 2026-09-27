"""
回测小时级马丁分批挂单（与 paper_trader.py 同一套 plan / replay 规则）。

- 每根 1H 收盘时：没有持仓就按当时结构挂单；挂单一层都没成交就按最新结构重挂（或撤单）
- 挂单成交、止损、止盈用 5 分钟 K 线高低点判定（同根先成交、再判止损，新成交那根不判止盈）
- 资金费率、订单簿挂单墙没有历史数据，按中性处理

用法：python backtest.py [天数=60] [合约，all=配置里全部] ['{"参数":值}']
"""
import json
import sys
from datetime import datetime

import pandas as pd

import bt_parallel as bp
import okx_client as ox
import paper_trader as pt

H = 3_600_000


def record(inst, camp, r):
    return {"inst": inst.split("-")[0], "side": camp["side"],
            "start": datetime.fromtimestamp(camp["created"] / 1000).strftime("%m-%d %H:%M"),
            "end": datetime.fromtimestamp(r["exit_ts"] / 1000).strftime("%m-%d %H:%M"),
            "layers": r["max_layer"], "qty": round(r["qty"], 4), "avg": round(r["avg_px"], 2),
            "exit": round(r["exit_px"], 2), "result": r["exit_reason"], "fee": r["fee"],
            "net": round(r["net"], 2), "max_loss_plan": camp["max_loss"]}


def run(inst, unit, days, cfg, workers=None):
    df1h = ox.candles(inst, "1H", 400 + days * 24 + 2)
    df4h = ox.candles(inst, "4H", 300 + days * 6 + 2)
    df1d = ox.candles(inst, "1D", 300 + days + 2)
    df5 = ox.candles(inst, "5m", days * 288 + 24)
    b5 = list(zip(df5.ts.astype("int64"), df5.h, df5.l))
    start = int(df1h.ts.iloc[-1]) + H - days * 24 * H
    Ts = list(range(start, int(df1h.ts.iloc[-1]) + H + 1, H))   # T = 某根 1H 的收盘时刻
    # 多进程并行算好每个时刻的挂单计划和缠论买卖点，下面只做按时间顺序的状态模拟
    pre = bp.precompute(bp.plan_task, Ts, {"df1h": df1h, "df4h": df4h, "df1d": df1d, "cfg": cfg, "unit": unit}, workers)
    j = 0
    rounds, camp, cool_until = [], None, 0
    for T in Ts:
        info = pre[T]
        if info is None:
            continue
        if (camp is None and T >= cool_until) or (camp is not None and camp["qty"] == 0):
            p, why, ctx = info["plan"]
            if p is None or (camp and p["side"] != camp["side"]):
                camp = None                                         # 撤单
            if p is not None:
                camp = {"side": p["side"], "layers": json.dumps(p["layers"]), "stop": p["stop"], "tp": None,
                        "atr": p["atr"], "obstacles": json.dumps(p["obstacles"]), "unit": unit,
                        "last_checked_ts": T, "qty": 0, "created": camp["created"] if camp else T,
                        "max_loss": ctx["max_loss_if_all_filled"]}
        if camp is None:
            continue
        if camp["qty"] > 0 and cfg.get("use_chan"):   # 缠论保护：持仓后出现反向三类买卖点，按 1H 收盘价提前离场
            L_ = json.loads(camp["layers"])
            first_fill = min(x["filled_ts"] for x in L_ if x["filled_ts"])
            want = "三卖" if camp["side"] == "long" else "三买"
            if any(t == want and ts_ >= first_fill for t, ts_ in info["pts"]):
                r = pt.close_at(camp, cfg, L_, info["close"], T, "缠论离场")
                rounds.append(record(inst, camp, r))
                cool_until = T + (cfg["cooldown_after_loss_min"] if r["net"] <= 0 else cfg["cooldown_after_win_min"]) * 60_000
                camp = None
                continue
        while j < len(b5) and b5[j][0] < T:
            j += 1
        k = j
        while k < len(b5) and b5[k][0] < T + H:
            k += 1
        r = pt.replay(camp, cfg, b5[j:k], bar_ms=300_000)
        camp.update(layers=json.dumps(r["layers"]), tp=r["tp"], qty=r["qty"], last_checked_ts=r["last_checked_ts"])
        if r["closed"]:
            rounds.append(record(inst, camp, r))
            cool_until = r["exit_ts"] + (cfg["cooldown_after_loss_min"] if r["net"] <= 0 else cfg["cooldown_after_win_min"]) * 60_000
            camp = None
    return pd.DataFrame(rounds), camp


def summary(df, label):
    w = int((df.net > 0).sum()); n = len(df)
    eq = df.net.cumsum(); mdd = float((eq.cummax().clip(lower=0) - eq).max())
    gw = df[df.net > 0].net.sum(); gl = -df[df.net <= 0].net.sum()
    print(f"{label}：{n} 轮，胜 {w} 负 {n - w}，胜率 {w / n * 100:.1f}%，净利 {df.net.sum():+.2f}U，"
          f"手续费 {df.fee.sum():.1f}U，均盈 {df[df.net > 0].net.mean() if w else 0:+.2f}，均亏 {df[df.net <= 0].net.mean() if w < n else 0:+.2f}，"
          f"盈亏因子 {gw / gl if gl else float('inf'):.2f}，最大回撤 {mdd:.2f}U，"
          f"层数分布 {dict(df.layers.value_counts().sort_index())}")


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cfg = pt.load_cfg()
    insts = dict(cfg["instruments"])
    if len(sys.argv) > 2 and sys.argv[2] != "all":
        insts = {sys.argv[2]: insts.get(sys.argv[2], {"unit": 1.0})}
    if len(sys.argv) > 3:
        cfg.update(json.loads(sys.argv[3])); print("参数覆盖：", sys.argv[3])
    frames = []
    for inst, ic in insts.items():
        df, open_c = run(inst, ic["unit"], days, cfg)
        if len(df):
            summary(df, f"{inst}（1倍={ic['unit']}）")
            frames.append(df)
        else:
            print(f"{inst}：无成交")
        if open_c and open_c["qty"]:
            print(f"   回测结束时仍持仓：{open_c['side']} 数量 {open_c['qty']:.4g}")
    if frames:
        df = pd.concat(frames)
        if len(frames) > 1:
            summary(df, "合计")
        print(df.to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
