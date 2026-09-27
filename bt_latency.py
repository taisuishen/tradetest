"""扫描延迟对收益的影响：15m 收盘立即入场 vs 晚 5 / 10 / 15 分钟入场（当前实盘策略：只做趋势 + 动态分档 + OKX 费率）。"""
import sys, time
import pandas as pd
import chan15_lab as lab, fees
from bt_sizing import LIVE, summarize

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    data = {}
    for inst, unit in (("ETH-USDT-SWAP", 1.0), ("BTC-USDT-SWAP", 0.03)):
        fm = {"taker": fees.rates(inst)["taker"], "funding": fees.funding_history(inst, int(time.time() * 1000) - (days + 1) * 86_400_000)}
        data[inst] = (unit, fm, *lab.precompute(inst, days))
    mid = data["ETH-USDT-SWAP"][2][len(data["ETH-USDT-SWAP"][2]) // 2]
    rows = []
    for d, name in ((0, "15m 收盘立即入场"), (1, "晚 5 分钟入场"), (2, "晚 10 分钟入场"), (3, "晚 15 分钟入场")):
        out = []
        for inst, (unit, fm, Ts, p15, p1h, b5) in data.items():
            x = lab.simulate(Ts, p15, p1h, b5, unit, {**LIVE, "sizing": "step", "entry_delay_5m": d}, fm=fm); x["inst"] = inst[:3]; out.append(x)
        rows.append(summarize(name, out, mid))
    print(pd.DataFrame(rows).to_string(index=False))
