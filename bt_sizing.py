"""动态仓位（0.5~5 倍）测试：与固定 1 倍、以及“固定为同样平均倍数”对比。用法：python bt_sizing.py [天数=120]"""
import sys
import pandas as pd
import chan15_lab as lab

LIVE = {**lab.BASE, **lab.MAIN, "eff1h_min": 0.25}    # 当前实盘策略


def summarize(name, dfs, mid):
    df = pd.concat(dfs).sort_values("ts")
    eq = df.net.cumsum(); mdd = float((eq.cummax().clip(lower=0) - eq).max())
    return {"方案": name, "笔数": len(df), "胜率%": round((df.net > 0).mean() * 100, 1),
            "平均倍数": round(df.mult.mean(), 2), "前半": round(df[df.ts < mid].net.sum(), 1), "后半": round(df[df.ts >= mid].net.sum(), 1),
            "全程净利": round(df.net.sum(), 1), "手续费": round(df.fee.sum(), 1), "资金费": round(df.funding.sum(), 1), "最大回撤": round(mdd, 1),
            "收益回撤比": round(df.net.sum() / mdd, 2) if mdd else float("inf"),
            "每1倍净利": round(df.net.sum() / df.mult.mean(), 1)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    import time, fees
    fm_mode = sys.argv[2] if len(sys.argv) > 2 else "okx"      # okx=按 OKX 费率 + 实际资金费；fixed=每 1 倍固定 2U
    data, fms = {}, {}
    for inst, unit in (("ETH-USDT-SWAP", 1.0), ("BTC-USDT-SWAP", 0.03)):
        data[inst] = (unit, *lab.precompute(inst, days))
        fms[inst] = ({"taker": fees.rates(inst)["taker"], "funding": fees.funding_history(inst, int(time.time() * 1000) - (days + 1) * 86_400_000)}
                     if fm_mode == "okx" else None)
    print(f"费用模型：{'OKX 吃单费率 ' + str(fees.rates('ETH-USDT-SWAP')) + ' + 实际资金费' if fm_mode == 'okx' else '每 1 倍固定 2U'}")
    mid = data["ETH-USDT-SWAP"][1][len(data["ETH-USDT-SWAP"][1]) // 2]
    def run(v):
        out = []
        for inst, (unit, Ts, p15, p1h, b5) in data.items():
            d = lab.simulate(Ts, p15, p1h, b5, unit, v, fm=fms[inst]); d["inst"] = inst[:3]; out.append(d)
        return out
    base = run(LIVE); step = run({**LIVE, "sizing": "step"}); lin = run({**LIVE, "sizing": "linear"})
    m_step = pd.concat(step).mult.mean(); m_lin = pd.concat(lin).mult.mean()
    rows = [summarize("固定 1 倍（当前）", base, mid),
            summarize("动态分档 0.5/1/2/3/5 倍", step, mid),
            summarize(f"对照：固定 {m_step:.2f} 倍（=分档平均）", run({**LIVE, "fixed_mult": m_step}), mid),
            summarize("动态线性 0.5~5 倍", lin, mid),
            summarize(f"对照：固定 {m_lin:.2f} 倍（=线性平均）", run({**LIVE, "fixed_mult": m_lin}), mid)]
    print(pd.DataFrame(rows).to_string(index=False))
    b = pd.concat(base)
    b["档位"] = pd.cut(b.score, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01], labels=["<0.2(0.5倍)", "0.2-0.4(1倍)", "0.4-0.6(2倍)", "0.6-0.8(3倍)", "≥0.8(5倍)"])
    print("\n评分能不能预测单笔好坏（固定 1 倍下按评分分组）：")
    print(b.groupby("档位", observed=False).net.agg(笔数="count", 胜率=lambda x: round((x > 0).mean() * 100, 1) if len(x) else 0,
                                                    每笔平均净利="mean", 合计="sum").round(2).to_string())
    print(f"\n评分与单笔净利的相关系数：{b.score.corr(b.net):.3f}")
    s = pd.concat(step)
    print("\n动态分档逐笔：\n" + s.sort_values("ts")[["inst", "type", "side", "start", "score", "mult", "result", "net_per_1x", "net"]].to_string(index=False))
