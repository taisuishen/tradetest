"""计算“区间套 + 大方向过滤 + 只做三类买卖点”在最近 N 天的年化收益。用法：python bt_annual.py [天数=120]"""
import sys
import pandas as pd
import chan15_lab as lab

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    v = {**lab.BASE, "nest": True, "trend_filter": True, "types": {"三买", "三卖"}}
    frames, notional = [], {}
    for inst, unit in (("ETH-USDT-SWAP", 1.0), ("BTC-USDT-SWAP", 0.03)):
        Ts, p15, p1h, b5 = lab.precompute(inst, days)
        df = lab.simulate(Ts, p15, p1h, b5, unit, v)
        df["inst"] = inst.split("-")[0]
        df["notional"] = df.entry * unit
        notional[inst] = df.notional.mean()
        frames.append(df)
    df = pd.concat(frames).sort_values("ts").reset_index(drop=True)
    eq = df.net.cumsum(); mdd = float((eq.cummax().clip(lower=0) - eq).max())
    cap = sum(notional.values())
    net = df.net.sum(); ann = net * 365 / days
    print(f"回测 {days} 天：{len(df)} 笔，胜率 {(df.net > 0).mean() * 100:.1f}%，净利 {net:+.1f}U，手续费 {2 * len(df):.0f}U，"
          f"组合最大回撤 {mdd:.1f}U")
    print(f"每笔名义本金：" + "，".join(f"{k.split('-')[0]} 约 {v:,.0f}U" for k, v in notional.items()) + f"，合计约 {cap:,.0f}U")
    print(f"年化利润（固定仓位、不复利）：约 {ann:+,.0f}U / 年")
    print(f"按不加杠杆的名义本金 {cap:,.0f}U 计：年化收益率约 {ann / cap * 100:+.1f}%，最大回撤约 {mdd / cap * 100:.1f}%，"
          f"收益回撤比 {ann / mdd if mdd else float('inf'):.2f}")
    for inst in ("ETH", "BTC"):
        d = df[df.inst == inst]; c = notional[f"{inst}-USDT-SWAP"]
        e = d.net.cumsum(); m = float((e.cummax().clip(lower=0) - e).max()) if len(d) else 0
        a = d.net.sum() * 365 / days
        print(f"  {inst}：{len(d)} 笔，胜率 {(d.net > 0).mean() * 100:.1f}%，净利 {d.net.sum():+.1f}U，年化 {a:+,.0f}U（{a / c * 100:+.1f}%），最大回撤 {m:.1f}U")
    # 按月
    df["月"] = pd.to_datetime(df.ts, unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m")
    print("\n按月净利（U）：\n" + df.groupby(["月", "inst"]).net.sum().unstack(fill_value=0).round(1).assign(合计=lambda x: x.sum(axis=1)).to_string())
