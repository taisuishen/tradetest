"""策略 v3 最近 N 天按月收益：多空都做（实盘）/ 只做多 / 只做空，每笔名义 = 开仓时已实现权益 × 1（= 逐仓 10 倍、保证金 10%），复利。
各方案 × 各品种的撮合互相独立，用多进程并行跑。

用法：python bt_monthly.py [天数=180]                    最近 N 天
      python bt_monthly.py 2025-01-01 2026-01-01          指定区间（北京时间，含起不含止）
      首次运行会拉 K 线并预计算（缓存在 bt/）；费用按 OKX Lv1 标准吃单 0.05%，不计滑点
"""
import os, sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import chan15_lab2 as L

TZ = "Asia/Shanghai"
if len(sys.argv) > 2:                       # 指定区间：按结束时间往前推天数
    _s, _e = (pd.Timestamp(x, tz=TZ) for x in sys.argv[1:3])
    DAYS, END_MS = (_e - _s).days, int(_e.timestamp() * 1000)
else:
    DAYS, END_MS = (int(sys.argv[1]) if len(sys.argv) > 1 else 180), None
VARIANTS = {"多空都做（实盘）": {"三买", "三卖"}, "只做多": {"三买"}, "只做空": {"三卖"}}
_D = {}


def _init():
    insts = L.live_insts()
    _D.update({i: d for i, d in L.load(DAYS, insts, "_live", END_MS).items() if i in dict(insts)})


def job(args):
    name, inst = args
    d = _D[inst]
    t = L.simulate(d, {"sigs": [("trend", {"eff_min": 0.2, "zs_w_max": 1.5, "types": VARIANTS[name]})]})
    if len(t):
        t["inst"] = inst.split("-")[0]; t["方案"] = name
        t["ret"] = t.net / (t.entry * d[0] * t.mult)
    return t


def compound(df):
    """按事件顺序：开仓时名义 = 已实现权益，平仓时权益 += 名义 × 收益率。返回 (逐笔平仓后的权益曲线, 最多同时持仓)。"""
    ev = sorted([(r.ts, 1, k) for k, r in df.iterrows()] + [(r.end, 0, k) for k, r in df.iterrows()], key=lambda x: (x[0], x[1]))
    eq, notional, curve, n, mx = 1.0, {}, [], 0, 0
    for ts, is_open, k in ev:
        if is_open:
            notional[k] = eq; n += 1; mx = max(mx, n)
        else:
            eq += notional[k] * df.at[k, "ret"]; n -= 1; curve.append((ts, eq))
    cv = pd.DataFrame(curve, columns=["ts", "eq"])
    cv["月"] = pd.to_datetime(cv.ts, unit="ms", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m")
    return cv, mx


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _init()
    insts = list(_D)
    tasks = [(v, i) for v in VARIANTS for i in insts]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1), initializer=_init) as ex:
        res = [r for r in ex.map(job, tasks) if len(r)]
    all_ = pd.concat(res)
    months = sorted(pd.to_datetime(all_.end, unit="ms", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m").unique())
    table, summary = {}, []
    for name in VARIANTS:
        df = all_[all_["方案"] == name].sort_values("ts").reset_index(drop=True)
        df["月"] = pd.to_datetime(df.end, unit="ms", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m")
        cv, mx = compound(df)
        col, prev = {}, 1.0
        for m in months:
            g = df[df["月"] == m]
            end = cv[cv["月"] == m]["eq"].iloc[-1] if (cv["月"] == m).any() else prev
            col[m] = f"{end / prev - 1:+.1%}（{len(g)} 笔 {(g.net > 0).mean() if len(g) else 0:.0%}）"
            prev = end
        table[name] = col
        peak = cv["eq"].cummax(); mdd = float((1 - cv["eq"] / peak).max())
        top5 = df.ret.nlargest(5).sum()
        summary.append({"方案": name, "笔数": len(df), "胜率": f"{(df.net > 0).mean():.0%}", "6 个月复利": f"{cv['eq'].iloc[-1] - 1:+.1%}",
                        "不复利合计": f"{df.ret.sum():+.1%}", "最大回撤": f"{mdd:.1%}", "最多同时持仓": mx,
                        "去掉最赚 5 笔（不复利）": f"{df.ret.sum() - top5:+.1%}",
                        "多单合计": f"{df[df.side > 0].ret.sum():+.1%}（{(df.side > 0).sum()} 笔）",
                        "空单合计": f"{df[df.side < 0].ret.sum():+.1%}（{(df.side < 0).sum()} 笔）"})
    pd.set_option("display.width", 250); pd.set_option("display.unicode.east_asian_width", True)
    print("按月复利收益（括号内：当月平仓笔数 胜率）")
    print(pd.DataFrame(table).to_string())
    print("\n" + pd.DataFrame(summary).to_string(index=False))
