"""
Christian（Qullamaggie）日线整理突破（bt_top10.m10_christian）在服务器 8 个品种上的回测。
每个品种各自一个仓位；组合按“每笔名义 = 开仓时已实现权益 × 1”复利，多个品种可以同时持仓。
吃单 0.05%，不计滑点；止损用 5 分钟 K 线判定。分段：最近 2 年 / 2025 全年 / 最近 180 天。

用法：python bt_christian.py [天数=730]
"""
import sys

import numpy as np
import pandas as pd

import bt_top10 as T
import chan15_lab2 as L

TZ = "Asia/Shanghai"


def run_inst(inst, start):
    D, H1, m5 = T.load("1D", inst), T.load("1H", inst), T.load("5m", inst)
    tr = T.m10_christian(D, H1, m5, max(start, int(m5.ts.iloc[0]) + 86_400_000))
    if len(tr):
        tr["inst"] = inst.split("-")[0]
    return tr


def summary(df, days, name):
    if not len(df):
        return {"方案": name, "笔数": 0}
    x = df.rename(columns={"entry_ts": "ts", "exit_ts": "end"}).sort_values("ts").reset_index(drop=True)
    cv, mx = L.compound(x)
    mdd = float((1 - cv["eq"] / cv["eq"].cummax()).max())
    r = x.ret.values
    day = pd.to_datetime(x.ts, unit="ms", utc=True).dt.tz_convert(TZ).dt.floor("D")
    gaps = np.diff(np.sort(x.ts.values)) / 86_400_000
    return {"方案": name, "笔数": len(x), "每月笔数": f"{len(x) / days * 30:.1f}", "有开仓的天数": f"{day.nunique() / days:.0%}",
            "最长空窗": f"{gaps.max():.0f} 天" if len(gaps) else "-", "胜率": f"{(r > 0).mean():.0%}",
            "均盈 / 均亏": " / ".join(f"{x.mean() * 100:+.1f}%" if len(x) else "-" for x in (r[r > 0], r[r <= 0])),
            "复利": f"{(cv['eq'].iloc[-1] - 1) * 100:+.0f}%", "最大回撤": f"{mdd:.0%}", "最多同时持仓": mx,
            "去掉最赚 5 笔（不复利）": f"{(r.sum() - np.sort(r)[-5:].sum()) * 100:+.0f}%"}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 260); pd.set_option("display.unicode.east_asian_width", True)
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 730
    insts = [i for i, _ in L.live_insts()]
    end = min(int(T.load("5m", i).ts.iloc[-1]) for i in insts)
    start = end - days * T.MS["1D"]
    tr = pd.concat([run_inst(i, start) for i in insts])
    tr = tr[tr.entry_ts < end]
    e25s = int(pd.Timestamp("2025-01-01", tz=TZ).timestamp() * 1000); e25e = int(pd.Timestamp("2026-01-01", tz=TZ).timestamp() * 1000)
    seg = [(f"最近 {days} 天（8 个品种）", tr, days)]
    if days >= 730:
        seg += [("2025 全年", tr[(tr.entry_ts >= e25s) & (tr.entry_ts < e25e)], 365)]
    if days > 180:
        seg += [("最近 180 天", tr[tr.entry_ts >= end - 180 * T.MS["1D"]], 180)]
    print(pd.DataFrame([summary(d, n, k) for k, d, n in seg]).set_index("方案").T.to_string())
    per = pd.DataFrame([{**summary(tr[tr.inst == c], days, c)} for c in sorted(tr.inst.unique())]).set_index("方案")
    print(f"\n分品种（最近 {days} 天）：\n" + per[["笔数", "胜率", "均盈 / 均亏", "复利", "最大回撤"]].to_string())
    mon = tr.assign(月=pd.to_datetime(tr.entry_ts, unit="ms", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m")).groupby("月").agg(笔数=("ret", "size"), 合计=("ret", lambda s: f"{s.sum() * 100:+.1f}%"))
    print("\n按开仓月份（不复利合计）：\n" + mon.T.to_string())
    tr.to_csv(L.OUT / "christian_trades.csv", index=False)
