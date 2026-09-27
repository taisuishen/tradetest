"""
动态仓位：按信号 / 趋势强弱开 0.5~5 倍（回测 chan15_lab.py / bt_sizing.py 与实盘 trader15.py 共用）。

评分 0~1，权重事先定好，没有按回测结果调：
  大方向同向程度 30%（1H*0.6+4H*0.4 趋势分，同向 +3 为满分）
  1H 趋势效率     30%（25% 为 0 分，60% 及以上满分）
  1H 缠论走势     20%（已形成与开仓方向相同的趋势得满分）
  4H ADX          20%（15 为 0 分，40 及以上满分）
分档：<0.2→0.5 倍，<0.4→1 倍，<0.6→2 倍，<0.8→3 倍，≥0.8→5 倍
"""


def strength(trend, eff1h, chan1h_trend, adx4h, side):
    want = "上涨" if side > 0 else "下跌"
    s1 = min(max(trend * side / 3, 0), 1)
    s2 = min(max((eff1h - 0.25) / 0.35, 0), 1)
    s3 = 1.0 if chan1h_trend == want else 0.0
    s4 = min(max((adx4h - 15) / 25, 0), 1)
    return 0.3 * s1 + 0.3 * s2 + 0.2 * s3 + 0.2 * s4


def size_from_score(score, mode="step"):
    if mode == "linear":
        return round(0.5 + 4.5 * score, 2)
    for th, m in ((0.2, 0.5), (0.4, 1.0), (0.6, 2.0), (0.8, 3.0)):
        if score < th:
            return m
    return 5.0
