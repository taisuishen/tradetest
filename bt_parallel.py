"""
回测并行加速：每个 1H 时间点的分析（指标、结构、缠论、挂单计划）只依赖该时刻之前的数据，彼此独立，
所以先用多进程把所有时间点算好，再在主进程里按时间顺序模拟开平仓（持仓、冷却等状态只在主进程里）。
Python 多线程受 GIL 限制对纯计算无效，这里用多进程。
"""
import os
from concurrent.futures import ProcessPoolExecutor

H = 3_600_000
_G = {}


def _init(data):
    _G.update(data)


def _slices(T):
    df1h, df4h, df1d = _G["df1h"], _G["df4h"], _G["df1d"]
    d = {"df1h": df1h[df1h.ts + H <= T].iloc[-400:].reset_index(drop=True),
         "df4h": df4h[df4h.ts + 4 * H <= T].iloc[-300:].reset_index(drop=True),
         "df1d": df1d[df1d.ts + 24 * H <= T].iloc[-300:].reset_index(drop=True),
         "frank": 50.0, "walls": []}
    if len(d["df1h"]) < 300 or len(d["df4h"]) < 200 or len(d["df1d"]) < 170:
        return None
    d["price"] = float(d["df1h"].c.iloc[-1])
    return d


def _points(df1h):
    import chan
    return [(p["type"], int(df1h.ts.iloc[p["i"]])) for p in chan.analyze(df1h)["points"]]


def plan_task(T):
    """分批挂单策略：该时刻的挂单计划 + 缠论买卖点（供提前离场判断）。"""
    import paper_trader as pt
    d = _slices(T)
    if d is None:
        return T, None
    cfg = _G["cfg"]
    return T, {"plan": pt.plan(d, cfg, _G["unit"]), "close": d["price"],
               "pts": _points(d["df1h"]) if cfg.get("use_chan") else []}


def chan_task(T):
    """纯缠论策略：该时刻的开仓判断 + 缠论买卖点（供反向离场判断）。"""
    import chan_strategy as cs
    d = _slices(T)
    if d is None:
        return T, None
    return T, {"decide": cs.decide(d, _G["cfg"], _G["unit"], set()), "close": d["price"], "pts": _points(d["df1h"])}


def precompute(task, Ts, data, workers=None):
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(data,)) as ex:
        return dict(ex.map(task, Ts, chunksize=max(1, len(Ts) // (workers * 8))))
