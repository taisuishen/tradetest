"""
K 线本地缓存 + 并行拉取（回测用）。

OKX 历史 K 线每次最多 100 根、按 IP 限速每 2 秒 20 次。每段请求的时间窗口可以事先算好，
所以用多线程并行拉，并用令牌桶把速度控制在限速以内（默认每秒 6 次，给同机运行的模拟盘留出余量）；
拉过的数据存到 cache/ 目录，下次只补拉缺的部分。
"""
import gzip
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import okx_client as ox

CACHE = Path(__file__).parent / "cache"
BAR_MS = ox.BAR_MS


class RateLimiter:
    def __init__(self, per_sec):
        self.gap, self.lock, self.next = 1.0 / per_sec, threading.Lock(), time.time()

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next)
            self.next = t + self.gap
        if t > now:
            time.sleep(t - now)


def _fetch_window(inst, bar, after_ms, limiter):
    limiter.wait()
    rows = ox.get("/api/v5/market/history-candles", {"instId": inst, "bar": bar, "limit": 100, "after": after_ms})
    return [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[7]), r[8]) for r in rows]


def candles(inst, bar, n, workers=6, per_sec=6, progress=True):
    """返回最近 n 根已收盘 K 线（升序，列同 okx_client.candles）。"""
    dur = BAR_MS[bar]
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{inst}_{bar}.pkl.gz"
    have = {}
    if f.exists():
        with gzip.open(f, "rb") as fh:
            have = pickle.load(fh)
    now = int(time.time() * 1000)
    end = now // dur * dur                        # 当前未收盘 K 线的开盘时间
    start = end - n * dur
    # 需要补的时间窗口：每段 100 根，"after=X" 返回 X 之前的 100 根
    missing = [t for t in range(start, end, dur) if t not in have]
    windows = sorted({(t // (100 * dur) + 1) * 100 * dur for t in missing})
    windows = [min(w, end) for w in windows]
    if windows:
        lim = RateLimiter(per_sec)
        t0 = time.time()
        done, failed = [0], []
        def job(w):
            try:
                r = _fetch_window(inst, bar, w, lim)
            except Exception:
                failed.append(w)                     # 限频等失败：先记下，最后低速补拉
                return []
            done[0] += 1
            if progress and done[0] % 200 == 0:
                print(f"  {inst} {bar}：{done[0]}/{len(windows)} 段，{time.time() - t0:.0f}s", flush=True)
            return r
        def keep(rows):
            for r in rows:
                if r[6] == "1" or r[0] + dur <= now - 1000:
                    have[r[0]] = r[:6]
        try:
            with ThreadPoolExecutor(workers) as ex:
                for rows in ex.map(job, windows):
                    keep(rows)
            for w in list(failed):                   # 失败的窗口单线程慢慢补
                time.sleep(0.5)
                keep(_fetch_window(inst, bar, w, lim))
                failed.remove(w)
        finally:                                     # 无论成功与否都把已拉到的存下来，下次接着补
            with gzip.open(f, "wb") as fh:
                pickle.dump(have, fh)
        if progress:
            print(f"  {inst} {bar}：补拉 {len(windows)} 段，用时 {time.time() - t0:.0f}s，缓存共 {len(have)} 根", flush=True)
    ts = sorted(t for t in have if start <= t < end)
    df = pd.DataFrame([(t, *have[t][1:]) for t in ts], columns=["ts", "o", "h", "l", "c", "volQuote"])
    df["time"] = pd.to_datetime(df.ts, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    return df
