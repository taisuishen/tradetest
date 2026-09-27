"""OKX 公共行情接口（无需 API Key）。"""
import random
import time
import requests
import pandas as pd

BASE = "https://www.okx.com"
_session = requests.Session()
# 可重试的 OKX 错误码：50011 限频、50001 服务暂不可用、50004 接口超时、50013 系统繁忙、50026 系统错误
RETRY_CODES = {"50011", "50001", "50004", "50013", "50026"}
BAR_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1H": 3_600_000, "2H": 7_200_000,
          "4H": 14_400_000, "1D": 86_400_000}


def get(path, params=None, retries=5):
    """网络异常 / 5xx / 可重试错误码时按指数退避重试（1, 2, 4, 8… 秒，最长 20 秒，带随机抖动）。"""
    last = None
    for i in range(retries):
        try:
            r = _session.get(BASE + path, params=params, timeout=15)
            if r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            j = r.json()
            if j.get("code") == "0":
                return j["data"]
            if j.get("code") not in RETRY_CODES:
                raise RuntimeError(f"{path} {params} -> {j.get('code')} {j.get('msg')}")
            last = RuntimeError(f"{path} -> {j.get('code')} {j.get('msg')}")
        except (requests.RequestException, ValueError) as e:
            last = e
        if i < retries - 1:
            time.sleep(min(2 ** i, 20) + random.random())
    raise RuntimeError(f"{path} 重试 {retries} 次仍失败：{last}")


def reachable():
    """检测能否连上 OKX（服务器时间接口）。"""
    try:
        return bool(get("/api/v5/public/time", retries=3))
    except Exception:
        return False


def instrument(inst_id):
    inst_type = "SWAP" if inst_id.endswith("SWAP") else "FUTURES"
    return get("/api/v5/public/instruments", {"instType": inst_type, "instId": inst_id})[0]


def candles(inst_id, bar, n=500):
    """返回按时间升序的 K 线 DataFrame，只保留已收盘 K 线（收盘原则）。"""
    rows = get("/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": 300})
    while len(rows) < n:
        more = get("/api/v5/market/history-candles",
                   {"instId": inst_id, "bar": bar, "limit": 100, "after": rows[-1][0]})
        if not more:
            break
        rows += more
    df = pd.DataFrame(rows[:n], columns=["ts", "o", "h", "l", "c", "vol", "volCcy", "volQuote", "confirm"])
    df = df.astype({"ts": "int64", "o": float, "h": float, "l": float, "c": float,
                    "vol": float, "volCcy": float, "volQuote": float, "confirm": int})
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    # 已收盘：OKX 标记 confirm=1，或该 K 线的结束时间已过（刚收盘时 confirm 标记可能晚几秒更新，不等它）
    dur = BAR_MS.get(bar)
    now = time.time() * 1000
    df = df[(df.confirm == 1) | ((df.ts + dur <= now - 1000) if dur else False)].reset_index(drop=True)
    df["time"] = pd.to_datetime(df.ts, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    return df


def ticker(inst_id):
    return get("/api/v5/market/ticker", {"instId": inst_id})[0]


def books_full(inst_id, sz=5000):
    return get("/api/v5/market/books-full", {"instId": inst_id, "sz": sz})[0]


def funding(inst_id):
    return get("/api/v5/public/funding-rate", {"instId": inst_id})[0]


def funding_history(inst_id, limit=90):
    return get("/api/v5/public/funding-rate-history", {"instId": inst_id, "limit": limit})


def open_interest(inst_id):
    return get("/api/v5/public/open-interest", {"instType": "SWAP", "instId": inst_id})[0]


def oi_history(inst_id, period="1H", limit=100):
    return get("/api/v5/rubik/stat/contracts/open-interest-history",
               {"instId": inst_id, "period": period, "limit": limit})


def ls_ratio_all(inst_id, period="1H", limit=100):
    return get("/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
               {"instId": inst_id, "period": period, "limit": limit})


def ls_ratio_top_account(inst_id, period="1H", limit=100):
    return get("/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader",
               {"instId": inst_id, "period": period, "limit": limit})


def ls_ratio_top_position(inst_id, period="1H", limit=100):
    return get("/api/v5/rubik/stat/contracts/long-short-position-ratio-contract-top-trader",
               {"instId": inst_id, "period": period, "limit": limit})


def taker_volume(inst_id, period="1H", limit=100):
    return get("/api/v5/rubik/stat/taker-volume-contract",
               {"instId": inst_id, "period": period, "limit": limit})


def trades(inst_id, limit=500):
    return get("/api/v5/market/trades", {"instId": inst_id, "limit": limit})


def mark_price(inst_id):
    return get("/api/v5/public/mark-price", {"instType": "SWAP", "instId": inst_id})[0]


def index_ticker(index_id):
    return get("/api/v5/market/index-tickers", {"instId": index_id})[0]


def liquidations(uly, limit=100):
    d = get("/api/v5/public/liquidation-orders",
            {"instType": "SWAP", "uly": uly, "state": "filled", "limit": limit})
    return d[0]["details"] if d else []
