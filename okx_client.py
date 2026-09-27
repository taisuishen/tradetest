"""OKX 公共行情接口（无需 API Key）。"""
import time
import requests
import pandas as pd

BASE = "https://www.okx.com"
_session = requests.Session()


def get(path, params=None, retries=3):
    for i in range(retries):
        try:
            r = _session.get(BASE + path, params=params, timeout=15)
            j = r.json()
            if j.get("code") == "0":
                return j["data"]
            if j.get("code") == "50011":  # 限频
                time.sleep(1 + i)
                continue
            raise RuntimeError(f"{path} {params} -> {j.get('code')} {j.get('msg')}")
        except (requests.RequestException, ValueError):
            if i == retries - 1:
                raise
            time.sleep(1 + i)
    raise RuntimeError(f"{path} 请求失败")


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
    df = df[df.confirm == 1].reset_index(drop=True)
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
