"""
OKX 合约真实费用：交易手续费 + 资金费（回测与实盘共用）。

交易手续费：成交金额 × 费率。开仓、止损、反向离场都是市价单，按吃单（taker）费率，开平各收一次。
  - 配置了 OKX 只读 API Key 时，调用 /api/v5/account/trade-fee 取账户真实费率（与 VIP 等级有关），缓存 24 小时
  - 否则用 OKX 普通用户 Lv1 U 本位永续标准费率：挂单 0.02%，吃单 0.05%
  API Key 放环境变量 OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE，或 okx_api.json（已在 .gitignore 里），读取与签名见 okx_trade.py
资金费：持仓期间每经过一个结算时刻，按 OKX 实际资金费率计：费用 = 方向 × 费率 × 持仓数量 × 开仓价
  （费率为正时多单付、空单收；用开仓价近似结算时的持仓价值）
"""
import json
import time
from pathlib import Path

import okx_client as ox

ROOT = Path(__file__).parent
CACHE = ROOT / "fee_rates.json"
DEFAULT = {"taker": 0.0005, "maker": 0.0002, "source": "OKX 普通用户 Lv1 标准费率（未配置 API Key）"}


def _creds():
    import okx_trade            # Key 的读取与签名统一在 okx_trade（含模拟盘请求头）
    return okx_trade.creds()


def _signed_get(path, creds):
    import okx_trade
    return okx_trade.request("GET", path)


def rates(inst):
    """返回 {"taker", "maker", "source"}；有 API Key 就取账户真实费率（缓存 24 小时），失败或没有 Key 用 Lv1 标准费率。"""
    creds = _creds()
    if not creds:
        return dict(DEFAULT)
    family = "-".join(inst.split("-")[:2])
    try:
        cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    except Exception:
        cache = {}
    c = cache.get(family)
    if c and time.time() - c["ts"] < 86400:
        return c["rates"]
    try:
        d = _signed_get(f"/api/v5/account/trade-fee?instType=SWAP&instFamily={family}", creds)[0]
        taker = abs(float(d.get("takerU") or d.get("taker")))   # OKX 返回负数表示收取的费率
        maker = abs(float(d.get("makerU") or d.get("maker")))
        r = {"taker": taker, "maker": maker, "source": f"OKX 账户真实费率（等级 {d.get('level', '-')}）"}
        cache[family] = {"ts": time.time(), "rates": r}
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        return r
    except Exception as e:
        return {**DEFAULT, "source": f"获取账户费率失败（{str(e)[:60]}），暂用 Lv1 标准费率"}


def funding_history(inst, since_ms):
    """since_ms 之后的历史资金费 [(结算时间, 费率)]，升序。"""
    rows, after = [], None
    while True:
        p = {"instId": inst, "limit": 100}
        if after:
            p["after"] = after
        d = ox.get("/api/v5/public/funding-rate-history", p)
        if not d:
            break
        rows += [(int(x["fundingTime"]), float(x["realizedRate"] or x["fundingRate"])) for x in d]
        after = d[-1]["fundingTime"]
        if int(after) <= since_ms or len(d) < 100:
            break
    return sorted({r for r in rows if r[0] > since_ms})


def trade_fee(entry_px, exit_px, qty, taker):
    return (entry_px + exit_px) * qty * taker


def funding_cost(side, qty, entry_px, start_ms, end_ms, funding):
    """持仓期间 (start, end] 内每次结算：多单付 费率×价值，空单反之。正数为成本。"""
    return sum(side * r * qty * entry_px for t, r in funding if start_ms < t <= end_ms)
