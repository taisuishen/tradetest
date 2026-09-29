"""
OKX 下单接口（私有 REST）：逐仓、只做多，开仓时把止损单一起挂到交易所上。

  - 一律逐仓：下单 tdMode=isolated，设杠杆 / 平仓 mgnMode=isolated
  - 开仓：市价买入，同时附带止损单（attachAlgoOrds：最新价触发，市价成交），程序断线也会止损
  - 平仓：close-position 市价全平，并撤掉该合约剩下的止损单
  - 对账：positions 查当前持仓，positions-history 取已平仓位的真实开平均价、手续费、资金费、净盈亏

API Key 放 okx_api.json（已在 .gitignore 里）或环境变量 OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE；
"simulated": true（或环境变量 OKX_SIMULATED=1）表示模拟盘，请求带 x-simulated-trading: 1。
"""
import base64
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from urllib.parse import urlencode

import requests

import okx_client as ox

ROOT = Path(__file__).parent
_session = requests.Session()
_STATE = {}          # 进程内缓存：账户配置、合约规格、已设置过杠杆的合约


class OkxError(RuntimeError):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


def creds():
    """返回 {"key", "secret", "passphrase", "simulated"}，没有配置返回 None。"""
    k, s, p = os.environ.get("OKX_API_KEY"), os.environ.get("OKX_API_SECRET"), os.environ.get("OKX_API_PASSPHRASE")
    sim = os.environ.get("OKX_SIMULATED")
    f = ROOT / "okx_api.json"
    if not k and f.exists():
        c = json.loads(f.read_text(encoding="utf-8"))
        k, s, p, sim = c.get("api_key"), c.get("secret"), c.get("passphrase"), c.get("simulated", sim)
    if not (k and s and p):
        return None
    return {"key": k, "secret": s, "passphrase": p, "simulated": str(sim).lower() in ("1", "true", "yes")}


def request(method, path, params=None, body=None, retries=4, idempotent=False):
    """签名请求。GET 的参数拼进 requestPath 一起签名；POST 的 body 用同一个 JSON 字符串签名和发送。
    限频 / 繁忙时 GET 和 idempotent=True 的 POST（如设置杠杆，重复调用没有副作用）会退避重试；下单类 POST 不重试。"""
    c = creds()
    if not c:
        raise OkxError("没有配置 OKX API Key（okx_api.json 或环境变量）")
    if params:
        path = path + "?" + urlencode(params)
    data = json.dumps(body, separators=(",", ":")) if body is not None else ""
    last = None
    for i in range(retries):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        sign = base64.b64encode(hmac.new(c["secret"].encode(), (ts + method + path + data).encode(), hashlib.sha256).digest()).decode()
        headers = {"OK-ACCESS-KEY": c["key"], "OK-ACCESS-SIGN": sign, "OK-ACCESS-TIMESTAMP": ts,
                   "OK-ACCESS-PASSPHRASE": c["passphrase"], "Content-Type": "application/json"}
        if c["simulated"]:
            headers["x-simulated-trading"] = "1"
        try:
            r = _session.request(method, ox.BASE + path, data=data or None, headers=headers, timeout=15)
            j = r.json()
        except (requests.RequestException, ValueError) as e:
            last = e
            if method == "POST":     # 下单类请求网络异常时不盲目重试（可能已经成交），交给调用方对账
                raise OkxError(f"{path} 网络异常：{e}")
            time.sleep(1 + i)
            continue
        if j.get("code") == "0":
            return j["data"]
        if j.get("code") in ox.RETRY_CODES and (method == "GET" or idempotent) and i < retries - 1:
            last = OkxError(f"{path} -> {j.get('code')} {j.get('msg')}", j.get("code"))
            time.sleep(1 + i)
            continue
        detail = "; ".join(f"{d.get('sCode')} {d.get('sMsg')}" for d in j.get("data") or [] if isinstance(d, dict) and d.get("sCode") not in (None, "0"))
        raise OkxError(f"{path} -> {j.get('code')} {j.get('msg')}{'（' + detail + '）' if detail else ''}", j.get("code"))
    raise OkxError(f"{path} 重试 {retries} 次仍失败：{last}")


def get(path, **params):
    return request("GET", path, params=params or None)


def post(path, body, idempotent=False):
    return request("POST", path, body=body, idempotent=idempotent)


# ---------------- 账户与合约 ----------------
def account():
    """账户等级与持仓模式（进程内缓存）。acctLv：1 简单（不能做合约）、2 单币种保证金、3 跨币种、4 组合保证金。"""
    if "account" not in _STATE:
        d = get("/api/v5/account/config")[0]
        _STATE["account"] = {"acctLv": d.get("acctLv"), "posMode": d.get("posMode"), "uid": d.get("uid")}
    return _STATE["account"]


def check_account():
    a = account()
    if a["acctLv"] == "1":
        raise OkxError("OKX 账户是“简单交易模式”，不能交易永续合约：请在 OKX 交易设置里改为“单币种保证金”或更高模式")
    return a


def _pos_side():
    """双向持仓模式要带 posSide=long；单向（净持仓）模式不带。"""
    return {"posSide": "long"} if account()["posMode"] == "long_short_mode" else {}


def spec(inst):
    """合约规格：每张面值 ctVal、下单步长 lotSz、最小张数 minSz、价格精度 tickSz。"""
    sp = _STATE.setdefault("spec", {})
    if inst not in sp:
        d = ox.instrument(inst)
        sp[inst] = {k: Decimal(d[k]) for k in ("ctVal", "lotSz", "minSz", "tickSz")}
    return sp[inst]


def num(d):
    """Decimal → 普通小数字符串（避免 1.8E+2 这种科学计数法，OKX 不认）。"""
    return format(Decimal(d).normalize(), "f")


def contracts(inst, qty):
    """币数 → 张数（按 lotSz 向下取整）。"""
    s = spec(inst)
    n = (Decimal(str(qty)) / s["ctVal"] / s["lotSz"]).to_integral_value(ROUND_DOWN) * s["lotSz"]
    return n if n >= s["minSz"] else Decimal(0)


def price_str(inst, px):
    """价格按 tickSz 向下取整（多单止损略低一点，不会比策略止损更早触发）。"""
    t = spec(inst)["tickSz"]
    return num((Decimal(str(px)) / t).to_integral_value(ROUND_DOWN) * t)


def ensure_leverage(inst, lever):
    """逐仓杠杆，每个进程每个合约只设一次。"""
    done = _STATE.setdefault("lever", {})
    if done.get(inst) == lever:
        return
    post("/api/v5/account/set-leverage", {"instId": inst, "lever": str(lever), "mgnMode": "isolated", **_pos_side()}, idempotent=True)
    done[inst] = lever


def balance_usdt():
    d = get("/api/v5/account/balance", ccy="USDT")[0]
    u = next((x for x in d.get("details", []) if x.get("ccy") == "USDT"), {})
    return {"totalEq": float(d.get("totalEq") or 0), "eq": float(u.get("eq") or 0), "availBal": float(u.get("availBal") or 0)}


# ---------------- 持仓与挂单 ----------------
def position(inst):
    """当前逐仓多单：{"sz": 张数, "avgPx", "liqPx", "lever"}；没有返回 None。"""
    for p in get("/api/v5/account/positions", instType="SWAP", instId=inst):
        if p.get("mgnMode") != "isolated" or not p.get("pos") or float(p["pos"]) == 0:
            continue
        if p.get("posSide") == "short" or (p.get("posSide") == "net" and float(p["pos"]) < 0):
            continue
        return {"sz": abs(float(p["pos"])), "avgPx": float(p.get("avgPx") or 0), "liqPx": float(p.get("liqPx") or 0),
                "lever": p.get("lever"), "upl": float(p.get("upl") or 0)}
    return None


def pending_stops(inst):
    """该合约挂着的止损类委托（附带止损在开仓成交后会变成 conditional / oco 委托）。"""
    out = []
    for t in ("conditional", "oco"):
        out += get("/api/v5/trade/orders-algo-pending", ordType=t, instId=inst)
    return out


def cancel_stops(inst):
    algos = [{"algoId": a["algoId"], "instId": inst} for a in pending_stops(inst)]
    if algos:
        post("/api/v5/trade/cancel-algos", algos)
    return len(algos)


def _order(inst, ord_id, tries=10):
    for _ in range(tries):
        o = get("/api/v5/trade/order", instId=inst, ordId=ord_id)[0]
        if o.get("state") in ("filled", "canceled", "mmp_canceled"):
            return o
        time.sleep(0.3)
    return o


def _cl_id(prefix, tid):
    return f"{prefix}{tid}x{int(time.time()) % 100_000_000}"      # 字母数字、≤32 位；带时间避免换库后重号


# ---------------- 开平仓 ----------------
def open_long(inst, qty, stop_px, lever, tid):
    """市价开多 + 附带止损。返回 {"sz", "avgPx", "fee", "ordId", "stop"}。
    成交后核对交易所上确实挂着止损单；没有就补挂一张，补挂也失败就立即平仓（绝不留没有止损的仓位）。"""
    check_account()
    if position(inst):
        raise OkxError(f"{inst} 交易所上已有逐仓多单，不重复开仓（请先人工确认）")
    sz = contracts(inst, qty)
    if not sz:
        s = spec(inst)
        raise OkxError(f"{inst} 下单数量 {qty} 个币不足最小 {s['minSz']} 张（每张 {s['ctVal']}）")
    ensure_leverage(inst, lever)
    stop = price_str(inst, stop_px)
    cl = _cl_id("t15o", tid)
    body = {"instId": inst, "tdMode": "isolated", "side": "buy", "ordType": "market", "sz": num(sz), "clOrdId": cl,
            **_pos_side(),
            "attachAlgoOrds": [{"attachAlgoClOrdId": cl + "s", "slTriggerPx": stop, "slOrdPx": "-1", "slTriggerPxType": "last"}]}
    r = post("/api/v5/trade/order", body)[0]
    o = _order(inst, r["ordId"])
    if float(o.get("accFillSz") or 0) == 0:
        raise OkxError(f"{inst} 市价单未成交（状态 {o.get('state')}）")
    res = {"sz": float(o["accFillSz"]), "avgPx": float(o["avgPx"]), "fee": -float(o.get("fee") or 0), "ordId": r["ordId"], "stop": float(stop)}
    time.sleep(0.5)
    note = ensure_stop(inst, stop_px, tid)
    pos = position(inst)
    if pos and pos["liqPx"] and pos["liqPx"] >= float(stop):
        note = (note + f"；警告：强平价 {pos['liqPx']} 不低于止损价 {stop}").lstrip("；")
    if note:
        res["note"] = note
    return res


def ensure_stop(inst, stop_px, tid=0):
    """确认交易所上挂着止损单；没有就按当前持仓补挂一张（只减仓、最新价触发、市价成交），补挂失败就立即平仓。
    返回说明文字（已有止损时为空）。"""
    if pending_stops(inst):
        return ""
    pos = position(inst)
    if not pos:
        return ""
    stop = price_str(inst, stop_px)
    try:
        post("/api/v5/trade/order-algo", {"instId": inst, "tdMode": "isolated", "side": "sell", "ordType": "conditional",
                                          "sz": num(str(pos["sz"])), "reduceOnly": "true", **_pos_side(),
                                          "slTriggerPx": stop, "slOrdPx": "-1", "slTriggerPxType": "last"})
        return f"止损单未生效，已补挂止损 {stop}"
    except Exception as e:
        close_long(inst, tid)
        raise OkxError(f"{inst} 挂止损失败，已立即平仓（不留没有止损的仓位）：{e}")


def close_long(inst, tid=0):
    """市价全平逐仓多单并撤掉剩余止损单；已经没有仓位时只撤单。返回是否真的下了平仓单。"""
    closed = False
    if position(inst):
        post("/api/v5/trade/close-position", {"instId": inst, "mgnMode": "isolated", "autoCxl": True,
                                              "clOrdId": _cl_id("t15c", tid), "posSide": _pos_side().get("posSide", "net")})
        closed = True
        time.sleep(0.5)
    try:
        cancel_stops(inst)
    except Exception:
        pass                 # 仓位平掉后附带止损通常会自动失效，撤单失败不影响结果
    return closed


def closed_record(inst, since_ms):
    """since_ms 之后开、并已平掉的逐仓多单：真实开平均价、手续费、资金费、净盈亏（realizedPnl 已含手续费和资金费）。
    同一合约的仓位是一笔接一笔的，所以取 since_ms 之后最早开的那一条（接口按时间倒序返回，后面几笔也满足“之后”）。
    type：1 部分平仓 2 完全平仓 3 强平 4 部分强平 5 自动减仓（ADL）。平仓后几秒内可能还查不到，查不到返回 None。"""
    # 留 5 秒本机与交易所的时钟误差；单向持仓模式下 posId 会复用，不能用来区分
    hs = [h for h in get("/api/v5/account/positions-history", instType="SWAP", instId=inst, mgnMode="isolated", limit="20")
          if int(h.get("cTime") or 0) >= since_ms - 5000 and int(h.get("uTime") or 0) >= since_ms and h.get("direction") != "short"]
    if not hs:
        return None
    h = min(hs, key=lambda x: int(x["cTime"]))
    return {"openAvgPx": float(h.get("openAvgPx") or 0), "closeAvgPx": float(h.get("closeAvgPx") or 0),
            "net": float(h.get("realizedPnl") or 0), "fee": -float(h.get("fee") or 0), "funding": -float(h.get("fundingFee") or 0),
            "type": {"1": "部分平仓", "2": "平仓", "3": "强平", "4": "部分强平", "5": "自动减仓"}.get(h.get("type"), h.get("type")),
            "uTime": int(h.get("uTime") or 0)}
