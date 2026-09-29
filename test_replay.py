"""
实盘程序端到端回放测试：让 trader15.py 原封不动地按真实时间顺序跑过去 N 天（每 5 分钟一轮 run_once，和服务器一样），
行情来自本地缓存的真实 K 线，OKX 换成一个会自己动的假交易所：
  - 按 1 分钟 K 线算浮盈，价格碰到止损单就按止损价（跳空则按开盘价）成交；维护余额、逐仓保证金、持仓、仓位历史
  - 手续费吃单 0.05%；下单保证金不够就拒单（51008）
可注入故障（--faults）：下单成交后网络断开、平仓失败、仓位历史延迟出现、下单途中进程被杀（随后“重启”）
每一轮检查不变量：交易所每个持仓都有止损单；交易所持仓与数据库 live 记录一一对应；没有孤儿仓位。
最后与回测（chan15_lab2 v5）逐笔比较开仓。

用法：python test_replay.py [天数=30] [--faults [--crash 0.03] [--seed 1]] [--no-compare]
      需要先有本地 K 线缓存（kcache，1m / 15m / 1H / 4H）
"""
import gzip
import json
import pickle
import random
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

import fees
import okx_client as ox
import okx_trade as lt
import trader15 as tr

M1, M15 = 60_000, 900_000
NOW = [0]
CACHE = Path(__file__).parent / "cache"


class Crash(BaseException):
    """模拟进程在下单途中被杀：不是 Exception，trader15 的 except Exception 接不住，会一路冒到最外层。"""


# ---------------- 行情（真实 K 线缓存）----------------
class Market:
    def __init__(self, insts):
        self.k = {}
        for inst in insts:
            for bar in ("1m", "15m", "1H", "4H"):
                h = pickle.load(gzip.open(CACHE / f"{inst}_{bar}.pkl.gz", "rb"))
                ts = np.array(sorted(h), dtype="int64")
                self.k[(inst, bar)] = (ts, np.array([h[t][1:6] for t in ts], dtype=float))   # o h l c volQuote
        self.memo = {}

    def closed(self, inst, bar, n):
        ts, a = self.k[(inst, bar)]
        j = np.searchsorted(ts, NOW[0] - ox.BAR_MS[bar], side="right")      # ts + dur <= NOW 的已收盘 K 线
        return ts[max(0, j - n):j], a[max(0, j - n):j]

    def price(self, inst):
        ts, a = self.closed(inst, "1m", 1)
        return float(a[-1, 3])

    def candles(self, inst, bar, n=500):
        ts, a = self.closed(inst, bar, n)
        key = (inst, bar, n, int(ts[-1]))
        if key not in self.memo:
            d = pd.DataFrame({"ts": ts, "o": a[:, 0], "h": a[:, 1], "l": a[:, 2], "c": a[:, 3], "volQuote": a[:, 4]})
            d["time"] = pd.to_datetime(d.ts, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
            self.memo[key] = d
        return self.memo[key].copy()

    def raw_1m(self, inst, limit, before=None):
        """OKX /market/candles 格式（新 → 旧，全是已收盘）。"""
        ts, a = self.k[(inst, "1m")]
        j = np.searchsorted(ts, NOW[0] - M1, side="right")
        if before is not None:
            j = min(j, np.searchsorted(ts, before, side="left"))
        i = max(0, j - limit)
        return [[str(ts[x]), *(str(v) for v in a[x, :4]), "0", "0", str(a[x, 4]), "1"] for x in range(j - 1, i - 1, -1)]


# ---------------- 假交易所 ----------------
class FakeOKX:
    def __init__(self, mkt, specs, cash=100_000.0, faults=False, seed=1, crash=0.03):
        self.m, self.specs, self.cash, self.faults, self.crash = mkt, specs, cash, faults, crash
        self.rnd = random.Random(seed)
        self.pos, self.algos, self.hist, self.orders = {}, {}, [], {}
        self.scan, self.n = {}, 0
        self.log = {"开仓": 0, "交易所止损": 0, "平仓": 0, "拒单": 0, "注入-下单后断网": 0, "注入-平仓失败": 0, "注入-进程被杀": 0}

    def notional(self, inst, sz, px):
        return sz * float(self.specs[inst]["ctVal"]) * px

    def upl(self, inst):
        p = self.pos[inst]
        return (self.m.price(inst) - p["avgPx"]) * p["side"] * p["sz"] * float(self.specs[inst]["ctVal"])

    def equity(self):
        return self.cash + sum(self.upl(i) for i in self.pos)

    def avail(self):
        return self.cash - sum(p["margin"] for p in self.pos.values())

    def _close(self, inst, px, t, typ="2"):
        p = self.pos.pop(inst)
        ct = float(self.specs[inst]["ctVal"])
        gross = (px - p["avgPx"]) * p["side"] * p["sz"] * ct
        fee = self.notional(inst, p["sz"], px) * 0.0005
        self.cash += gross - fee
        self.hist.append({"instId": inst, "direction": "long" if p["side"] > 0 else "short", "openAvgPx": str(p["avgPx"]),
                          "closeAvgPx": str(px), "realizedPnl": str(gross - fee - p["fee"]), "fee": str(-(fee + p["fee"])),
                          "fundingFee": "0", "type": typ, "cTime": str(p["cTime"]), "uTime": str(t),
                          "_visible": t + (self.rnd.choice([0, 60_000, 180_000]) if self.faults else 0)})
        self.algos.pop(inst, None)

    def advance(self, t):
        """时间推进到 t：逐根 1m K 线检查止损单是否触发。"""
        for inst in list(self.pos):
            ts, a = self.m.k[(inst, "1m")]
            i0 = np.searchsorted(ts, self.scan.get(inst, self.pos[inst]["cTime"]) - M1 + 1, side="left")
            i1 = np.searchsorted(ts, t - M1, side="right")
            p = self.pos[inst]
            for x in range(i0, i1):
                st = [float(g["slTriggerPx"]) for g in self.algos.get(inst, [])]
                if not st:
                    break
                o, h, l = a[x, 0], a[x, 1], a[x, 2]
                trig = max(st) if p["side"] > 0 else min(st)
                if (l <= trig) if p["side"] > 0 else (h >= trig):
                    fill = min(o, trig) if p["side"] > 0 else max(o, trig)       # 跳空穿过止损按开盘价
                    self._close(inst, fill, int(ts[x]) + 30_000)
                    self.log["交易所止损"] += 1
                    break
            self.scan[inst] = t

    def request(self, method, path, params=None, body=None, retries=4, idempotent=False):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(path); q = {k: v[0] for k, v in parse_qs(u.query).items()}; q.update(params or {}); p = u.path
        t = NOW[0]
        if p == "/api/v5/account/config":
            return [{"acctLv": "2", "posMode": "net_mode", "uid": "0"}]
        if p == "/api/v5/account/balance":
            return [{"totalEq": str(self.equity()), "details": [{"ccy": "USDT", "eq": str(self.equity()), "availBal": str(self.avail())}]}]
        if p == "/api/v5/account/set-leverage":
            assert body["mgnMode"] == "isolated"; return [{}]
        if p == "/api/v5/account/positions":
            x = self.pos.get(q["instId"])
            return [{"mgnMode": "isolated", "pos": str(x["sz"] * x["side"]), "posSide": "net", "avgPx": str(x["avgPx"]),
                     "liqPx": str(x["avgPx"] * (1 - 0.095 * x["side"])), "lever": str(x["lever"]), "upl": str(self.upl(q["instId"]))}] if x else []
        if p == "/api/v5/trade/orders-algo-pending":
            return [g for g in self.algos.get(q["instId"], []) if g["ordType"] == q["ordType"]]
        if p == "/api/v5/trade/cancel-algos":
            for g in body:
                self.algos[g["instId"]] = [x for x in self.algos.get(g["instId"], []) if x["algoId"] != g["algoId"]]
            return [{}]
        if p == "/api/v5/trade/order" and method == "POST":
            assert body["tdMode"] == "isolated" and body["ordType"] == "market"
            inst, side = body["instId"], 1 if body["side"] == "buy" else -1
            assert inst not in self.pos, "程序在已有持仓的合约上又开了仓"
            px, sz = self.m.price(inst), float(body["sz"])
            nt = self.notional(inst, sz, px); lever = 10
            if nt / lever > self.avail():
                self.log["拒单"] += 1
                raise lt.OkxError("/api/v5/trade/order -> 1 All operations failed（51008 Order failed. Insufficient margin）", "1")
            sl = body["attachAlgoOrds"][0]
            assert (float(sl["slTriggerPx"]) - px) * side < 0, "止损单在价格的错误一侧"
            fee = nt * 0.0005; self.cash -= fee
            self.pos[inst] = {"side": side, "sz": sz, "avgPx": px, "margin": nt / lever, "lever": lever, "cTime": t, "fee": fee}
            self.n += 1; oid = str(self.n)
            self.algos[inst] = [{"algoId": "a" + oid, "ordType": "conditional", "slTriggerPx": sl["slTriggerPx"], "sz": body["sz"],
                                 "side": "sell" if side > 0 else "buy", "state": "live"}]
            self.orders[oid] = {"state": "filled", "accFillSz": body["sz"], "avgPx": str(px), "fee": str(-fee)}
            self.scan[inst] = t; self.log["开仓"] += 1
            if self.faults:
                r = self.rnd.random()
                if r < self.crash:
                    self.log["注入-进程被杀"] += 1; raise Crash("下单后进程被杀")
                if r < self.crash + 0.10:
                    self.log["注入-下单后断网"] += 1; raise lt.OkxError("/api/v5/trade/order 网络异常：Read timed out")
            return [{"ordId": oid, "sCode": "0"}]
        if p == "/api/v5/trade/order":
            return [self.orders[q["ordId"]]]
        if p == "/api/v5/trade/order-algo":
            inst = body["instId"]
            assert body.get("reduceOnly") == "true" and inst in self.pos
            self.algos.setdefault(inst, []).append({"algoId": f"b{t}", "ordType": "conditional", "slTriggerPx": body["slTriggerPx"],
                                                    "sz": body["sz"], "side": body["side"], "state": "live"})
            return [{"algoId": f"b{t}", "sCode": "0"}]
        if p == "/api/v5/trade/close-position":
            assert body["mgnMode"] == "isolated"
            if self.faults and self.rnd.random() < 0.2:
                self.log["注入-平仓失败"] += 1; raise lt.OkxError("/api/v5/trade/close-position -> 50013 System busy")
            self._close(body["instId"], self.m.price(body["instId"]), t); self.log["平仓"] += 1
            return [{}]
        if p == "/api/v5/account/positions-history":
            return [h for h in self.hist[::-1] if h["instId"] == q["instId"] and h["_visible"] <= t][:20]
        raise AssertionError(f"假交易所没有模拟这个接口：{method} {p}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    days = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 30
    faults = "--faults" in sys.argv
    opt = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    seed, crash = opt("--seed", 1), opt("--crash", 0.03)
    cfg0 = tr.DEFAULT_CFG
    insts = list(cfg0["instruments"])
    specs = {i: {k: ox.instrument(i)[k] for k in ("ctVal", "lotSz", "minSz", "tickSz", "maxMktSz")} for i in insts}   # 真实合约规格（联网取一次）
    mkt = Market(insts)
    ex = FakeOKX(mkt, specs, faults=faults, seed=seed, crash=crash)

    # 接管外部依赖：行情、OKX 私有接口、时间、费率
    ox.candles = mkt.candles
    ox.ticker = lambda inst: {"last": str(mkt.price(inst))}
    ox.reachable = lambda: True
    ox.instrument = lambda inst: specs[inst]
    real_get = ox.get
    def fake_get(path, params=None, retries=5):
        if path in ("/api/v5/market/candles", "/api/v5/market/history-candles") and params.get("bar") == "1m":
            return mkt.raw_1m(params["instId"], int(params["limit"]), int(params["after"]) if "after" in params else None)
        if path == "/api/v5/public/time":
            return [{"ts": str(NOW[0])}]
        return real_get(path, params, retries)
    ox.get = fake_get
    lt.request = ex.request
    lt.creds = lambda: {"key": "k", "secret": "s", "passphrase": "p", "simulated": True}
    fees.rates = lambda inst: dict(fees.DEFAULT)
    fees.funding_history = lambda inst, since: []
    clock = types.SimpleNamespace(time=lambda: NOW[0] / 1000, sleep=lambda s: None)
    tr.time = clock; lt.time = clock
    tr.now_ms = lambda: NOW[0]
    from html.parser import HTMLParser
    class _V(HTMLParser):
        def __init__(self):
            super().__init__(); self.st = []
        def handle_starttag(self, tag, a):
            self.st.append(tag)
        def handle_endtag(self, tag):
            assert self.st and self.st.pop() == tag, f"推送卡片 HTML 标签不配对：{tag}"
    PUSHES = []
    def fake_alert(cfg, text, card=False):         # 不真的发送，但检查每条推送卡片的 HTML 能被 Telegram 解析
        v = _V(); v.feed(text); assert not v.st, f"推送卡片 HTML 标签没闭合：{v.st}"
        PUSHES.append(text)
    tr.alert = fake_alert
    tr._CFG["alert_trades"] = True
    tr.write_dashboard = lambda con, cfg: None

    tmp = Path(tempfile.mkdtemp(prefix="t15replay_"))
    tr.DB, tr.HEARTBEAT, tr.CFG_PATH, tr.ROOT = tmp / "trader15.db", tmp / "heartbeat.json", tmp / "trader15_config.json", tmp
    tr.CFG_PATH.write_text(json.dumps({**cfg0, "live_trading": True, "alert_trades": True}, ensure_ascii=False), encoding="utf-8")

    end = min(int(mkt.k[(i, "1m")][0][-1]) for i in insts) // M15 * M15
    start = end - days * 86_400_000
    steps = list(range(start + 2000, end, 300_000))        # 每 5 分钟一轮；15m 收盘后 2 秒那一轮会发现新信号
    bad, crashes, t0 = [], 0, time.time()
    for k, t in enumerate(steps):
        ex.advance(t); NOW[0] = t
        try:
            tr.run_once()
        except Crash:
            crashes += 1                                    # “重启”：进程内缓存全部清空
            lt._STATE.clear(); tr._LIVE_ERR_TS.clear(); tr._LIVE_ALERT_TS.clear(); tr._STOP_CHECK_TS.clear()
        con = tr.db()
        live = {r["inst"]: r for r in con.execute("select * from trades where live_status in ('open', 'opening')")}
        for inst, p in ex.pos.items():
            if inst not in live:
                bad.append(f"{pd.Timestamp(t, unit='ms')} {inst} 交易所有持仓，但数据库里没有在跟踪（孤儿仓位）")
            if not ex.algos.get(inst):
                bad.append(f"{pd.Timestamp(t, unit='ms')} {inst} 交易所持仓没有止损单")
        for inst, r in live.items():
            if r["live_status"] == "open" and inst not in ex.pos and not any(h["instId"] == inst and int(h["uTime"]) > t - 600_000 for h in ex.hist):
                bad.append(f"{pd.Timestamp(t, unit='ms')} {inst} 数据库记为持仓中，但交易所没有持仓")
        con.close()
        if k % 1000 == 0:
            print(f"  {k}/{len(steps)} 轮，{time.time() - t0:.0f}s，权益 {ex.equity():,.0f}U，持仓 {len(ex.pos)}", flush=True)

    # 收尾：再跑几轮让在途的平仓记录补齐
    for t in range(end, end + 3600_000, 300_000):
        ex.advance(t); NOW[0] = t
        tr.run_once()
    con = tr.db()
    df = pd.read_sql("select * from trades order by id", con)
    errs = pd.read_sql("select * from signals where decision='错误'", con)
    print(f"\n== 回放 {days} 天（{len(steps)} 轮，{'注入故障' if faults else '无故障'}），用时 {time.time() - t0:.0f}s ==")
    print("假交易所：", ex.log, f"｜进程被杀后重启 {crashes} 次")
    print("模拟交易：", len(df), "笔；OKX 状态：", df.live_status.value_counts().to_dict())
    closed = df[df.live_status == "closed"]
    print(f"OKX 已平 {len(closed)} 笔，实际净利合计 {closed.live_net.sum():+,.0f}U；期末权益 {ex.equity():,.0f}U（期初 100,000U），"
          f"期末交易所持仓 {len(ex.pos)}，数据库 open {int((df.live_status == 'open').sum())}")
    print(f"不变量违反：{len(bad)} 次" + ("".join("\n  " + b for b in bad[:15])))
    print(f"推送 {len(PUSHES)} 条（HTML 格式全部通过检查），示例：\n" + "\n---\n".join(__import__("notify").plain(x) for x in PUSHES[:3]))
    print(f"错误记录 {len(errs)} 条" + "".join(f"\n  {r.inst} {r.detail[:110]}" for r in errs.head(8).itertuples()))
    df[["id", "inst", "side", "point", "entry_ts", "entry_px", "stop", "exit_reason", "live_status", "live_entry", "live_exit", "live_net"]].to_csv(tmp / "trades.csv", index=False)
    print("逐笔明细：", tmp / "trades.csv")
    return df, start, end


def compare_backtest(df, start, end):
    """与回测（bt_v5 的 v5 规则，5m K 线撮合）逐笔比较开仓：同一合约、开仓时间相差 15 分钟以内算一致。"""
    import bt_ttrack as B
    import bt_v5
    import chan15_lab2 as L
    insts = L.live_insts()
    data = {i: d for i, d in L.load(180, insts, "_live").items() if i in dict(insts)}
    data = B.attach(data, B.features(data, 180))
    bt_end = min(d[1][-1] for d in data.values())
    rows = []
    for i, d in data.items():
        t = L.simulate(d, {"sigs": bt_v5.V["v5（两者都加，当前实盘）"]}, t_from=start)
        if len(t):
            rows.append(t.assign(inst=i))
    bt = pd.concat(rows)
    bt = bt[bt.ts < bt_end - 3600_000]
    rp = df[(df.entry_ts >= start) & (df.entry_ts < bt_end - 3600_000)]
    both = sum(any((rp.inst == r.inst) & (rp.entry_ts - r.ts).abs().le(M15)) for r in bt.itertuples())
    only_bt = [f"{r.inst[:4]} {pd.Timestamp(r.ts, unit='ms', tz='UTC').tz_convert('Asia/Shanghai'):%m-%d %H:%M}" for r in bt.itertuples()
               if not any((rp.inst == r.inst) & (rp.entry_ts - r.ts).abs().le(M15))]
    only_rp = [f"{r.inst[:4]} {pd.Timestamp(r.entry_ts, unit='ms', tz='UTC').tz_convert('Asia/Shanghai'):%m-%d %H:%M}" for r in rp.itertuples()
               if not any((bt.inst == r.inst) & (bt.ts - r.entry_ts).abs().le(M15))]
    print(f"\n== 与回测逐笔比较（{len(bt)} 笔回测 / {len(rp)} 笔回放）==\n一致 {both} 笔；只在回测里 {len(only_bt)} 笔 {only_bt[:10]}；"
          f"只在回放里 {len(only_rp)} 笔 {only_rp[:10]}")


if __name__ == "__main__":
    df, start, end = main()
    if "--no-compare" not in sys.argv:
        compare_backtest(df, start, end)
