"""
OKX 下单前自检（在绑定了 IP 白名单的服务器上运行）。

  python okx_live_check.py             只读检查：Key 能否登录、账户模式、USDT 余额、每个品种的下单张数，并设置逐仓杠杆
  python okx_live_check.py roundtrip [合约] [long|short]
                                       另外在【模拟盘】上用最小数量完整走一遍：开仓 + 附带止损 → 核对止损单 → 平仓 → 撤单 → 读仓位历史
                                       （默认 DOGE-USDT-SWAP 多单；只允许模拟盘，Key 不是模拟盘会直接拒绝）
"""
import sys
import time

import okx_trade as lt
import trader15


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = trader15.load_cfg()
    c = lt.creds()
    if not c:
        print("没有找到 API Key：请填写 okx_api.json（见 okx_api.example.json）"); return 1
    print(f"Key：{'模拟盘' if c['simulated'] else '【实盘】'}")
    a = lt.account()
    print(f"账户等级 acctLv={a['acctLv']}（1 简单 2 单币种保证金 3 跨币种 4 组合）｜持仓模式 {a['posMode']}")
    lt.check_account()
    b = lt.balance_usdt()
    print(f"USDT 权益 {b['eq']:.2f}，可用 {b['availBal']:.2f}")
    lev = cfg["live_leverage"]; k = cfg["live_size_factor"]; frac = cfg.get("live_equity_frac", 1.0)
    by_eq = cfg.get("live_sizing", "equity") == "equity"
    total = 0.0
    print(f"\n逐仓 {lev} 倍，" + (f"每笔名义价值 = 权益 {b['eq']:,.0f}U × {frac:g}：" if by_eq else f"下单数量 = 模拟仓位 × {k:g}："))
    for inst, ic in cfg["instruments"].items():
        s = lt.spec(inst)
        px = float(trader15.ox.ticker(inst)["last"])
        qty = b["eq"] * frac / px if by_eq else ic["unit"] * k
        n = lt.contracts(inst, qty)
        notional = float(n * s["ctVal"]) * px
        total += notional
        time.sleep(0.5)              # 设置杠杆接口限频较严，逐个慢慢设
        try:
            lt.ensure_leverage(inst, lev); lv = f"已设为逐仓 {lev} 倍"
        except Exception as e:
            lv = f"设置杠杆失败：{e}"
        print(f"  {inst:16s} {qty:g} 个币 → {lt.num(n)} 张（每张 {s['ctVal']}，最小 {s['minSz']}）≈ {notional:,.0f}U，保证金约 {notional / lev:,.0f}U｜{lv}")
    per = total / len(cfg["instruments"]) / lev
    print(f"\n全部品种同时持仓：名义 {total:,.0f}U（权益的 {total / max(b['eq'], 1):.1f} 倍），逐仓保证金约 {total / lev:,.0f}U（可用 {b['availBal']:,.0f}U）")
    if total / lev > b["availBal"]:
        print(f"  ⚠️ 可用余额最多够同时开 {int(b['availBal'] // per)} 笔，再开的单子会因保证金不足不下单（模拟交易照常）")
    if len(sys.argv) > 1 and sys.argv[1] == "roundtrip":
        if not c["simulated"]:
            print("\nroundtrip 只允许在模拟盘上运行（okx_api.json 里 simulated 需为 true），已拒绝"); return 1
        roundtrip(sys.argv[2] if len(sys.argv) > 2 else "DOGE-USDT-SWAP", lev,
                  -1 if len(sys.argv) > 3 and sys.argv[3] == "short" else 1)
    return 0


def roundtrip(inst, lev, side=1):
    d = "多" if side > 0 else "空"
    print(f"\n== 模拟盘完整流程测试：{inst} {d}单 ==")
    if lt.position(inst):
        print("该合约已有逐仓仓位，为免干扰，不做测试"); return
    s = lt.spec(inst)
    px = float(trader15.ox.ticker(inst)["last"])
    qty = float(s["minSz"] * s["ctVal"])
    t0 = int(time.time() * 1000)
    r = lt.open_position(inst, side, qty, px * (1 - 0.03 * side), lev, 0)
    print(f"1. 开{d} {r['sz']:g} 张 @{r['avgPx']}，手续费 {r['fee']:.4f}U，止损 {r['stop']}{'；' + r['note'] if r.get('note') else ''}")
    p = lt.position(inst, side)
    print(f"2. 交易所持仓：{p}")
    st = lt.pending_stops(inst)
    print(f"3. 挂着的止损单：{[(x.get('ordType'), x.get('slTriggerPx'), x.get('sz'), x.get('state')) for x in st]}")
    time.sleep(2)
    print(f"4. 平仓：{'已下平仓单' if lt.close_position(inst, side, 0) else '已无持仓'}；剩余止损单 {len(lt.pending_stops(inst))} 张；持仓 {lt.position(inst)}")
    for i in range(10):
        rec = lt.closed_record(inst, t0, side)
        if rec:
            print(f"5. 仓位历史：{rec}"); break
        time.sleep(1)
    else:
        print("5. 10 秒内仓位历史还没出来（稍后在 OKX 页面核对）")
    print("完整流程测试通过" if rec and not lt.position(inst) and not lt.pending_stops(inst) else "⚠️ 有步骤不符合预期，请把以上输出发给我")


if __name__ == "__main__":
    sys.exit(main())
