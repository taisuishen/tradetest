"""
推送消息排版（Telegram 用 HTML 粗体；钉钉 / 企业微信群机器人发纯文本，自动去掉标签）。
多单 🟢、空单 🔴；盈利 ✅、亏损 ❌；止损单成交 🛑、补挂止损 🛡️、故障 ⚠️。每项一行，不挤成一行。
"""
import html
import re
from datetime import datetime, timedelta, timezone

TZ = timezone(timedelta(hours=8))
LINE = "━━━━━━━━━━━━━━"


def esc(v):
    return html.escape(str(v), quote=False)


def px(v):
    """价格：大数带千分位，小数保留有效数字。"""
    if v is None:
        return "-"
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{v:.6g}"


def money(v):
    return f"{float(v):+,.2f}U"


def when(ms):
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M:%S")


def dur(ms):
    m = int(ms // 60_000)
    return f"{m // 1440} 天 {m % 1440 // 60} 小时" if m >= 1440 else f"{m // 60} 小时 {m % 60} 分" if m >= 60 else f"{m} 分钟"


def side_word(side, verb="开"):
    return f"{verb}{'多' if side > 0 else '空'}"


def dot(side):
    return "🟢" if side > 0 else "🔴"


def plain(text):
    """群机器人不支持 HTML：去掉标签、还原转义。"""
    return html.unescape(re.sub(r"</?(b|i|code)>", "", text))


def _card(title, rows, foot=None):
    out = [title, LINE] + [r for r in rows if r]
    if foot:
        out.append(f"<i>{foot}</i>")
    return "\n".join(out)


# ---------------- 模拟交易（策略决策）----------------
def open_card(tid, inst, side, point, entry, stop, risk_u, trend, chan1h, eff, zs_w, chase, weak_tt=None, ts=None):
    rows = [f"🎯 信号：{esc(point)}（15 分钟）",
            f"💵 入场：<b>{px(entry)}</b>",
            f"🛑 止损：{px(stop)}（{(stop - entry) / entry * 100:+.2f}%）",
            f"⚖️ 风险：{risk_u:,.2f}U（模拟 1 倍仓位）",
            f"📊 大方向 {trend:+.2f} · 1H 缠论{esc(chan1h)}",
            f"📐 1H 效率 {eff:.0%} · 中枢 {'-' if zs_w is None else f'{zs_w:.2f}'} ATR · 离买卖点 {chase:.2f} ATR"]
    if weak_tt:
        rows.append(f"🔎 低效率信号 · TradeTrack 1H {weak_tt[0]:+d} / 4H {weak_tt[1]:+d}")
    rows.append(f"🕒 {when(ts)}" if ts else None)
    return _card(f"{dot(side)} <b>{side_word(side)} · {esc(inst)}</b>", rows, f"模拟交易 #{tid}")


def close_card(tid, inst, side, reason, entry, exit_, net, entry_ts, exit_ts):
    pct = (exit_ - entry) / entry * side * 100
    rows = [f"📌 原因：{esc(reason)}",
            f"💵 {px(entry)} → {px(exit_)}（<b>{pct:+.2f}%</b>）",
            f"⏱ 持仓 {dur(exit_ts - entry_ts)}",
            f"💰 模拟净利 <b>{money(net)}</b>（1 倍仓位，已扣手续费）"]
    return _card(f"{'✅' if net > 0 else '❌'} <b>{side_word(side, '平')} · {esc(inst)}</b>", rows, f"模拟交易 #{tid}")


# ---------------- OKX 实际成交 ----------------
def live_open_card(tid, mode, inst, side, sz, avg, notional, lev, stop, note=None):
    rows = [f"📦 数量：{sz:g} 张 · 名义 {notional:,.0f}U",
            f"💵 成交均价：<b>{px(avg)}</b>",
            f"🏦 逐仓 {lev} 倍 · 保证金约 {notional / lev:,.0f}U",
            f"🛡️ 止损单：{px(stop)}（{(stop - avg) / avg * 100:+.2f}%）",
            f"📝 {esc(note)}" if note else None]
    return _card(f"{dot(side)} <b>{esc(mode)} · {side_word(side)}成交</b>\n<b>{esc(inst)}</b>", rows, f"#{tid}")


def live_close_card(tid, mode, inst, side, rec, equity=None):
    pct = (rec["closeAvgPx"] - rec["openAvgPx"]) / rec["openAvgPx"] * side * 100 if rec["openAvgPx"] else 0.0
    rows = [f"💵 {px(rec['openAvgPx'])} → {px(rec['closeAvgPx'])}（{pct:+.2f}%）",
            f"💰 净盈亏：<b>{money(rec['net'])}</b>",
            f"🧾 手续费 {rec['fee']:,.2f}U · 资金费 {rec['funding']:+,.2f}U",
            f"📌 {esc(rec['type'])}" if rec["type"] != "平仓" else None,
            f"🏦 账户权益：{equity:,.0f}U" if equity else None]
    return _card(f"{'✅' if rec['net'] > 0 else '❌'} <b>{esc(mode)} · {side_word(side, '平')}</b>\n<b>{esc(inst)}</b>", rows, f"#{tid}")


def exchange_flat_card(tid, mode, inst, side):
    return _card(f"🛑 <b>{esc(mode)} · 持仓已平</b>\n<b>{esc(inst)}</b>",
                 [f"交易所上的{'多' if side > 0 else '空'}单已平（止损单触发或人工平仓）",
                  "稍后自动补记真实盈亏；模拟交易继续按自己的规则跟踪"], f"#{tid}")


def info_card(icon, title, inst, lines, tid=None):
    return _card(f"{icon} <b>{esc(title)}</b>" + (f"\n<b>{esc(inst)}</b>" if inst else ""),
                 [esc(x) for x in lines], f"#{tid}" if tid else None)
