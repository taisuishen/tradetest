"""
看门狗：每 10 分钟检查一次模拟交易是否还在正常运行。

- 心跳（heartbeat.json）超过 15 分钟没更新 → 判定程序停了：
  Linux 上以 root 运行时先重启 systemd 定时器，然后直接拉起一次 trader15.py run，并推送告警
- 最近运行一直失败（多为网络问题）→ 只记录，告警由 trader15 自己在连续失败时发送
- Linux 上看板服务没在运行 → 重启它

用法：python watchdog.py（由 systemd 定时器或 Windows 计划任务调用）
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import trader15 as t  # noqa: E402

STALE_MIN = 15
WLOG = ROOT / "watchdog.log"


def wlog(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(WLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def systemctl(*args):
    if shutil.which("systemctl") and os.name != "nt" and os.geteuid() == 0:
        return subprocess.run(["systemctl", *args], capture_output=True, text=True, timeout=30)
    return None


def main():
    cfg = t.load_cfg()
    hb = t.read_heartbeat()
    last = hb.get("last_run_ts", 0)
    age = (time.time() * 1000 - last) / 60000 if last else float("inf")
    if age > STALE_MIN:
        wlog(f"心跳已 {age:.0f} 分钟未更新（阈值 {STALE_MIN} 分钟），判定程序停止，开始自动恢复")
        r = systemctl("restart", "paper-trader.timer")
        if r is not None:
            # Linux（root）：通过 systemd 以普通用户身份跑一次，避免以 root 直接运行导致数据库文件属主变成 root
            wlog(f"重启 paper-trader.timer：{'成功' if r.returncode == 0 else r.stderr.strip()}")
            r2 = subprocess.run(["systemctl", "start", "paper-trader.service"], capture_output=True, text=True, timeout=280)
            wlog(f"启动一次 paper-trader.service：{'成功' if r2.returncode == 0 else r2.stderr.strip()}")
        else:
            try:
                p = subprocess.run([sys.executable, str(ROOT / "trader15.py"), "run"], cwd=ROOT, timeout=240,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                wlog(f"已拉起一次 trader15.py（退出码 {p.returncode}）")
            except subprocess.TimeoutExpired:
                wlog("拉起的 trader15.py 超过 240 秒未结束，已放弃等待")
        hb2 = t.read_heartbeat()
        ok = hb2.get("last_run_ts", 0) > last
        t.alert(cfg, f"看门狗：程序已停止 {age:.0f} 分钟，{'已自动恢复' if ok else '自动恢复失败，请人工检查'}")
    else:
        ce = hb.get("consecutive_errors", 0)
        wlog(f"正常：心跳 {age:.1f} 分钟前" + (f"，但最近连续 {ce} 次运行失败：{hb.get('last_error', '')}" if ce else ""))
    r = systemctl("is-active", "paper-dashboard.service")
    if r is not None and r.stdout.strip() != "active":
        r2 = systemctl("restart", "paper-dashboard.service")
        wlog(f"看板服务状态 {r.stdout.strip()}，已重启：{'成功' if r2 and r2.returncode == 0 else '失败'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
