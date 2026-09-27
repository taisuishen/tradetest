"""
看门狗：每 10 分钟检查一次常驻交易进程（trader15.py loop）是否还活着。

- 心跳（heartbeat.json）超过 5 分钟没更新 → 判定进程已停止或卡死，自动恢复：
    Linux（root）：systemctl restart paper-trader.service（以普通用户身份运行，避免文件属主变成 root）
    Windows：结束卡死的旧进程（核对命令行确实是 trader15.py），再后台拉起新的常驻进程
  然后推送告警（如果配置了）
- 最近几轮一直失败（多为网络问题）→ 只记录，告警由 trader15 自己在连续失败时发送
- 看板服务没在运行 → 重启（Linux：paper-dashboard.service；Windows：本机 127.0.0.1:8080）

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

STALE_MIN = 5
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
        r = systemctl("restart", "paper-trader.service")
        if r is not None:
            wlog(f"重启 paper-trader.service：{'成功' if r.returncode == 0 else r.stderr.strip()}")
        elif os.name == "nt":
            kill_windows_trader(hb.get("pid"))
            start_windows_loop()
        else:
            subprocess.Popen([sys.executable, str(ROOT / "trader15.py"), "loop"], cwd=ROOT, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wlog("已后台拉起 trader15.py loop")
        time.sleep(20)
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
    if os.name == "nt":
        ensure_windows_dashboard(cfg.get("dashboard_port", 8080))


def _pyw():
    pyw = Path(sys.executable).with_name("pythonw.exe")
    return str(pyw if pyw.exists() else sys.executable)


def kill_windows_trader(pid):
    """只结束命令行里确实包含 trader15.py 的那个进程，避免误杀。"""
    if not pid:
        return
    ps = (f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}'; "
          "if ($p -and $p.CommandLine -match 'trader15.py') { Stop-Process -Id $p.ProcessId -Force; 'killed' }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
    if "killed" in r.stdout:
        wlog(f"已结束卡死的旧进程 PID {pid}")


def start_windows_loop():
    subprocess.Popen([_pyw(), str(ROOT / "trader15.py"), "loop"], cwd=ROOT, creationflags=0x00000008 | 0x00000200,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wlog("已后台拉起 trader15.py loop")


def ensure_windows_dashboard(port):
    """Windows：本机看板（只监听 127.0.0.1）没在运行就后台拉起。"""
    import socket
    with socket.socket() as s:
        s.settimeout(2)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return
    subprocess.Popen([_pyw(), "-m", "http.server", str(port), "--bind", "127.0.0.1",
                      "--directory", str(ROOT / "web")], cwd=ROOT, creationflags=0x00000008 | 0x00000200,  # DETACHED | NEW_GROUP
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wlog(f"本机看板未运行，已在 http://127.0.0.1:{port}/ 启动")


if __name__ == "__main__":
    if sys.stdout:                     # pythonw（Windows 后台运行）没有控制台，stdout 为 None
        sys.stdout.reconfigure(encoding="utf-8")
    main()
