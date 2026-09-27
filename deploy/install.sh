#!/usr/bin/env bash
# 在 Linux 服务器上一键部署模拟交易（支持 CentOS 7/8/9、Rocky/Alma、Ubuntu/Debian）：
#   每 5 分钟运行一次 trader15.py（systemd timer），看板通过 8080 端口提供，看门狗每 10 分钟检活
# 用法：sudo bash deploy/install.sh [看板端口，默认 8080]
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-8080}"
RUN_USER="${SUDO_USER:-$(whoami)}"

echo "==> 安装目录：$APP_DIR  运行用户：$RUN_USER  看板端口：$PORT"

# 找一个 ≥3.9 的 Python（pandas 2.x 需要）
PY=""
find_py() {
  for c in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys, venv; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
      PY="$(command -v "$c")"; return 0
    fi
  done
  return 1
}

if command -v apt-get >/dev/null 2>&1; then
  echo "==> Debian/Ubuntu：安装依赖"
  apt-get update -qq
  apt-get install -y -qq python3 python3-venv python3-pip tzdata curl >/dev/null
elif command -v dnf >/dev/null 2>&1; then
  echo "==> CentOS 8/9 / Rocky / Alma：安装依赖"
  dnf install -y -q tzdata curl >/dev/null || true
  find_py || dnf install -y -q python3.11 >/dev/null 2>&1 || dnf install -y -q python39 >/dev/null 2>&1 || true
elif command -v yum >/dev/null 2>&1; then
  echo "==> CentOS 7：安装依赖（官方源已停服，失败会自动改用独立 Python）"
  yum install -y -q tzdata curl >/dev/null 2>&1 || true
fi

if ! find_py; then
  # 系统源里没有新版 Python（典型是 CentOS 7）：用 uv 下载独立的 Python 3.11，装到 /opt 下所有用户可读
  echo "==> 系统没有 Python ≥3.9，用 uv 安装独立的 Python 3.11"
  export UV_INSTALL_DIR=/usr/local/bin UV_PYTHON_INSTALL_DIR=/opt/uv-python
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  /usr/local/bin/uv python install 3.11
  PY="$(/usr/local/bin/uv python find 3.11)"
  chmod -R a+rX /opt/uv-python
fi
echo "==> 使用 Python：$PY（$("$PY" --version)）"

sudo -u "$RUN_USER" "$PY" -m venv "$APP_DIR/.venv"
sudo -u "$RUN_USER" "$APP_DIR/.venv/bin/pip" install -q --upgrade pip
sudo -u "$RUN_USER" "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"
sudo -u "$RUN_USER" mkdir -p "$APP_DIR/web"

if [ "${SKIP_SYSTEMD:-0}" = "1" ]; then   # 仅用于在容器里测试 Python 环境部分
  "$APP_DIR/.venv/bin/python" -c "import pandas, numpy, requests; print('依赖安装正常：pandas', pandas.__version__)"
  "$APP_DIR/.venv/bin/python" "$APP_DIR/trader15.py" run
  exit 0
fi

cat > /etc/systemd/system/paper-trader.service <<EOF
[Unit]
Description=15m chan paper trader (one run)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=$RUN_USER
WorkingDirectory=$APP_DIR
Environment=TZ=Asia/Shanghai
ExecStart=$APP_DIR/.venv/bin/python $APP_DIR/trader15.py run
TimeoutStartSec=240
EOF

cat > /etc/systemd/system/paper-trader.timer <<EOF
[Unit]
Description=Run 15m chan paper trader every 5 minutes

[Timer]
OnCalendar=*:0/5
Persistent=true
AccuracySec=5s

[Install]
WantedBy=timers.target
EOF

# 看板：只对外提供 web/ 目录（数据库和配置不暴露）
cat > /etc/systemd/system/paper-dashboard.service <<EOF
[Unit]
Description=Paper trader dashboard
After=network.target

[Service]
User=$RUN_USER
WorkingDirectory=$APP_DIR/web
ExecStart=$APP_DIR/.venv/bin/python -m http.server $PORT --directory $APP_DIR/web
RestartSec=5
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 看门狗：每 10 分钟检查心跳，超过 15 分钟没运行就重启定时器并拉起一次；看板服务挂了也会重启（以 root 运行）
cat > /etc/systemd/system/paper-watchdog.service <<EOF
[Unit]
Description=Paper trader watchdog
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=$APP_DIR
Environment=TZ=Asia/Shanghai
ExecStart=$APP_DIR/.venv/bin/python $APP_DIR/watchdog.py
TimeoutStartSec=300
EOF

cat > /etc/systemd/system/paper-watchdog.timer <<EOF
[Unit]
Description=Run paper trader watchdog every 10 minutes

[Timer]
OnBootSec=3min
OnUnitActiveSec=10min
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now paper-trader.timer paper-dashboard.service paper-watchdog.timer
systemctl start paper-trader.service || true

# CentOS 默认开着 firewalld：放行看板端口（云厂商安全组仍需另外放行）
if systemctl is-active --quiet firewalld 2>/dev/null; then
  firewall-cmd --permanent --add-port="$PORT/tcp" >/dev/null && firewall-cmd --reload >/dev/null && echo "==> firewalld 已放行 $PORT/tcp"
fi

echo "==> 完成"
echo "   查看定时器：systemctl list-timers 'paper-*'"
echo "   看门狗日志：tail -f $APP_DIR/watchdog.log"
echo "   查看日志：  tail -f $APP_DIR/trader15.log"
echo "   统计：      $APP_DIR/.venv/bin/python $APP_DIR/trader15.py report"
echo "   看板：      http://<服务器IP>:$PORT/   （记得在云厂商安全组放行该端口）"
