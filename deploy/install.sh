#!/usr/bin/env bash
# 在 Linux 服务器（Ubuntu/Debian）上一键部署模拟交易：
#   每 5 分钟运行一次 trader15.py（systemd timer），看板通过 8080 端口提供
# 用法：sudo bash deploy/install.sh [看板端口，默认 8080]
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-8080}"
RUN_USER="${SUDO_USER:-$(whoami)}"

echo "==> 安装目录：$APP_DIR  运行用户：$RUN_USER  看板端口：$PORT"
if command -v apt-get >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq python3 python3-venv python3-pip tzdata >/dev/null
fi

sudo -u "$RUN_USER" python3 -m venv "$APP_DIR/.venv"
sudo -u "$RUN_USER" "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"
sudo -u "$RUN_USER" mkdir -p "$APP_DIR/web"

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
ExecStart=/usr/bin/python3 -m http.server $PORT --directory $APP_DIR/web
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

echo "==> 完成"
echo "   查看定时器：systemctl list-timers 'paper-*'"
echo "   看门狗日志：tail -f $APP_DIR/watchdog.log"
echo "   查看日志：  tail -f $APP_DIR/trader15.log"
echo "   统计：      $APP_DIR/.venv/bin/python $APP_DIR/trader15.py report"
echo "   看板：      http://<服务器IP>:$PORT/   （记得在云厂商安全组放行该端口）"
