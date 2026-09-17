#!/bin/bash
# 启动 roboticsservice，强制绑定到指定网卡/IP（面向 PICO 的那块）。
#
# 背景（Session B，2026-08-12 踩出来的坑）：
#   RoboticsServiceProcess 默认会自己挑"第一个全局 IPv4 接口"来广播/监听设备
#   TCP 端口（63901），这台机器有多网卡（一块接机器人、一块接 PICO）时经常选错，
#   选错的表现是：App 填对了 PC IP 依然 "TCP connection failed"，
#   而 `ss -tlnp` 会看到 63901 根本没在监听，或者监听在错误的网卡上。
#   /opt/apps/roboticsservice/libPicoNetworkOverride.so 是解决这个问题的 LD_PRELOAD
#   补丁（读 XENSEVR_PICO_IP 环境变量，强制这两处用指定 IP），直接用它，
#   不要指望默认的 runService.sh。
#
# 用法：
#   ./run_pico_service.sh <面向PICO那块网卡的IP>
#   例：./run_pico_service.sh 192.168.43.7
#
# PICO App 里"Enter PC service's IP"要填同一个 IP。

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "用法: $0 <PC 面向 PICO 那块网卡的 IP>" >&2
    echo "先跑: ip -4 addr show   看哪块网卡和 PICO 同网段" >&2
    exit 1
fi

PICO_IP="$1"
DIR=/opt/apps/roboticsservice

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$DIR:$DIR/lib:$DIR/SDK/x64"
export QT_PLUGIN_PATH="$DIR/plugins/:${QT_PLUGIN_PATH:-}"
export QT_QML_PATH="$DIR/qml/:${QT_QML_PATH:-}"
export XENSEVR_PICO_IP="$PICO_IP"
export LD_PRELOAD="$DIR/libPicoNetworkOverride.so"

echo "XENSEVR_PICO_IP=$PICO_IP"
cd "$DIR"
exec ./RoboticsServiceProcess
