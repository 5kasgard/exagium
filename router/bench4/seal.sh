#!/bin/bash
# bench-4 network seal, applied on the host before any attempt. usage: sudo seal.sh METER_PORT
# Containers on docker0 get no forwarded traffic at all (no DNS, no internet), and on the host they may
# reach only the meter's port. Refused packets are logged with the prefix B4DROP so they can be counted.
set -eu
PORT=$1
iptables -I DOCKER-USER 1 -i docker0 -j REJECT
iptables -I DOCKER-USER 1 -i docker0 -j LOG --log-prefix "B4DROP " --log-level 4
iptables -I INPUT 1 -i docker0 -j REJECT
iptables -I INPUT 1 -i docker0 -j LOG --log-prefix "B4DROP " --log-level 4
iptables -I INPUT 1 -i docker0 -p tcp --dport "$PORT" -j ACCEPT
iptables -S DOCKER-USER; iptables -S INPUT | head -5
# The same for IPv6, in case the runner's Docker ever enables it.
if command -v ip6tables >/dev/null; then
  ip6tables -N DOCKER-USER 2>/dev/null || true
  ip6tables -I DOCKER-USER 1 -i docker0 -j REJECT || true
  ip6tables -I FORWARD 1 -i docker0 -j REJECT || true
  ip6tables -I INPUT 1 -i docker0 -j REJECT || true
fi
