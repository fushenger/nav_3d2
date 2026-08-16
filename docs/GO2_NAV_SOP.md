# Go2 导航 SOP

当前未启用避障，只能在清空的测试区域运行，并确保实体急停可用。

## 1. 检查网络

```bash
ip -br link
ping -c 3 192.168.123.161
```

## 2. 启动

```bash
cd /home/arts-inpector/ART_3dnav
source /opt/ros/noetic/setup.bash
source devel/setup.bash

roslaunch nav_bringup bringup_mid360_nav.launch \
  start_go2:=true start_in_auto:=true \
  start_obstacle_processor:=false ignore_obstacles:=true \
  go2_iface:=enp86s0 go2_peer:=192.168.123.161
```

网卡不同时，把 `enp86s0` 换成 `ip -br link` 显示的 Go2 网卡。

## 3. RViz 给点

1. `Fixed Frame` 选择 `map`。
2. 用 `2D Pose Estimate` 设置机器人初始位姿。
3. 选择 `3D Nav Goal`：左键拖动设置位置和朝向；再右键上下拖动设置高度并发送。
4. `TRACKING` 表示导航中，`COMPLETED` 表示完成。

首次只选择同层、2 至 3 米内的目标。

## 4. 停止

正常停止：在启动终端按 `Ctrl+C`。

紧急情况：先按机器狗实体急停，再按 `Ctrl+C`。
