# 手工分层可通行地图

本流程用于先人工确认可通行区域，再交给 `planning_3d` 生成路径。它不依赖轨迹种子或
自动区域生长，因此可以避开楼梯连接处漏点、墙边误分割等问题。

处理逻辑是：在 CloudCompare 中将每个可通行表面分别圈出并导出，然后对每个文件独立
拟合平面。平台会得到不同高度的平面；楼梯可以按每级踏面分别拟合，或将整段楼梯作为
一个倾斜平面。最终只合并通过拟合检查的真实源点，默认不填补未被雷达观测到的区域；也可
选择将这些已确认点投影到各自平面。

## 1. 圈选区域

打开原始地图，而不是自动提取结果：

```bash
CloudCompare /home/arts-inpector/ART_3dnav/src/data/point_cloud/scans.pcd
```

在 CloudCompare 中为每个可通行表面做一次独立的 `Segment`：

- 平台、房间地面、楼梯踏面各导出为一个 PCD。
- 选区只包含轮子实际会走到的表面，避开墙、扶手、桌腿、悬空物和楼梯侧边。
- 对楼梯，优先每级踏面一个文件，能保留真实的台阶高度；只有确认机器人将其当作连续
  坡面行驶时，才将整段楼梯导出为一个文件并使用 `svd` 拟合。
- 保存到一个新目录，例如：
  `/home/arts-inpector/ART_3dnav/src/data/traversable_cloud/manual_regions/`。

建议的文件布局：

```text
manual_regions/
  floor_1.pcd
  landing_1.pcd
  stair_01.pcd
  stair_02.pcd
  floor_2.pcd
```

CloudCompare 导出的 PCD 可以包含或不包含 intensity 字段；工具会统一转换为规划器需要的
`PointXYZI` 格式。

## 2. 为各高度配置独立拟合

在 `manual_regions` 目录新建 `manual_planes.json`。PCD 路径相对于该 JSON 文件。

```json
{
  "regions": [
    {
      "name": "floor_1",
      "pcd": "floor_1.pcd",
      "fit": "ransac",
      "max_plane_distance": 0.04,
      "min_inlier_ratio": 0.85,
      "max_slope_deg": 10,
      "project_to_plane": true
    },
    {
      "name": "stair_01",
      "pcd": "stair_01.pcd",
      "fit": "ransac",
      "max_plane_distance": 0.035,
      "min_inlier_ratio": 0.80,
      "max_slope_deg": 10,
      "project_to_plane": true
    },
    {
      "name": "stair_band_if_needed",
      "pcd": "stair_band.pcd",
      "fit": "svd",
      "max_plane_distance": 0.16,
      "min_inlier_ratio": 0.80,
      "max_slope_deg": 60,
      "project_to_plane": true
    }
  ]
}
```

`ransac` 适合单个平台或单级踏面，会拒绝偏离该平面的点。`svd` 用所有选择点拟合最小二乘
平面，适合已经人工确认的连续坡面或整段楼梯；它会将踏面近似为斜坡，必须先确认该近似
符合机器人能力。未使用的示例区域可从 JSON 中删除。

`project_to_plane: true` 会将已通过残差检查的点投影到各自平面，去掉点云的小幅高度抖动。
设为 `false` 时仍会做平面验收，但输出保留原始点坐标。

## 3. 生成手工版本 PCD

```bash
cd /home/arts-inpector/ART_3dnav

python3 scripts/map_process_tools.py manual-plane-map \
  --manifest src/data/traversable_cloud/manual_regions/manual_planes.json \
  --output-pcd src/data/traversable_cloud/traversable_areas_manual.pcd \
  --report src/data/traversable_cloud/traversable_areas_manual_report.json
```

输出会逐个报告区域的拟合平面高度、坡度、接受点数和残差检查结果。任一区域的坡度过大或
平面内点比例不足时，整个命令会停止且不覆盖已有结果；回到 CloudCompare 缩小该选区，或
仅在确认合理时增大该区域的 `max_plane_distance`。

报告中每个区域都有 `height_at_region_centroid`，这是该区域中心位置的拟合平面高度，可用来
核对楼层与踏面是否保持正确的相对高度。

## 4. 运行规划器

关闭旧规划器和 RViz 后执行：

```bash
source /opt/ros/noetic/setup.bash && cd /home/arts-inpector/ART_3dnav && source devel/setup.bash && roslaunch planning_3d 3d_planning.launch pcd_path:=/home/arts-inpector/ART_3dnav/src/data/traversable_cloud/traversable_areas_manual.pcd
```

`planning_3d` 会在启动时加载 PCD 并构建 PRM，因此每次重新生成手工地图后都需要重启该
规划器。
