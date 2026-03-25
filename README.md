# 3D实时重定向演示模块

这是一个模块化的3D实时重定向系统，将代码放入dex_retargeting (https://github.com/dexsuite/dex-retargeting) 的dex-retargeting\example目录下即可运行。

## Demo 视频

<p align="center">
  <video controls width="720" playsinline>
    <source src="https://raw.githubusercontent.com/games-liker/Real-time-3D-control-of-a-monocular-camera-robotic-arm/main/demo.mp4" type="video/mp4">
    你的浏览器不支持视频播放，请点击这里下载/查看：demo.mp4
  </video>
</p>

如果 GitHub 没有直接播放视频，你也可以点击下方文件 `demo.mp4` 进行查看。

## 模块结构

```
3d_retargeting/
├── __init__.py          # 包初始化文件，导出主要接口
├── config.py            # 配置和常量（手掌多边形索引、相机配置、机械臂映射）
├── data_structures.py   # 数据结构定义（DetectionResult, DepthResult, IKResult）
├── utils.py             # 工具函数（面积计算、深度映射、组合体信息）
├── smoothers.py         # 平滑器类（RealtimeSmoother3D, AreaDepthSmoother）
├── camera_manager.py    # 多视角相机管理器
├── single_hand_detector.py  # 手部检测器（MediaPipe封装）
├── arm_ik_solver.py    # 机械臂IK求解器
├── producer.py          # 图像采集进程
├── detection.py         # 手部检测进程
├── depth_estimation.py  # 深度估计进程
├── ik_solver.py         # IK求解进程
├── renderer.py          # 渲染进程
├── main.py              # 主函数和入口
└── README.md            # 本文件
```

## 模块说明

### config.py
包含所有配置和常量：
- `PALM_POLYGON_INDICES`: 手掌多边形关节点索引
- `CameraConfig`: 相机配置数据类
- `MULTI_VIEW_CAMERAS`: 预定义的多视角相机配置
- `VIEW_LAYOUTS`: 视角布局预设
- `ARM_HAND_ASSEMBLY_MAP`: 机械臂组合映射表

### data_structures.py
定义进程间通信的数据结构：
- `DetectionResult`: 检测结果（不包含图像）
- `DepthResult`: 基于面积的深度估计结果
- `IKResult`: IK求解结果

### utils.py
工具函数：
- `compute_polygon_area()`: 使用鞋带公式计算多边形面积
- `area_ratio_to_depth()`: 将手掌面积占比映射到深度值
- `get_assembly_info()`: 根据机器人名称和手型获取组合体信息

### smoothers.py
平滑器类：
- `RealtimeSmoother3D`: 3D位置平滑器，用于平滑手腕3D位置
- `AreaDepthSmoother`: 面积深度平滑器，用于平滑面积估计的深度值

### camera_manager.py
多视角相机管理器：
- `MultiViewCameraManager`: 管理多个相机视角的创建、捕获和图像合成

### single_hand_detector.py
手部检测器（MediaPipe封装）：
- `SingleHandDetector`: MediaPipe手部检测的封装类
- `OPERATOR2MANO_RIGHT`: 右手坐标系转换矩阵
- `OPERATOR2MANO_LEFT`: 左手坐标系转换矩阵

### arm_ik_solver.py
机械臂IK求解器：
- `ArmIKSolver`: 机械臂逆运动学求解器类
- `get_end_effector_frame()`: 获取末端执行器坐标系函数

### 进程模块

为了更好的模块化和可维护性，进程函数被分解为独立的文件：

- **producer.py**: 图像采集进程
  - `producer_process()`: 从摄像头读取图像帧

- **detection.py**: 手部检测进程
  - `detection_process()`: 执行MediaPipe手部检测，提取手掌多边形关键点

- **depth_estimation.py**: 深度估计进程
  - `depth_estimation_process()`: 基于手掌面积计算深度

- **ik_solver.py**: IK求解进程
  - `ik_process_3d()`: 执行逆运动学求解，计算机器人关节角度

- **renderer.py**: 渲染进程
  - `render_process_multiview()`: 负责3D场景渲染和机器人可视化

### main.py
主入口文件，包含：
- `main()`: 主函数，负责创建进程和队列，启动整个系统

## 使用方法

### 基本使用

**直接运行（推荐）**
```bash
cd example/3d_retargeting
python main.py
```

## 主要参数说明

### 面积深度估计参数
- `min_area_ratio`: 最小面积占比（手最远时），默认 0.005
- `max_area_ratio`: 最大面积占比（手最近时），默认 0.15
- `min_depth`: 最小深度值（手最近时），默认 0.01
- `max_depth`: 最大深度值（手最远时），默认 0.4

### 平滑参数
- `enable_smoothing`: 是否启用平滑，默认 True
- `smoothing_alpha`: 3D位置平滑系数，默认 0.3（越小越平滑）
- `area_smoothing_alpha`: 面积平滑系数，默认 0.2

### 多视角参数
- `view_layout`: 视角布局，可选 "2x2", "1x4", "2x1", "1x3", "single"，默认 "2x1"
- `show_multiview`: 是否显示多视角窗口，默认 True

### 其他参数
- `robot_name`: 机器人标识符，默认 ability
- `retargeting_type`: 重定向类型，默认 dexpilot
- `hand_type`: 手型（left/right），默认 right
- `camera_path`: 摄像头路径，默认 None（使用默认摄像头）
- `image_scale_factor`: 图像缩放因子，默认 1.0
- `queue_size`: 队列大小，默认 100

## 修改指南

### 修改相机配置
编辑 `config.py` 中的 `MULTI_VIEW_CAMERAS` 字典，添加或修改相机配置。

### 修改深度映射函数
编辑 `utils.py` 中的 `area_ratio_to_depth()` 函数，调整映射逻辑。

### 修改平滑算法
编辑 `smoothers.py` 中的平滑器类，实现自定义平滑算法。

### 修改进程逻辑
每个进程都有独立的文件，可以直接编辑对应的文件：
- 修改图像采集逻辑 → `producer.py`
- 修改手部检测逻辑 → `detection.py`
- 修改深度估计逻辑 → `depth_estimation.py`
- 修改IK求解逻辑 → `ik_solver.py`
- 修改渲染逻辑 → `renderer.py`

### 添加新的数据结构
在 `data_structures.py` 中添加新的数据类。


## 注意事项

1. 确保 `single_hand_detector.py` 和 `arm_ik_solver.py` 都在当前目录中
2. 确保机器人URDF文件路径正确
3. 多视角显示需要足够的GPU资源
4. 按 'q' 键退出程序

## 引用与致谢

本项目的代码实现基于 GitHub 上的 `dexsuite/dex-retargeting`。感谢原作者开源与贡献：
https://github.com/dexsuite/dex-retargeting