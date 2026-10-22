# 飞鼠工具（Hand Gesture Mouse Control）

通过摄像头识别手部关键点，控制鼠标移动、点击和滚动。基于 Python、MediaPipe Tasks、OpenCV 和 PyAutoGUI。

## 安装与运行

本次升级以 **Windows x64 + Python 3.12** 为验证环境。请在新虚拟环境中安装，避免与旧版 OpenCV 或 TensorFlow 环境混用。

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m app.download_model
python -m app.hand_gesture
```

也可以使用 `python app/hand_gesture.py` 启动。按 **q** 退出窗口；PyAutoGUI 的默认紧急停止机制保留。

模型下载命令从 Google 官方地址获取固定版本的 `hand_landmarker.task`，校验 SHA-256 后保存到 `models/`。模型不随仓库提交；下载成功后可离线运行。程序启动和导入时不会自动下载。已有模型可通过 `python -m app.hand_gesture --model 路径/hand_landmarker.task` 指定。

## 固定依赖

| 依赖 | 版本 | 用途 |
| --- | --- | --- |
| MediaPipe | 1.1.1 | Tasks Hand Landmarker 手部关键点与跟踪 |
| opencv-contrib-python | 5.0.0.93 | 摄像头、颜色转换、关键点绘制和窗口 |
| NumPy | 2.5.4 | 图像数组和鼠标坐标计算 |
| PyAutoGUI | 0.9.54 | 鼠标与键盘操作 |

**只安装一种 OpenCV 包。** MediaPipe 已依赖 `opencv-contrib-python`，因此项目统一使用它，不再另装 `opencv-python`。两者都通过 `import cv2` 使用；本项目需要窗口，不能换为 headless 包。

原依赖文件没有锁定版本，README 中的 0.8.3 / 4.5.1 只是过时说明，无法据此确定原作者实际运行的版本。此次锁定的是直接依赖，不是包含所有间接依赖的完整锁文件。

## 手势

- 移动手腕：移动鼠标。
- 拇指与食指捏合：左键单击。
- 食指与中指靠拢：右键单击。
- 拇指与中指捏合：向下滚动。
- 拇指与无名指捏合：向上滚动。
- 拇指与小指捏合：左键双击。

此次保持原有手势、距离阈值、滚动速度和鼠标平滑算法。参数位于 `app/hand_gesture.py`：`gesture_distance_threshold`、`click_delay`、`smoothing_factor`；摄像头默认编号为 0。

## 依赖更新带来的变化

代码已从停止维护的 `mp.solutions.hands` 迁移到 **Tasks Hand Landmarker**，使用 VIDEO 模式、严格递增的时间戳，并分别配置检测、手部存在及跟踪置信度。连续帧使用跟踪，以减少重复运行手掌检测。旧 Hands 本身也有跟踪，因此这不是相对旧版的性能倍增承诺。

新版 Tasks 为后续异步 LIVE_STREAM、按实际左右手选择控制手和基于世界坐标设计手势提供接口。本次仍同步处理视频帧，未新增这些交互功能。OpenCV 5 带来持续更新的 CPU 优化、NumPy 2 兼容和视频输入维护；摄像头实际帧率、延迟和误识别率需要实机对比，不能只凭版本号推断。

同时移除了未参与实际推理的 TensorFlow、PyTorch、Numba 导入及被覆盖的重复 MouseControl 类，避免安装额外深度学习框架才能启动。模型使用 CPU；没有启用 CUDA。

详细说明和官方来源见 [依赖升级说明](docs/dependency-upgrade.md)。

## 验证

```powershell
python -m unittest discover -s tests -v
python -m pip check
```

测试不打开摄像头或操作真实鼠标。下载模型后，会执行真实模型的连续空白帧推理；未下载时该集成测试会明确跳过。摄像头采集、真实手势和端到端延迟仍需手动验证。
