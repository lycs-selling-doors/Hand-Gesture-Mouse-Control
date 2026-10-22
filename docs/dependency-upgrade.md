# 依赖升级说明（2026-10-22）

## 版本与兼容改动

原 requirements.txt 中的 MediaPipe、OpenCV 等均没有版本约束，因此没有可还原的原始安装版本；旧 README 标注的 MediaPipe 0.8.3、OpenCV 4.5.1 也与未锁定的安装方式不一致。

本次直接依赖固定为 MediaPipe 1.1.1、opencv-contrib-python 5.0.0.93、NumPy 2.5.4 和 PyAutoGUI 0.9.54。NumPy 2.5.4 要求 Python >=3.12，当前验证基线为 Windows x64 + Python 3.12。

MediaPipe 的元数据依赖 opencv-contrib-python；OpenCV 官方明确要求四种 cv2 分发包只选一种。因此使用包含基础模块的 contrib 桌面版，并移除 opencv-python，避免两个包覆盖同一个 cv2 命名空间。程序保留 imshow，不能使用 headless 分发版。

新版 MediaPipe 不再提供本项目依赖的旧 solutions API。代码迁移为 HandLandmarker.create_from_options / detect_for_video；关键点仍使用原来的 21 点编号。新版结果通过 hand_landmarks 列表读取，绘图使用 Tasks 的 HAND_CONNECTIONS 配合 cv2.line/circle，不再依赖旧 drawing_utils。

Tasks 需要独立模型。新增显式下载命令，固定 Google 模型版本 1 的 URL，并校验 SHA-256（fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1）。下载或校验失败会保留原文件；正常启动不访问模型下载地址。

## 对这个项目有用的能力

| 能力 | 本次状态 | 对项目的意义与限制 |
| --- | --- | --- |
| Tasks VIDEO 跟踪 | 已接入 | 连续帧尽量复用手部跟踪，减少重复手掌检测；旧 Hands 也有跟踪，不能宣称仅升级就更快。 |
| 独立的手部存在置信度 | 已配置为 0.7 | 跟踪到的手消失或置信度不足时可重新检测，为后续抑制失手误操作提供调节入口；阈值效果需实测。 |
| LIVE_STREAM 异步推理 | API 可用，未接入 | 后续可以让预览与推理解耦；忙碌时可能丢帧，接入需评估点击时序与结果过期问题。 |
| 左右手分类、世界坐标 | API 可返回，当前交互未使用 | 可后续实现指定控制手、按手掌比例归一化捏合距离；它们并非首次出现在新版 MediaPipe。 |
| 新版 Python 构建支持 | 上游 1.1.0 增加 Python 3.13/3.14 构建支持 | 减少后续环境升级障碍；本次只验证整个项目在 Python 3.12 上运行，不把上游支持当作全项目实测。 |
| OpenCV 5 / NumPy 2 | 已安装和验证 | 获得现代数组生态及持续的 CPU/SIMD、视频输入维护；本项目主要使用摄像头、cvtColor 和绘图，不会自动启用 DNN、CUDA 或其他未使用模块。 |

MediaPipe 1.1.1 的版本说明主要涉及 LiteRT 更新和其他任务/平台修复，没有声称手部模型在本项目上变得更准确。本次继续使用官方 Hand Landmarker v1 模型，不宣称精度提升。鼠标移动仍保留原有 0.05 秒间隔及 PyAutoGUI 默认暂停，依赖升级无法直接消除这些应用层延迟。

清理了未实际启用的 TensorFlow/PyTorch/Numba 代码及重复类。原 GPU 检查不会让 MediaPipe 或 pip 版 OpenCV 自动启用 CUDA；此次明确使用 CPU。模型、摄像头和窗口现在按生命周期关闭，异常时也会清理。

## 验证范围

- 在独立 Windows x64 / Python 3.12 环境安装 requirements.txt 并检查依赖一致性。
- 真实加载官方模型，对连续空白视频帧执行推理，确认没有手时返回空结果。
- 回归检查重复/回退时间戳、Tasks 关键点绘图、模型下载完整性，以及采集失败或退出时的资源释放。
- 摄像头、真实手势准确率和端到端延迟仍需要实机手动对比；没有采集本机摄像头或驱动真实鼠标。

## 官方资料

- [MediaPipe PyPI 与依赖](https://pypi.org/project/mediapipe/1.1.1/)
- [MediaPipe 发布说明](https://github.com/google-ai-edge/mediapipe/releases)
- [Hand Landmarker Python API、模式与置信度](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [Hand Landmarker 模型及输出](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index)
- [OpenCV Python 安装规则：只选一种分发包](https://pypi.org/project/opencv-contrib-python/5.0.0.93/)
- [OpenCV 5 新特性及迁移背景](https://github.com/opencv/opencv/wiki/OpenCV-5)
- [NumPy 2.5.4 Python 要求](https://pypi.org/project/numpy/2.5.4/)
