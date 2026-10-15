# 授权与运行模型

项目可授权的原创代码采用根目录MIT许可证。该许可不为第三方模型权重、恢复数据或其他第三方材料重新授权。

- `models/local/`：当前修复与分割所需的本地研究资产，包含模型参数、图结构及模板；仅按已有使用权在本地研究与使用，不在这里声明其可公开分发。
- `models/detection/face_detection_yunet_2023mar.onnx`：来自OpenCV Zoo，对应许可保存在同目录 `YuNet-LICENSE.txt`。官方来源：[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。
- `models/detection/face_landmarker.task`：Google MediaPipe Face Landmarker模型包，来源及模型卡见[官方模型页](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index#models)。不把项目代码的MIT许可套用到模型数据。
- PySide6、PyTorch、MediaPipe、OpenCV、NumPy、Pillow等依赖分别适用各自许可证。
