# darknet_face_with_landmark
### 借鉴 [darknet](https://github.com/AlexeyAB/darknet) 做适量修改，用于人脸检测以及关键点检测


## Installation
##### Clone and install
1. git clone https://github.com/ouyanghuiyu/darknet_face_with_landmark.git
2. 使用scripts/retinaface2yololandmark.py脚本将retinaface的标记文件转为yolo的格式使用
3. 其他编译训练都和原版darknet相同
4. 使用yolo_landmark.py进行测试，更换里面的模型配置文件即可


## 测试
<p align="center"><img src="test_imgs/output/selfie.jpg"\></p>