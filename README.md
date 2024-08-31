### 一、 docker
#### 1. docker构建
    sudo docker build  -t vision-rush-image:1.0.1 --network host .
#### 2. 容器启动
    sudo docker run -d --name  vision_rush_image  --gpus=all  --net host  vision-rush-image:1.0.1

### 二、 训练

#### 1. 更改数据集路径

    将训练所需的训练集txt文件、验证集txt文件以及标签txt文件分别放置在dataset文件夹下，并命名为相同的文件名（dataset下有各个txt示例）

#### 2. 更改超参数
    针对所采用的两个模型，在main_train.py分别需要更改如下参数：
    RepLKNet---cfg.network.name = 'replknet'; cfg.train.batch_size = 16
    ConvNeXt---cfg.network.name = 'convnext'; cfg.train.batch_size = 24

#### 3. 启动训练
    bash main.sh

#### 4. 模型融合
    在merge.py中更改ConvNeXt模型路径以及RepLKNet模型路径，执行python merge.py后获取最终推理测试模型。

### 三、 推理

示例如下，通过post请求接口请求，请求参数为图像路径，响应输出为模型预测的deepfake分数

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import requests
import json
import requests
import json

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
}

url = 'http://ip:10005/inter_api'
image_path = './dataset/val_dataset/51aa9b8d0da890cd1d0c5029e3d89e3c.jpg'
data_map = {'img_path':image_path}
response = requests.post(url, data=json.dumps(data_map), headers=header)
content = response.content
print(json.loads(content))
```
