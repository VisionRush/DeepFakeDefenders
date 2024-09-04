<h2 align="center"> <a href="">DeepFake Defenders</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

<h5 align="center">
    
<!-- PROJECT SHIELDS -->
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/VisionRush/DeepFakeDefenders/blob/main/LICENSE) 
![GitHub contributors](https://img.shields.io/github/contributors/VisionRush/DeepFakeDefenders)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVisionRush%2FDeepFakeDefenders&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitors&edge_flat=false)](https://hits.seeyoufarm.com)
![GitHub Repo stars](https://img.shields.io/github/stars/VisionRush/DeepFakeDefenders)
[![GitHub issues](https://img.shields.io/github/issues/VisionRush/DeepFakeDefenders?color=critical&label=Issues)](https://github.com/PKU-YuanGroup/MoE-LLaVA/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/VisionRush/DeepFakeDefenders?color=success&label=Issues)](https://github.com/PKU-YuanGroup/MoE-LLaVA/issues?q=is%3Aissue+is%3Aclosed)  <br>

</h5>

💡 We also provide [[中文文档 / CHINESE DOC](README_zh.md)]. We very welcome and appreciate your contributions to this project.


## 📣 News

* **[2024.09.05]**  🔥 We officially released the initial version of Deepfake defenders, and we won the third prize in the deepfake challenge at [[the conference on the bund](https://www.atecup.cn/deepfake)].

## 🚀 Quickly Start

### 1. Deploy in Docker
#### Building

```shell
sudo docker build  -t vision-rush-image:1.0.1 --network host .
```

#### Running

```shell
sudo docker run -d --name  vision_rush_image  --gpus=all  --net host  vision-rush-image:1.0.1
```

### 2. Training from Scratch

#### Modifying the dataset path

Place the training-set **(\*.txt)** file, validation-set **(\*.txt)** file, and label **(\*.txt)** file required for training in the dataset folder and name them with the same file name (there are various txt examples under dataset)

#### Modifying the Hyper-parameters

For the two models (RepLKNet and ConvNeXt) used, the following parameters need to be changed in `main_train.py`:

```python
# For RepLKNet.
cfg.network.name = 'replknet'; cfg.train.batch_size = 16
# For ConvNeXt.
cfg.network.name = 'convnext'; cfg.train.batch_size = 24
```

#### Using the training script

```shell
bash main.sh
```

#### Model Assembling

Replace the ConvNeXt model path and the RepLKNet model path in `merge.py`, and execute `python merge.py` to obtain the final inference test model.

### Inference

The following example uses the **POST** request interface to request the image path as the request parameter, and the response output is the deepfake score predicted by the model.

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



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=VisionRush/DeepFakeDefenders&type=Date)](https://star-history.com/#DeepFakeDefenders/DeepFakeDefenders&Date)
