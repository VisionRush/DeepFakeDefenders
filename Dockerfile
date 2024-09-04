# 引入cuda版本
FROM  nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# 设置工作目录
WORKDIR /code

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone
RUN apt-get update -y 
RUN apt-get install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install python3.8 python3-pip curl libgl1 libglib2.0-0 ffmpeg libsm6 libxext6 -y  && apt-get clean &&  rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/pytho3 python3 /usr/bin/python3.8 0
RUN update-alternatives --set python3 /usr/bin/python3.8

# 复制该./requirements.txt文件到工作目录中，安装python依赖库。
ADD ./requirements.txt /code/requirements.txt
RUN pip3 install pip --upgrade  -i https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip3 install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/ && rm -rf `pip3 cache dir`

# 复制模型及代码到工作目录
ADD ./core /code/core
ADD ./dataset /code/dataset
ADD ./model /code/model
ADD ./pre_model /code/pre_model
ADD ./toolkit /code/toolkit
ADD ./infer_api.py /code/infer_api.py
ADD ./main_infer.py /code/main_infer.py
ADD ./main_train.py /code/main_train.py
ADD ./merge.py /code/merge.py
ADD ./main.sh /code/main.sh
ADD ./README.md /code/README.md
ADD ./Dockerfile /code/Dockerfile

#运行python文件
ENTRYPOINT ["python3","infer_api.py"]
