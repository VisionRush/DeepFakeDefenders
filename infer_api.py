import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import sys
import os
import json
from  main_infer import INFER_API


infer_api = INFER_API()

# 创建FastAPI实例
app = FastAPI()


class inputModel(BaseModel):
    img_path: str = Field(..., description="image path", examples=[""])

# 调用模型接口, post请求
@app.post("/inter_api")
def inter_api(input_model: inputModel):
    img_path = input_model.img_path
    infer_api = INFER_API()
    score = infer_api.test(img_path)
    return  score


# 运行
if __name__ == '__main__':
    uvicorn.run(app='infer_api:app',
                host='0.0.0.0',
                port=10005,
                reload=False,
                workers=1
                )
