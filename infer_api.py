import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import sys
import os
import json
from main_infer import INFER_API


infer_api = INFER_API()

# create FastAPI instance
app = FastAPI()


class inputModel(BaseModel):
    img_path: str = Field(..., description="image path", examples=[""])

# Call model interface, post request
@app.post("/inter_api")
def inter_api(input_model: inputModel):
    img_path = input_model.img_path
    infer_api = INFER_API()
    score = infer_api.test(img_path)
    return  score


# run
if __name__ == '__main__':
    uvicorn.run(app='infer_api:app',
                host='0.0.0.0',
                port=10005,
                reload=False,
                workers=1
                )
