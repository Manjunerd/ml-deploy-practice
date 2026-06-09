import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Salary Prediction API is running"}

@app.post("/predict")
def predict(data: dict):

    age = data["age"]

    result = model.predict([[age]])

    return {
        "prediction": int(result[0])
    }


if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1',port=8000)