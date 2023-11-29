from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from typing import List
import pandas as pd

from api_models.models import Item, Items
from ml_models.ml_model import MLModel

app = FastAPI()

model = MLModel()



@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return model.predict(Items(objects=[item]))[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return model.predict(Items(objects=items))


@app.post('/predict_items_from_csv_file')
def upload_csv(file: UploadFile) -> FileResponse:
    df = pd.read_csv(file.file)
    model.predictions_to_dataframe(df)

    df.to_csv('results.csv')
    response = FileResponse(path='results.csv', media_type='text/csv', filename='results.csv')
    return response