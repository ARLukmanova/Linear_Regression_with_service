import pickle
from typing import List

import pandas as pd
import numpy as np

from api_models.models import Items


class MLModel:

    def __init__(self):
        with open("ml_models/Imputer.pickle", "rb") as f:
            self.imp = pickle.load(f)
        with open("ml_models/Encoder.pickle", "rb") as f:
            self.enc = pickle.load(f)
        with open("ml_models/Scaler.pickle", "rb") as f:
            self.scal = pickle.load(f)
        with open("ml_models/Model.pickle", "rb") as f:
            self.model_linear = pickle.load(f)

    def predict(self, items: Items) -> List[float]:
        df = pd.DataFrame([dict(item) for item in items.objects])
        self.predictions_to_dataframe(df)
        result = df['selling_price'].to_list()
        return result

    def predictions_to_dataframe(self, get_df: pd.DataFrame) -> None:
        df = get_df.copy()
        if 'selling_price' in df.columns:
            df.drop('selling_price', axis =1, inplace=True)
        #Приведение к числовым значениям
        df['mileage'] = pd.to_numeric(df['mileage'].replace('kmpl|km/kg', '', regex=True).str.strip()).astype(float)
        df['engine'] = pd.to_numeric(df['engine'].replace('CC', '', regex=True).str.strip()).astype(float)
        df['max_power'] = pd.to_numeric(df['max_power'].replace('bhp', '', regex=True).str.strip()).astype(float)
        df.drop(['torque'], axis=1, inplace=True)

        #Заполнение медианными
        cols = ['mileage', 'engine', 'max_power', 'seats']
        df[cols] = self.imp.transform(df[cols])

        #Новые поля
        df['owner'] = df['owner'].replace(
            {'Test Drive Car': 0, 'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4})

        df['power_per_engine'] = df['max_power'] / df['engine']
        df['brand'] = df['name'].str.split(' ').str.get(0)
        df.drop(['name'], axis=1, inplace=True)

        #OneHotEncoding
        cat_cols = ['seats', 'fuel', 'seller_type', 'transmission', 'brand']
        df_enc = pd.DataFrame(self.enc.transform(df[cat_cols]).toarray(),
                columns=self.enc.get_feature_names_out(cat_cols), dtype=int)
        X= pd.concat([df.drop(columns=cat_cols), df_enc], axis=1)
        X = self.scal.transform(X)

        #Предсказание
        y_pred = self.model_linear.predict(X)
        get_df['selling_price'] = np.exp(y_pred)
