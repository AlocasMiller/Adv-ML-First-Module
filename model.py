import pandas as pd
import numpy as np
from random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
import joblib

class ml_model:
    def train(self, dataset_path, model_type):
        # Импортируем датафреймы
        train_df = pd.read_csv(dataset_path)

        # берем моду для строковых значений
        mode_features = ['HomePlanet']
        # берем медиану для численных значений
        median_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        # если пассажир находится в крио сне, то эти параметры = 0
        cryo_sleep_depending_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        for i in range(len(train_df)):
            train_df.loc[(train_df['Age'].isna()), 'Age'] = randint(26, 31)
            train_df.loc[(train_df['VIP'].isna()), 'VIP'] = False
            train_df.loc[(train_df['Destination'].isna()), 'Destination'] = 'TRAPPIST-1e'
            if pd.isnull(train_df.at[i, 'CryoSleep']):
                if any(train_df.at[i, feature] > 0 for feature in cryo_sleep_depending_features):
                    train_df.at[i, 'CryoSleep'] = False
                else:
                    train_df.at[i, 'CryoSleep'] = True

        for f in mode_features:
            train_df[f] = train_df[f].fillna(train_df[f].mode().iloc[0])

        for f in median_features:
            train_df[f] = train_df[f].fillna(train_df[f].median())

        bin_feats = ['CryoSleep', 'VIP', 'Transported']
        cat_feats = ['HomePlanet', 'Destination']

        # кодируем бинарные признаки
        for f in bin_feats:
            map_dict = {value: i for i, value in enumerate(set(train_df[f]))}
            train_df[f] = train_df[f].map(map_dict)

        # кодируем категориальные признаки ONE HOT ENCODING'ом
        for f in cat_feats:
            values = set(train_df[f])
            for v in values:
                train_df[f + '_' + v] = train_df[f] == v
            train_df = train_df.drop(columns=f)

        # выделяем не нужные нам признаки
        uselessFeatures = ['PassengerId', 'Cabin', 'Name']
        target = 'Transported'

        # создаем тренировочные датафреймы
        X_train = train_df.drop(columns=target).drop(columns=uselessFeatures).values
        y_train = train_df[target].values

        if model_type == "cat":
            cat_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, logging_level='Silent')
            cat_model_train = cross_val_score(cat_model, X_train, y_train, cv=10)
            cat_model_train.save_model("adv-ML-First-Module/catboost_model.bin")
            print(123)

        if model_type == "forest":
            rf_model = RandomForestClassifier(max_depth=7)
            rf_model_train = cross_val_score(rf_model, X_train, y_train, cv=10)
            joblib.dump(rf_model_train, "adv-ML-First-Module/random_forest_model.joblib")


    def predict(self, model_type, input_data):
        print("qwe")
        # if model_type == "cat":
        #
        #     rf_model = CatBoostClassifier(max_depth=7)
        #     accurates = []
        #     rf_model.fit(X_train, y_train)
        #     pred = rf_model.predict(X_test)
        #     pred = pred.astype(bool)
        #
        # elif model_type == "forest":
        #     rf_model = RandomForestClassifier(max_depth=7)
        #     accurates = []
        #     rf_model.fit(X_train, y_train)
        #     pred = rf_model.predict(X_test)
        #     pred = pred.astype(bool)