import pandas as pd
import numpy as np
from random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
import joblib
import os

def data_preparation(dataset):

    # Импортируем датафрейм
    if (dataset == "test.csv"):
        test_df = pd.read_csv(dataset)
        targets_df = pd.read_csv('sample_submission.csv')
        train_df = pd.merge(test_df, targets_df, how='inner', on='PassengerId')
    else:
        train_df = pd.read_csv(dataset)

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

    return train_df

class ml_model:
    def train(self, dataset, model_type):

        train_df = data_preparation(dataset)

        # выделяем не нужные нам признаки
        uselessFeatures = ['PassengerId', 'Cabin', 'Name']
        target = 'Transported'

        # создаем тренировочные датафреймы
        X_train = train_df.drop(columns=target).drop(columns=uselessFeatures).values
        y_train = train_df[target].values

        if model_type == "cat":
            cat_model = CatBoostClassifier(iterations=100, learning_rate=0.01, depth=6, logging_level='Silent')
            cat_model.fit(X_train, y_train)
            cat_model.save_model("model/catboost_model.bin")
            print("CatBoost is learned")


        if model_type == "forest":
            rf_model = RandomForestClassifier(max_depth=7)
            rf_model.fit(X_train, y_train)
            joblib.dump(rf_model, "model/random_forest_model.joblib")
            print("RandomForest is learned")

    def predict(self, dataset, model_type):
        targets_df = pd.read_csv('sample_submission.csv')
        test_df = data_preparation(dataset)
        # выделяем не нужные нам признаки
        uselessFeatures = ['PassengerId', 'Cabin', 'Name']
        # выделяем нужный нам признак
        target = 'Transported'

        X_test = test_df.drop(columns=target).drop(columns=uselessFeatures).values
        y_test = test_df[target].values

        if model_type == "cat":
            cat_model = CatBoostClassifier().load_model("model/catboost_model.bin")
            predict = cat_model.predict(X_test).astype(bool)
            output = pd.DataFrame({'PassengerId': targets_df['PassengerId'], 'Transported': predict})
            output.to_csv('results/result_cat.csv', index=False)
            # scores = cross_val_score(cat_model, X_test, y_test, cv=10)
            # scores_mean = np.mean(scores)
            # print(scores_mean)

        if model_type == "forest":
            rf_model = joblib.load("model/random_forest_model.joblib")
            predict = rf_model.predict(X_test).astype(bool)
            output = pd.DataFrame({'PassengerId': targets_df['PassengerId'], 'Transported': predict})
            output.to_csv('results/result_forest.csv', index=False)
            # scores = cross_val_score(rf_model, X_test, y_test, cv=10)
            # scores_mean = np.mean(scores)
            # print(scores_mean)