import pandas as pd
import numpy as np
from random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier

# Импортируем датафреймы
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
targets_df = pd.read_csv('sample_submission.csv')
test_df = pd.merge(test_df, targets_df, how='inner', on='PassengerId')

# объединяем тренировочный и тестовый датафреймы
merged_df = pd.concat([train_df, test_df], ignore_index=True)
dfs = [train_df, test_df, merged_df]
# dfs = [train_df, test_df]


# берем моду для строковых значений
mode_features = ['HomePlanet']
# берем медиану для численных значений
median_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# если пассажир находится в крио сне, то эти параметры = 0
cryo_sleep_depending_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# смотрим где пропуски
# print(merged_df.isna().sum(axis=0))

for df in dfs:
    for i in range(len(df)):
        df.loc[(df['Age'].isna()), 'Age'] = randint(26, 31)
        df.loc[(df['VIP'].isna()), 'VIP'] = False
        df.loc[(df['Destination'].isna()), 'Destination'] = 'TRAPPIST-1e'
        if pd.isnull(df.at[i, 'CryoSleep']):
            if any(df.at[i, feature] > 0 for feature in cryo_sleep_depending_features):
                df.at[i, 'CryoSleep'] = False
            else:
                df.at[i, 'CryoSleep'] = True


    for f in mode_features:
        df[f] = df[f].fillna(df[f].mode().iloc[0])

    for f in median_features:
        df[f] = df[f].fillna(df[f].median())

# проверяем остались ли пропуски
# print("------------------------------")
# print(merged_df.isna().sum(axis=0))

bin_feats = ['CryoSleep', 'VIP', 'Transported']
cat_feats = ['HomePlanet', 'Destination']

# кодируем бинарные признаки
for f in bin_feats:
    map_dict = {value: i for i, value in enumerate(set(merged_df[f]))}
    for df in dfs:
        df[f] = df[f].map(map_dict)

# кодируем категориальные признаки ONE HOT ENCODING'ом
for f in cat_feats:
    values = set(train_df[f])
    for v in values:
        for df in dfs:
            df[f + '_' + v] = df[f] == v
    train_df = train_df.drop(columns=f)
    test_df = test_df.drop(columns=f)

# выделяем не нужные нам признаки
uselessFeatures = ['PassengerId', 'Cabin', 'Name']
target = 'Transported'

# создаем тренировочные датафреймы
X_train = train_df.drop(columns=target).drop(columns=uselessFeatures).values
y_train = train_df[target].values

# создаем тестовые датафреймы
X_test = test_df.drop(columns=target).drop(columns=uselessFeatures).values
y_test = test_df[target].values

# создаем модель для
rf_model = RandomForestClassifier(max_depth=7)
scores_rf = cross_val_score(rf_model, X_train, y_train, cv=10)
scores_rf_mean = np.mean(scores_rf)
print("rf: {}", scores_rf_mean)

ct_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, logging_level='Silent')
scores_ct = cross_val_score(ct_model, X_train, y_train, cv=10)
scores_ct_mean = np.mean(scores_ct)
print("ct: {}", scores_ct_mean)

# как протестить
rf_model = RandomForestClassifier(max_depth=7)
accurates = []
rf_model.fit(X_train, y_train)
pred = rf_model.predict(X_test)
pred = pred.astype(bool)
output = pd.DataFrame({'PassengerId': targets_df['PassengerId'],'Transported': pred})
output.to_csv('predictions.csv', index=False)