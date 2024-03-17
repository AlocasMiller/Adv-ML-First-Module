1. Miller Igor, 972201
2. Как запустить без докера:
   1. Скачать репозиторий и открыть его в PyCharm(или другой IDE с поддержкой питона) 
   2. Для обучения модели запустить в терминале команду python main.py train train.csv cat для обучения CatBoostClassifier (или forest вместо cat для обучения RandomForestClassifier)
   3. Для предсказания запустить python main.py predict test.csv cat (или forest)
   4. Проверить результат в паgке result. Для cat результат в файле result_cat.csv, для forest - result_forest.csv
3. Как запустить с докером:
   
