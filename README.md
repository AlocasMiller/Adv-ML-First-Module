1. Miller Igor, 972201
2. Как запустить без докера:
   1. Скачать репозиторий и открыть его в PyCharm(или другой IDE с поддержкой питона) 
   2. Для обучения модели запустить в терминале команду python main.py train train.csv cat для обучения CatBoostClassifier (или forest вместо cat для обучения RandomForestClassifier)
   3. Для предсказания запустить python main.py predict test.csv cat (или forest)
   4. Проверить результат в паgке result. Для cat результат в файле result_cat.csv, для forest - result_forest.csv
3. Как запустить с докером:
   1. Зайти в ubuntu
   2. Ввести sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
   3. Ввести curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
   4. Ввести echo "deb [signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   5. Ввести sudo apt update
   6. Ввести sudo apt install -y docker-ce docker-ce-cli containerd.io
   7. Ввести sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   8. Ввести sudo chmod +x /usr/local/bin/docker-compose
   9. Ввести sudo docker run hello-world
   10. Ввести docker-compoce up
   11. Если вывело в последней строчке It's worked!!!
4. Чем пользовался:
   python = "^3.10"
   pandas = "^2.2.1"
   random11 = "^0.0.1"
   catboost = "^1.2.3"
   joblib = "^1.3.2"
   fire = "^0.6.0"
   scikit-learn = {extras = ["alldeps"], version = "^1.4.1.post1"}
   
