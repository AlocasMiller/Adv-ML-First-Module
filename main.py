import fire
from model import ml_model
class CLI(object):

    def __init__(self):
        self._model = ml_model()

    def train(self, dataset_path, model_type):
        return self._model.train(dataset_path, model_type)

    def predict(self, dataset_path, model_type):
        return self._model.predict(dataset_path, model_type)


# if __name__ == "__main__":
    # fire.Fire(CLI)

