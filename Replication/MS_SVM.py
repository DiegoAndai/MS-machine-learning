import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from joblib import dump, load
from skfeature.function.similarity_based import fisher_score

class multipleModelTrainer:

    def __init__(self, X, y):

        self.models_to_train = 100
        self.X = X
        self.y = y
        self.fisher_score_features = fisher_score(self.X, self.y)


class singleTrainer:

    def __init__ (self, model_id, **model_parameters):
        self.model_id = model_id
        self.model = SVC(**model_parameters)

    def get_feature_importance(self):
        coef = self.model.coef_.ravel()
        return np.argsort(coef)[::-1]

    def save(self):
        with open('saved_models/{}.joblib'.format(self.model_id), "wb") as file_dump:
            dump(self.model, file_dump)

    def load(self):
        with open('saved_models/{}.joblib'.format(self.model_id), "rb") as file_load:
            self.model = load(file_load)



if __name__ == "__main__":

    
