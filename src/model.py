from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pickle


class BaselineModel:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path


    def train(self,X, df_train):
       

        model = SGDClassifier()
        model.fit(X, df_train)

        with open(self.model_file_path, 'wb') as model_file:
            pickle.dump(model, model_file)

    def predict(self, df_test):
        with open(self.model_file_path, 'rb') as model_file:
            model: tree.DecisionTreeRegressor = pickle.load(model_file)
                
        tfidf = tfidfvectorizer.transform(df_test.sequence)
        
        return model.predict(tfidf)
