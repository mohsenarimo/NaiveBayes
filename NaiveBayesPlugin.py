import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sys

class NaiveBayesPlugin:
    def input(self,filename):
        data = pd.read_csv(filename)
        X = data.drop('Species', axis=1)
        y = data['Species']      
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=0.2)

    def run(self):
        gnb = GaussianNB()
        gnb.fit(self.X_train, self.y_train)
        self.y_pred = gnb.predict(self.X_test)
       
    def output(self,filename):
        pred_df = pd.DataFrame(self.y_pred, columns = ['prediction'])
        pred_df.to_csv(filename,index=False)
        #print(pred_df)
