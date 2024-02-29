import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import pickle

dataset  = pd.read_csv('cropdata.csv')

dataset_X = dataset.iloc[:,[0,1,2,3,4,5,6]].values
dataset_y = dataset.iloc[:,7].values


dataset_X


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
dataset_scaled=sc.fit_transform(dataset_X)


dataset_scaled = pd.DataFrame(dataset_scaled)


X = dataset_scaled
Y = dataset_y

X

Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42 , stratify=dataset['label'])


from sklearn.naive_bayes  import GaussianNB
NaiveBayes = GaussianNB()


NaiveBayes.fit(X_train,Y_train)


NaiveBayes.score(X_test,Y_test)

Y_pred = NaiveBayes.predict(X_test)


pickle.dump(NaiveBayes,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict(sc.transform(np.array([[106,18, 70, 23.603016, 60.3, 6.7, 140.91]]))))