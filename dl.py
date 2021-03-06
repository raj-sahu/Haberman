import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import scipy
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("haberman.csv")
df.head()
df=df.rename(columns={'age':'Age','year':'Operation_Year','nodes':'No_Of_Lymph_Nodes','status':'Survival_Status_After_5_years'})
df.head()

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix,f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler as StndSclr

y=df['Survival_Status_After_5_years']
m=df.drop('Survival_Status_After_5_years',axis=1)
x=m.copy()
sclr=StndSclr().fit(x)
x=sclr.transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y,shuffle=True, test_size=0.20, random_state=100)

y_train.value_counts()
y_test.value_counts()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics 
from keras.wrappers.scikit_learn import KerasClassifier



def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    #classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['f1-score'])
    return classifier

    classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 10)
    
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()


modelOutput=model.predict(X_test)
d=confusion_matrix(y_test,modelOutput)
print(d)
print(f1_score(y_test,modelOutput))

import pickle
filename = 'Haberman_model.sav'

pickle.dump([model,sclr], open(filename, 'wb'))


