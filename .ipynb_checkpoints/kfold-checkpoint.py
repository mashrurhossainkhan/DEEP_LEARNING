# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.genfromtxt("/kaggle/input/pima-indians-diabetes-database/diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[1:,0:8]
Y = dataset[1:,8]
import numpy as np

# Check for NaN values in Y
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(12, input_dim=8,kernel_initializer='uniform', activation='relu' ))
    model.add(Dense(8, kernel_initializer= 'uniform' , activation= 'relu' ))
    model.add(Dense(1, kernel_initializer= 'uniform' , activation= 'sigmoid' ))
    # Compile model
    model.compile(loss= "binary_crossentropy" , optimizer= "adam" , metrics=[ "accuracy" ])
    # Fit the model
    model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))