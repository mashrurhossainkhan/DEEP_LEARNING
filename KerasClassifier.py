# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy

# Function to create model, required for KerasClassifier
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8,kernel_initializer='uniform', activation='relu' ))
    model.add(Dense(8, kernel_initializer= 'uniform' , activation= 'relu' ))
    model.add(Dense(1, kernel_initializer= 'uniform' , activation= 'sigmoid' ))
    # Compile model
    model.compile(loss= "binary_crossentropy" , optimizer= "adam" , metrics=[ "accuracy" ])
    return model
    
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.genfromtxt("/diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[1:,0:8]
Y = dataset[1:,8]

model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())