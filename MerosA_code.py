import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import KFold
import keras 
import matplotlib as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import SGD
import sklearn
from keras.regularizers import l2





# Read Dataset #
dataset = np.loadtxt("mnist_train.csv", delimiter=",", skiprows=(1))
testset = np.loadtxt("mnist_test.csv", delimiter=",", skiprows=(1))

# Preprocessing #


X = dataset[:, 1:len(dataset)]
Y = dataset[:,0]
X_test = testset[:, 1:len(testset)]
Y_test = testset[:,0]

X = X/255.0
X_test = X_test/255.0




Y = to_categorical(Y)
Y_test = to_categorical(Y_test)



# define cnn model
def define_model():
    model = Sequential()

    model.add(Dense(397, activation='relu', input_dim=784,kernel_regularizer=l2(0.1) ))
    model.add(Dense(60, activation='relu', input_dim=397, kernel_regularizer=l2(0.1)))
    model.add(Dense(10, activation='softmax', input_dim=10))
    # compile model
    opt = SGD(lr=0.05, momentum=0.6)
   # model.compile(optimizer=opt, loss='mean_squared_error', metrics=[mse])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
    return model


j=0
# evaluate a model using 5-fold cross-validation
ErrorList=[]
# prepare cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=None)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
    j+=1
    # define model
    model = define_model()
    # fit model
    #es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3 ,  patience=0, verbose=1, mode='auto')
    history = model.fit(X[train_ix], Y[train_ix],validation_data=(X[test_ix], Y[test_ix]), epochs=50, verbose=1)
    
    if j==1:
        mse_training = np
        mse_training = np.array([history.history['categorical_crossentropy']])
        mse_testing = np.array([history.history['val_categorical_crossentropy']])
        
    else:
        mse_training = np.concatenate((mse_training,np.array([history.history['categorical_crossentropy']])),axis=0)
        mse_testing = np.concatenate((mse_testing,np.array([history.history['val_categorical_crossentropy']])),axis=0)
    
    
    scores = model.evaluate(X_test, Y_test, verbose=0)
    ErrorList.append(scores[0])
    print( " Evaluation error:", scores[0])
    print("train", train_ix, "test", test_ix)
    
print("mean Error: ", np.mean(ErrorList))
pyplot.title('CE loss')
pyplot.plot(np.mean(mse_training,axis=0),label='train')
#pyplot.plot(np.mean(mse_testing,axis=0),label='validation')








