import numpy as np, pandas as pd 
from matplotlib import pyplot as plt

#loading the data
data=pd.read_csv('/kaggle/input/digit-recognizer.csv')

data.head()

data= np.array(data)
m,n = data.shape
np.random.shuffle(data)

data_dev=data[0:1000].T
Y_dev= data_dev[0]
X_dev= data_dev[1:n]

data_train =data[1000:m].T
Y_train= data_train[0]
X_train= data_train[1:n]

def init_params(array_sizes):
    L= len(array_sizes)-1
    W=[]
    b=[]
    for i in range(L):
        W.append(np.random.rand(array_sizes[i+1], array_sizes[i]) -0.5)
        b.append(np.random.rand(array_sizes[i+1], 1)-0.5) 
    return W, b, L

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def ReLU(Z): return np.maximum(Z, 0)
def ReLU_deriv(Z): return Z > 0
def softmax(Z): return np.exp(Z - Z.max(axis=0)) / sum(np.exp(Z))

def forward_prop(W,b,X):
    L=len(W)
    A=[X]
    Z= []
    for l in range(L):
        Z_l = W[l].dot(A[-1]) + b[l]
        Z.append(Z_l)
        A.append (softmax(Z_l) if l == L -1 else ReLU(Z_l))
    return Z, A

def backward_prop(Z, A, W,  Y):
    m, L = Y.size, len(W)
    dW, db = [None]*L, [None]*L
    dZ= A[L]- one_hot(Y)
   
    for l in reversed(range(L)):
        dW[l] = (1/m)*dZ.dot(A[l].T)
        db[l] = (1/m)*np.sum(dZ,axis=1, keepdims=True)
        if l > 0:
            dZ = W[l].T.dot(dZ)*ReLU_deriv(Z[l-1])
    return dW, db

def update_params(array_sizes, W, b, dW, db, alpha):
    for l in range(len(W)):
        W[l] = W[l] - alpha * dW[l]
        b[l] = b[l] - alpha * db[l]   
         
    return W,b
    
def get_predictions():
    pass
def get_accuracy():
    pass
def gradient_decent():
    pass
