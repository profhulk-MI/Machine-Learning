import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as plt


# Getting the data
X, y = make_classification(n_samples=50000, n_features=15, n_informative=10, n_redundant=5,
                           n_classes=2, weights=[0.7], class_sep=0.7, random_state=15)

#Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.25, random_state=15)

#Function to initialize weights
def initialize_weights(dim):
    return np.zeros(len(dim)),np.zeros(1)
    
# Sigmoid Function
def sigmoid(z):
    return 1/(1+np.exp(-z))
    
#Choosing the Log-Loss as Loss function
def logloss(y_true,y_pred):
    loss=-np.mean([((i*np.where(j>0,np.log10(j),0))+((1-i)*(np.where((1-j)>0,np.log10(1-j),0)))) for i,j in zip(y_true,y_pred)])
    return loss

# Gradient 
def gradient_dw(x,y,w,b,alpha,N):
    w=np.array(w)
    dw=x*(y-sigmoid((np.dot(w,x)+b)))-alpha*(w/N)
    return dw
    
 def gradient_db(x,y,w,b):
     db=y-sigmoid(np.dot(w,x)+b)
     return db
 
 def train(X_train,y_train,X_test,y_test,epochs,alpha,eta0,N,plot=False):
    w,b=initialize_weights(X_train[0])
    w=np.array(w)
    b=np.array(b)
    loss_epoch_train,loss_epoch_test=[],[]
    for n in tqdm(range(epochs)):
        for i,j in zip(X_train,y_train):
            dw=np.array(gradient_dw(i,j,w,b,alpha,N))
            db=np.array(gradient_db(i,j,w,b))
            w=np.array(w+(eta0*dw))
            b=b+(eta0*db)
        y_pred_train=[sigmoid((np.dot(w,i)+b))[0] for i in X_train]
        loss_epoch_train.append(logloss(y_train,y_pred_train))
        y_pred_test=[sigmoid((np.dot(w,i)+b))[0] for i in X_test]
        loss_epoch_test.append(logloss(y_test,y_pred_test))
        if len(loss_epoch_test)>2 and abs(loss_epoch_test[-2]-loss_epoch_test[-1])<1e-4:
            break
    if True:
        plt.title('Epoch vs Train&Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('LogLoss')
        plt.plot(np.arange(n+1),loss_epoch_train,label='Train Loss')
        plt.plot(np.arange(n+1),loss_epoch_test,label='Test Loss')
        plt.legend()
    return w,b
     
# Training
alpha=0.0001
eta0=0.0001
N=len(X_train)
epochs=50
w,b=train(X_train,y_train,X_test,y_test,epochs,alpha,eta0,N,True)
