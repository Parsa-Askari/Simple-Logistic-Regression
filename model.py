import numpy as np
from sklearn.model_selection import train_test_split
def Softmax(z):
    return np.exp(z)/np.sum(np.exp(z),axis=0)
def one_hot_encoding(y,k,n):
    one_hot=np.zeros((k,n))
    one_hot[y,np.arange(y.size)]=1
    return one_hot
class LogisticRegression:
    def __init__(self):
        pass
    def Train(self,X,Y,class_num,with_valid=False,batch_size=8,epochs=10,lr=0.000001):
        """
        n = number of samples
        m = number of features
        k = number of classes
        """
        n,m=X.shape
        k=class_num
        self.W=np.random.rand(class_num,m+1) # shape : (k, m)
        X=np.concatenate([X,np.ones((X.shape[0],1))],axis=1)
        self.loss_history={"train":[],"val":[]}
        for ep in range(epochs):
            trainloss_list=[]
            validloss_list=[]
            Data=np.concatenate((X,Y.reshape((-1,1))),axis=1)
            np.random.shuffle(Data)
            Y=Data[:,-1]
            X=Data[:,:-1]
            Y=Y.astype("int")
            if(with_valid==True):
                x_train,x_val,y_train,y_val=train_test_split(X,Y)
                n=len(y_train)
            else:
                x_train=X
                y_train=Y
            for i in range(0,n,batch_size):
                x_batch=x_train[i:i+batch_size,:] # shape : (n, m)
                y_batch=y_train[i:i+batch_size] # shape : (n)
                batch_size=min(len(y_batch),batch_size)
                y_hat=self.Predict(x_batch) # shape : (k , n)
                y_enc=one_hot_encoding(y_batch,k,batch_size)
                train_loss=self.Loss(y_batch,y_hat)
                if(with_valid==True):
                    y_val_hat=self.Predict(x_val) # shape : (k , n)
                    valid_loss=self.Loss(y_val,y_val_hat)
                    validloss_list.append(valid_loss)
                    
                trainloss_list.append(train_loss)

                loss_z_grad=y_hat-y_enc # shape : (k , n)

                W_grad=np.dot(loss_z_grad,x_batch)

                self.W -=lr*W_grad

            self.loss_history["train"].append(np.mean(trainloss_list))
            self.loss_history["val"].append(np.mean(validloss_list))
            print("="*50)
            print("train loss : ",self.loss_history["train"][-1])
            if(with_valid==True):
                print("valid loss : ",self.loss_history["val"][-1])
    def Predict(self,x_batch):
        z=np.dot(self.W,x_batch.T) # shape : (k , n)
        y_hat=Softmax(z) 
        return y_hat
    def Loss(self,y,y_hat):
        y_pred=y_hat[y,np.arange(y.size)]
        return -np.mean(y*np.log(y_pred))

