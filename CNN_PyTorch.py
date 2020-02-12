#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Q2 CNN on FashionMNIST
import matplotlib.pyplot as plt
from torch import tensor
import torch
import matplotlib as mpl
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


# Data label creation
def get_data():
    train_data = pd.read_csv('C:/Users/Chandan/Documents/CPTS 570/Homework/HW 2/data/fashion-mnist_train.csv')
    test_data = pd.read_csv('C:/Users/Chandan/Documents/CPTS 570/Homework/HW 2/data/fashion-mnist_test.csv')
    x_train = train_data[train_data.columns[1:]].values
    y_train = train_data.label.values
    x_test = test_data[test_data.columns[1:]].values
    y_test = test_data.label.values
    return map(tensor, (x_train, y_train, x_test, y_test)) # maps are useful functions to know
                                                           # here, we are just converting lists to pytorch tensors
    
    
x_train, y_train, x_test, y_test = get_data()
train_n, train_m = x_train.shape
test_n, test_m = x_test.shape
n_cls = y_train.max()+1

### Normalization
x_train, x_test = x_train.float(), x_test.float()
train_mean,train_std = x_train.mean(),x_train.std()
train_mean,train_std
def normalize(x, m, s): return (x-m)/s
x_train = normalize(x_train, train_mean, train_std)
x_test = normalize(x_test, train_mean, train_std) # note this normalize test data also with training mean and standard deviation




# Definition of the model
class FashionMnistNet(nn.Module):
    # Based on Lecunn's Lenet architecture
    def __init__(self):
        super(FashionMnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, stride = 2, padding = 2)   # creating layers
        self.conv2 = nn.Conv2d(8, 16, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.fc1 = nn.Linear(32, 10)
        #self.avg = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))     #Relu function layer then the max layer (operation)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        pool = nn.AdaptiveAvgPool2d(1)
        x = pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
# instantiating the model
model = FashionMnistNet()
print(model)


# In[5]:


# Optimal Learning rate

import math
def find_lr(net, loss_func, init_value = 1e-8, final_value=10., beta = 0.98, bs = 32):
    num = (train_n-1)//bs + 1 # num of batches 
    mult = (final_value/init_value) ** (1/num)
    lr = init_value
    optimizer = optim.SGD(net.parameters(), lr=lr)
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0.
    losses = []
    log_lrs = []
    for i in range((train_n-1)//bs + 1):
        batch_num += 1
        start_i = i*bs
        end_i = start_i+bs
        xb = x_train[start_i:end_i].reshape(bs, 1, 28, 28)
        yb = y_train[start_i:end_i]
        optimizer.zero_grad()
        outputs = net.forward(xb)
        loss = loss_func(outputs, yb)
        
        #Compute the smoothed loss
        print("loss: ", loss.item())
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
            
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        
        #Do the SGD step
        loss.backward()
        optimizer.step()
        
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
        
    return log_lrs, losses

model_lrfinder = FashionMnistNet()
bs = 32
loss_func = F.cross_entropy
log_lrs, losses = find_lr(model_lrfinder, loss_func)

plt.plot([10**x for x in log_lrs], losses)


# In[6]:


model_wnd = FashionMnistNet()
lr = 0.5 # learning rate
epochs = 10 # number of epochs
bs = 100
loss_func = F.cross_entropy
opt = optim.SGD(model_wnd.parameters(), lr=lr)
accuracy_vals_wnd = []
for epoch in range(epochs):
    model_wnd.train()
    for i in range((train_n-1)//bs + 1):
        start_i = i*bs
        end_i = start_i+bs
        xb = x_train[start_i:end_i].reshape(bs, 1, 28, 28)
        yb = y_train[start_i:end_i]
        loss = loss_func(model_wnd.forward(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    model_wnd.eval()
    with torch.no_grad():
        total_loss, accuracy = 0., 0.
        validation_size = int(test_n/10)
        for i in range(test_n):
            x = x_test[i].reshape(1, 1, 28, 28)
            y = y_test[i]
            pred = model_wnd.forward(x)
            accuracy += (torch.argmax(pred) == y).float()
        print("Accuracy: ", (accuracy*100/test_n).item())
        accuracy_vals_wnd.append((accuracy*100/test_n).item())
        
    
        
axis = (1 , 2 , 3 , 4, 5, 6, 7, 8, 9, 10)
plt.plot(axis,accuracy_vals_wnd , 'r--')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy ')
plt.legend(('Test Accuracy'))
plt.show()


# In[7]:


model_wnd = FashionMnistNet()
lr = 0.5 # learning rate
epochs = 10 # number of epochs
bs = 100  # Batches
loss_func = F.cross_entropy
opt = optim.SGD(model_wnd.parameters(), lr=lr)
accuracy_vals_wnd1 = []
for epoch in range(epochs):
    model_wnd.train()
    for i in range((train_n-1)//bs + 1):    #training in 1bs batches
        start_i = i*bs
        end_i = start_i+bs
        xb = x_train[start_i:end_i].reshape(bs, 1, 28, 28)
        yb = y_train[start_i:end_i]
        loss = loss_func(model_wnd.forward(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    model_wnd.eval()
    with torch.no_grad():
        total_loss, accuracy = 0., 0.
        validation_size = int(train_n/10)
        for i in range(train_n):
            x = x_train[i].reshape(1, 1, 28, 28)
            y = y_train[i]
            pred = model_wnd.forward(x)
            accuracy += (torch.argmax(pred) == y).float()
        print("Accuracy: ", (accuracy*100/train_n).item())
        accuracy_vals_wnd1.append((accuracy*100/train_n).item())
        
    
        
axis = (1 , 2 , 3 , 4, 5, 6, 7, 8, 9, 10)
plt.plot(axis,accuracy_vals_wnd1 , 'r--')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy ')
plt.legend(('Train Accuracy'))
plt.show()


# In[8]:


plt.plot(axis,accuracy_vals_wnd , 'r--',axis,accuracy_vals_wnd1 , 'b--')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy ')
plt.legend(('Test Accuracy','Training Accuracy'))
plt.show()


# In[23]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

MyData = pd.read_csv("/Users/Chandan/Documents/CPTS 570/Project/Predict-zika-res-master/Outputs/Imputed_data_1.csv", header=None)
Y_train = MyData[MyData.columns[2:3]].values
X_train = MyData[MyData.columns[7:]].values



scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)

gnb = GaussianNB()
GnBTraining_acc = []
GnBTest_acc = []


count = 0
for i in range(len(X_train)):
    y = gnb.fit(X_train,Y_train).predict([X_train[i,:]])                  # Laplace Smoothing by default
    if y != Y_train[i]:
        count = count + 1 
GnBTraining_acc.append((len(X_train)-count)/len(X_train)*100)

GnBTraining_acc


# In[20]:


y = gnb.fit(X_train,Y_train).predict([X_train[79,:]])
y


# In[22]:


y = gnb.fit(X_train,Y_train).predict([X_train[79,:]])
y


# In[24]:


y = gnb.fit(X_train,Y_train).predict([X_train[79,:]])
y


# In[ ]:




