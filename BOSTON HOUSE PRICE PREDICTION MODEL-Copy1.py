#!/usr/bin/env python
# coding: utf-8

# Step 1: Importing essential libraries and boston housing dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from IPython.display import HTML


# Step 2: Data preprocessing

# In[2]:


#load the dataset
boston = load_boston()
#description of the dataset
print(boston.DESCR)


# In[4]:


features = pd.DataFrame(boston.data, columns = boston.feature_names)
features


# In[5]:


features['AGE']


# In[6]:


target = pd.DataFrame(boston.target, columns = ['target'])
target


# In[8]:


max(target['target'])


# In[7]:


min(target['target'])


# In[9]:


#concatenate features and target into a single dataframe
#axis=1 makes it concatenate column-wise
df = pd.concat([features,target], axis=1)
df


# Step 3: Data visualization

# In[15]:


#use round to set the precision to 4 decimal places
df.describe().round(decimals = 4)


# In[16]:


#calculate correlation between every column on the data
corr = df.corr('pearson')
#take absolute values of correlation
corrs = [abs(corr[attr]['target']) for attr in list(features)]
#make a list of pairs[(corr,feature)]
l=list(zip(corrs,list(features)))
#sort the list of pairs in reverse order with correlation value as the key for sorting
l.sort(key=lambda x:x[0], reverse=True)

#unzip pairs to two lists
corrs,labels = list(zip((*l)))

#plot correlations wrt the target variable as a bar graph
index = np.arange(len(labels))
plt.figure(figsize = (15,5))
plt.bar(index,corrs,width=0.5)
plt.xlabel('Attributes')
plt.ylabel('Correlation with the target variable')
plt.xticks(index,labels)
plt.show()


# Step 4: Normalization

# In[38]:


X=df['LSTAT'].values
Y=df['target'].values
#before normalization
print(Y[:15])


# In[39]:


x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X.reshape(-1,1))
X = X[:,-1]
y_scaler = MinMaxScaler()
Y = y_scaler.fit_transform(Y.reshape(-1,1))
Y = Y[:,-1]

print(Y[:15])


# Step 5: Splitting data into fixed sets

# In[40]:


#0.2 indicates 20% of the data is randomly sampled as testing data
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size = 0.2)


# mean squared error : 3 functions (error, update and gradient descent)

# In[41]:


#mean squared error
def error(m,x,c,t):
    N = x.size
    e = sum(((m*x+c)-t) ** 2)
    return e*1/(2*N)


# In[42]:


#update function
def update(m,x,c,t,learning_rate):
    grad_m = sum(2*((m*x+c)-t)*x)
    grad_c = sum(2*((m*x+c)-t))
    m = m - grad_m * learning_rate
    c = c - grad_c * learning_rate
    return m,c


# In[43]:


#gradient descent function
def gradient_descent(init_m,init_c,x,t,learning_rate,iterations,error_threshold):
    m = init_m
    c = init_c
    error_values = list()
    mc_values = list()
    for i in range(iterations):
        e = error(m,x,c,t)
        if e < error_threshold:
            print('Error less than the threshold. Stopping gradient descent.')
            break
        error_values.append(e)
        m,c = update(m,x,c,t,learning_rate)
        mc_values.append((m,c))
    return m,c,error_values,mc_values


# using descent function

# In[44]:


get_ipython().run_cell_magic('time', '', 'init_m = 0.9\ninit_c = 0\nlearning_rate = 0.001\niterations = 250\nerror_threshold = 0.001\n\nm,c,error_values,mc_values = gradient_descent(init_m,init_c,xtrain,ytrain,learning_rate,iterations,error_threshold)')


# model training visualization

# In[45]:


mc_values_anim = mc_values[0:250:5]


# In[46]:


fig,ax = plt.subplots()
ln, = plt.plot([],[],'ro-',animated = True)

def init():
    plt.scatter(xtest,ytest,color='g')
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    return ln,

def update_frame(frame):
    m,c = mc_values_anim[frame]
    x1,y1 = -0.5, m * -0.5 + c
    x2,y2 = 1.5, m * 1.5 + c
    ln.set_data([x1, x2],[y1, y2])
    return ln,

anim = FuncAnimation(fig,update_frame,frames = range(len(mc_values_anim)),init_func = init,blit = True)
HTML((anim.to_html5_video()))


# Step 6: Error visualization

# In[47]:


#plotting the regression line upon the training dataset

plt.scatter(xtrain,ytrain,color = 'g')
plt.plot(xtrain, (m * xtrain + c), color = 'r')


# In[48]:


#plotting error values

plt.plot(np.arange(len(error_values)), error_values)
plt.ylabel('Error')
plt.xlabel('Iterations')


# Step 7: Prediction

# In[49]:


#calculate the predictions on the test set as a vectorized operation
predicted = (m * xtest) + c

#compute MSE for the predicted values on the testing set
mean_squared_error(ytest, predicted)


# In[50]:


#put xtest, ytest and predicted values into a single dataframe so that we can see the predicted 
#values alongside the testing set

p = pd.DataFrame(list(zip(xtest,ytest,predicted)), columns = ['x', 'target_y', 'predicted_y'])
p.head()


# In[51]:


plt.scatter(xtest, ytest, color = 'g')
plt.plot(xtest, predicted, color = 'r')


# In[52]:


#reshape to change the shape that is required by the scaler

predicted = predicted.reshape(-1,1)
xtest = xtest.reshape(-1,1)
ytest = ytest.reshape(-1,1)

xtest_scaled = x_scaler.inverse_transform(xtest)
ytest_scaled = y_scaler.inverse_transform(ytest)
predicted_scaled = y_scaler.inverse_transform(predicted)

#this removes extra dimension
xtest_scaled = xtest_scaled[:,-1]
ytest_scaled = ytest_scaled[:,-1]
predicted_scaled = predicted_scaled[:,-1]

p = pd.DataFrame(list(zip(xtest_scaled, ytest_scaled, predicted_scaled)), columns = ['x', 'target_y', 'predicted_y'])
p = p.round(decimals = 3)
p.head()


# In[ ]:




