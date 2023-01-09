#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np


# In[15]:


class LinearRegression:
    # Linear Regression constructor set learning rate = 0.001 , number of iterations of GD = 1000 by default
    def __init__(self , lr = 0.001 , n_itr = 1000):
        self.lr = lr
        self.n_itr = n_itr
        # Initialize weight and bias
        self.w = None
        self.b = None
        
        
     # Firts: Implement fit function that use GD to find values for parameter (w,b) 
    def fit (self , X , y):
        n_sample , n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        # GD
        for _ in range(n_sample):
            # Calculate model prediction
            y_predicted = np.dot(X , self.w) + self.b
            
            # Calculate gradient
            dw = (1/n_sample)* np.dot(X.T , (y_predicted - y))
            db = (1/n_sample)*np.sum(y_predicted - y)
            
            # Update w , b
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
    # Implement Prediction function that use paramters after they are update using GD in fit function       
    def predict(self , X):
        
        y_predicted = np.dot(X , self.w) + self.b
        return y_predicted

