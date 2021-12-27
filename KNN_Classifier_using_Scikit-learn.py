#!/usr/bin/env python
# coding: utf-8

# # Understanding K-Nearest Neighbors Classifier

# We use Scikit-learn to fit, predict, and find the accuarcy of our classifier. 
# We will use MNIST dataset, which has 10 classes, the digits 0 through 9. A reduced version of the MNIST dataset is one of scikit-learn's included datasets. We use use that in this classifier. 
# This is an implementable execution based on what i learned from datacamp course.

# In[1]:


#Importing necessary modules
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


# Load the digits dataset: digits
digits = datasets.load_digits()


# DATA VISUALIZATION 

# In[3]:


# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)


# In[4]:


# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)


# In[5]:


# Display digit 999
plt.imshow(digits.images[999], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


# Implementation: Train/Test split + Fit/Predict/Accuracy

# In[6]:


# Create feature and target arrays
X = np.array(digits.data)
y = np.array(digits.target)


# In[7]:


# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=36, stratify=y)


# In[8]:


# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)


# In[9]:


# Fit the classifier to the training data
knn.fit(X_train, y_train)


# In[10]:


# Predict the labels for the training data X_test
y_pred = knn.predict(X_test)


# In[11]:


# Print the predicted labels for the data points X_test
print("Prediction: {}".format(y_pred))


# In[12]:


# Print the accuracy
print(knn.score(X_test, y_test))


# Accuracy of 98.51% is achieved.

# In[ ]:




