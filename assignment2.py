# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:21:34 2020

@author: hp pc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ds=pd.read_csv("PCA_practice_dataset.csv")
print(ds)
X=ds.to_numpy()
X.shape
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)
X.shape
pca=PCA(n_components=10)
X=pca.fit_transform(X)
plt.figure(figsize=(10,8))
plt.scatter(X[ :,0],X [ :,1],cmap='plasma')
plt.xlabel('first principal components')
plt.ylabel('second principal components')

cumulitive_variance=np.cumsum(pca.explained_variance_ratio_)*100
threholds=[i for i in range(90,97+1,1)]
components=[np.argmax(cumulitive_variance>threhold)for threhold in threholds]
for component,threshold in zip(components,threholds):
    print("components required for {}% threshold are:{}".format(threshold,component))
    
plt.plot(components,range(90,97+1,1),'ro-',linewidth=2)
plt.title('scree plot')
plt.xlabel('principal component')
plt.ylabel('threshold in %')
plt.show()

X_orig=X
for component,var in zip(components,threshold):
    pca=PCA(n_components=component)
    X_transformed=pca.fit_transform(X_orig)
    print('performing d.r.to retain {}% threshold'.format(var))
    print('after performing d.r.to new shape of the dataset is :',X_transformed.shape)
    print('\n')

   

