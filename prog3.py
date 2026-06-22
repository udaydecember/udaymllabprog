import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris=load_iris()
data=iris.data
label=iris.target
label_name=iris.target_names

iris_df=pd.DataFrame(data,columns=iris.feature_names)
pca=PCA(n_components=2)
data_reduced=pca.fit_transform(data)
reduced_df=pd.DataFrame(data_reduced,
            columns=['principal component 1','principal component 2'])
reduced_df['labels']=label

plt.figure(figsize=(8,6))
colors=['r','g','b']

for i in np.unique(label):
    plt.scatter(
        reduced_df[reduced_df['labels']==i]['principal component 1'],
        reduced_df[reduced_df['labels']==i]['principal component 2'], 
        label=label_name[i],
        color=colors[i])
    
plt.title("Pca of iris datasets")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.show()
