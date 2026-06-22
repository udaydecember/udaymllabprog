import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data=fetch_olivetti_faces()
x=data.data
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

model=GaussianNB()
model.fit(x_train ,y_train)

y_precd=model.predict(x_test)

accuracy=accuracy_score(y_test,y_precd)
print("acuuracy:",accuracy*100)

fig, axes=plt.subplots(2,3,figsize=(10,8))
for ax,image,true,precd in zip(axes.ravel(),x_test,y_test,y_precd):
    ax.imshow(image.reshape(64,64),cmap='gray')
    ax.set_title(f"T:{True} P:{precd}")
    ax.axis('off')

plt.show()
