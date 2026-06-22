1)import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df=pd.read_csv(r"C:\Users\Pallavi AR\Downloads\pml\housing.csv")
print(df.head(10))
print(df.shape)
df.info()
df.nunique()
print(df.nunique())
df.isnull().sum()
print(df.isnull().sum())
df.duplicated
print(df.duplicated())
df.describe()
print(df.describe())
Numerical=df.select_dtypes(include=[np.number]).columns
print(Numerical)
for col in Numerical:
    plt.figure(figsize=(10,6))
    df[col].plot(kind='hist',title=col,bins=60,edgecolor='black')
    plt.ylabel('Frequency')

plt.show()

for col in Numerical:
     plt.figure(figsize=(8,6))
     sns.boxplot(df[col],edgecolor='blue')
    
plt.ylabel(col)
plt.title(col)
plt.show()
