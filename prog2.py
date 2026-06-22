import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data=pd.read_csv(r"C:\Users\Pallavi AR\Downloads\pml\housing.csv")
correaltion_matrix=data.select_dtypes(include='number').corr()
print(correaltion_matrix)
plt.figure(figsize=(10,6))
sns.heatmap(correaltion_matrix,cmap='viridis',annot=True,fmt='.2f',linewidths=0.5)
plt.title('Correletion matrics of claifornia husing feature')
plt.show()
sns.pairplot(data.select_dtypes(include='number'),
             diag_kind='kde',
             plot_kws={'alpha':0.5})
plt.suptitle('pair plot housing feature',y=1.02)
plt.show()
