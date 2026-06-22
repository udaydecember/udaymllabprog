import pandas as pd
data=pd.read_csv(r"C:\Users\Pallavi AR\Downloads\pml\Boston housing dataset (1).csv")
print(data)
h=['?']*(len(data.columns) - 1)
for i in range (len(data)):
    if data.iloc[i, -1].lower()=='yes':
        for j in range(len(h)):
            if h[j]=='?':
                h[j]=data.iloc[i,j]
            elif h[j]!=data.iloc[i,j]:
                h[j]='?'

print("\n final hypothesis")
print(h)
