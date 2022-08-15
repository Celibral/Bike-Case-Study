import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

df = pd.read_csv('df_data_cleaned.csv', index_col=[0])

#Feature Selection

y = df['Purchased Bike']
df = df[df['Purchased Bike'] == 1]
df = df.drop('Purchased Bike', axis=1)
X = df[['Income', 'Children', 'Cars', 'Commute Distance', 'Age']]


scaler = MinMaxScaler()

#  fit  the scaler to the train set
scaler.fit(X) 

X = pd.DataFrame(
    scaler.transform(X),
    columns=X.columns
)


# Create Elbow Plot

w = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    w.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, w)
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal number of clusters (k)')

plt.show()

# Apply KMeans 
kmean = KMeans(n_clusters=2)

kmean.fit(X)
y_pred = kmean.predict(X)

df_final = df

df_final['clusters'] = y_pred

df_final['Income'] = np.exp(df_final['Income'])
df_final['Age'] = np.exp(df_final['Age'])

results = []

print(df_final['clusters'].value_counts())

for i in range(0,2):
    print('CLUSTER %d' % i)
    results.append(df_final[df_final['clusters'] == i].describe())
