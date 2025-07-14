import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


df = pd.read_csv("ifood_df.csv")
df.head()


# Check shape and info
print(df.shape)
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Basic statistics
print(df.describe())


# Drop irrelevant or redundant columns
df = df.drop(['DtCustomer', 'Z_CostContact', 'Z_Revenue'], axis=1, errors='ignore')

# Fill or drop missing values
df = df.dropna()

# Convert categorical variables if needed
df = pd.get_dummies(df, drop_first=True)


features = df[[
    'Income', 'Kidhome', 'Teenhome', 'Recency',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
    'NumWebVisitsMonth'
]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)


pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_features)
df['PCA1'] = reduced[:,0]
df['PCA2'] = reduced[:,1]

sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title('Customer Segments (PCA Reduced)')
plt.show()

cluster_summary = df.groupby('Cluster')[features.columns].mean()
print(cluster_summary)

sns.boxplot(x='Cluster', y='Income', data=df)
plt.title('Income Distribution by Cluster')
plt.show()


df.to_csv('segmented_customers.csv', index=False)
