# %%
# load libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# %%
# load data
name_dem = "house_votes_Dem.csv"
name_rep = "house_votes_Rep.csv"
df = pd.read_csv(name_dem, encoding='Latin-1')
# rep_votes = pd.read_csv(name_rep)


# %%
# take a look at the data
df.info()


# %%
# separate out the numeric features
numeric_cols = df.dtypes[df.dtypes == 'int64'].index
df_num = df[numeric_cols]


# %%
# documentation for kmeans in sklearn
# help(KMeans)

# %% build a kmeans model
kmeans = KMeans(n_clusters=3, random_state=42, verbose=1)
kmeans.fit(df_num)

# %% look at the information in the model
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
print(kmeans.labels_)


# %%
# add the cluster labels to the original data frame
df['cluster'] = kmeans.labels_

# %%
intertias = [] # inertia is the distance of the data points to their cluster centers
k_values = range(1, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_num)
    intertias.append(kmeans.inertia_)    
  
# %% simple plot of the clusters
# help(plt.scatter)
plt.figure(figsize=(10, 5))
plt.plot(k_values, intertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# %%


