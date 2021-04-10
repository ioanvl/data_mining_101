import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import time
import warnings
warnings.filterwarnings('ignore')


tries = 100



wine_data = pd.read_csv('winequality-red.csv')

cumul = 0
for _ in range(tries):

    ran_seed = (int(round(time.time()))) % 5000
    np.random.seed(ran_seed)

    wine_index = wine_data.index.values
    np.random.shuffle(wine_index)

    wine_cut = int(np.floor(len(wine_data) * 0.25))
    train_index, test_index = wine_index[wine_cut:], wine_index[:wine_cut]
    train = wine_data.iloc[train_index]
    test = wine_data.iloc[test_index]

    value_scaler = StandardScaler()
    value_scaler.fit(train.iloc[:, :-1].values.reshape(len(train), -1))

    scaler = SVC()

    scaler.fit(value_scaler.transform(train.iloc[:,:-1].values), train.iloc[:, -1:].values.reshape(-1))
    test['pred'] = scaler.predict(value_scaler.transform(test.iloc[:,:-1].values))

    test['pred'] = pd.to_numeric(test['pred'])
    test['quality'] = pd.to_numeric(test['quality'])

    hit = len(test[test['quality'] == test['pred']])
    total = len(test)
    print(f"{hit} hits on {total} tests")
  
    cumul += (hit/total)

print(f"\nSuccess: {round((cumul / tries) * 100, 2)}%")

orig_wine_data = pd.read_csv('winequality-red.csv')


var_names = orig_wine_data.columns.values
train_names = var_names[:-1]
print(f"Variables:\n{var_names}\n==========================\n")
temp_names = np.delete(var_names, [8]) #remove pH
train_drop_names = temp_names[:-1]

drop_c = 0
mean_c = 0
k_means_c = 0

for _ in range(tries):
    wine_data = orig_wine_data.copy(deep=True)

    ran_seed = (int(round(time.time()))) % 5000
    np.random.seed(ran_seed)

    wine_index = wine_data.index.values
    np.random.shuffle(wine_index)

    wine_cut = int(np.floor(wine_index.size * 0.25))
    train_index, test_index = wine_index[wine_cut:], wine_index[:wine_cut]

    drop_cut = int(np.floor(train_index.size * 0.33))
    del_index = train_index[:drop_cut]
    wine_data.loc[del_index, 'pH'] = np.nan

    np.random.shuffle(train_index)
    train = wine_data.iloc[train_index]
    test = wine_data.iloc[test_index]

    #====================================================================

    value_scaler = StandardScaler()
    value_scaler.fit(train[train_drop_names].values.reshape(len(train), -1))

    scaler = SVC()

    scaler.fit(value_scaler.transform(train[train_drop_names].values), train.iloc[:, -1:].values.reshape(-1))
    test['pred'] = scaler.predict(value_scaler.transform(test[train_drop_names].values))

    test['pred'] = pd.to_numeric(test['pred'])
    test['quality'] = pd.to_numeric(test['quality'])

    hit = len(test[test['quality'] == test['pred']])
    total = len(test)
    print(f"{hit} hits on {total} tests\t\t", end='')

    drop_c += hit / total

    #====================================================================

    mean_ph = wine_data.iloc[train_index]['pH'].mean()
    #print(f"\nMean pH: {mean_ph}")

    train['pH2'] = train['pH'].replace(np.nan, mean_ph)

    train_mean_names = np.append(train_drop_names, 'pH2')


    value_scaler = StandardScaler()
    value_scaler.fit(train[train_mean_names].values.reshape(len(train), -1))

    scaler = SVC()

    scaler.fit(value_scaler.transform(train[train_mean_names].values), train['quality'].values.reshape(-1))
    test['pred2'] = scaler.predict(value_scaler.transform(test[train_names].values))

    test['pred2'] = pd.to_numeric(test['pred2'])
    test['quality'] = pd.to_numeric(test['quality'])

    hit = len(test[test['quality'] == test['pred2']])
    total = len(test)
    print(f"{hit} hits on {total} tests\t\t", end='')

    mean_c += hit / total

    #====================================================================

    clusters = 6

    kmeans_scaler = KMeans(n_clusters=clusters, random_state=0)
    kmeans_scaler.fit(train[train_drop_names])
    #print(kmeans_scaler.labels_)
    train['cluster'] = kmeans_scaler.labels_
    train['pH3'] = train['pH']

    for i in range(clusters):
        temp = (train[train['cluster'] == i])['pH'].mean()

        train.loc[((train['cluster'] == i) & (train['pH3'].isna())), 'pH3'] = temp

    train_k_means_names = np.append(train_drop_names, 'pH3')


    value_scaler = StandardScaler()
    value_scaler.fit(train[train_k_means_names].values.reshape(len(train), -1))

    scaler = SVC()

    scaler.fit(value_scaler.transform(train[train_k_means_names].fillna(0).values), train['quality'].values.reshape(-1))
    test['pred3'] = scaler.predict(value_scaler.transform(test[train_names].values))

    test['pred3'] = pd.to_numeric(test['pred3'])
    test['quality'] = pd.to_numeric(test['quality'])

    hit = len(test[test['quality'] == test['pred3']])
    total = len(test)
    print(f"{hit} hits on {total} tests")

    k_means_c += hit / total

print(f"Drop accuracy: {round(drop_c/tries, 2)}")
print(f"Mean accuracy: {round(mean_c/tries, 2)}")
print(f"K-Means accuracy: {round(k_means_c/tries, 2)}")
