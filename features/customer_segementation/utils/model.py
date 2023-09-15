import numpy as np
import pandas as pd
import datetime
import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth, fpmax
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import StandardScaler    
from sklearn.cluster import KMeans
from utils.kmeans_feature_importance import KMeansInterp

def preprocessing(df):
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Total'] = df['Quantity'] * df['UnitPrice']
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['myd'] = df['InvoiceDate'].dt.strftime('%Y-%m-%d')
    return df

def range_selection(df, date:str):
    min = df[date].min().split('-')
    max = df[date].max().split('-')
    min_date = datetime.date(int(min[0]), int(min[1]), int(min[2]))
    max_date = datetime.date(int(max[0]), int(max[1]), int(max[2]))
    return min_date, max_date

def create_rfm_table(df, date, transaction_id, customer_id):
    # Create RFM table
    snap_date = df[date].max()
    rfmTable = df.groupby(customer_id).agg({date: lambda x: (snap_date - x.max()).days, 
                transaction_id: lambda x: x.nunique(), 'Total': lambda x: x.sum()}).reset_index()
    rfmTable[date] = rfmTable[date].astype(int)
    rfmTable.rename(columns={date: 'recency', transaction_id: 'frequency', 'Total': 'monetary'}, inplace=True)
    return rfmTable

def rfm_segmentation(df, technique:int, num_segments:int):
    if technique == 1:
        data = df.copy()
        # K-means clustering
        X_ = data[['recency', 'frequency', 'monetary']]
        X = StandardScaler().fit_transform(X_)
        kms = KMeansInterp(
	    n_clusters=num_segments, random_state=42,
	    ordered_feature_names=X_.columns.tolist(), 
	    n_init = 'auto', max_iter = 1000,
	    feature_importance_method='wcss_min', # or 'unsup2sup'
        ).fit(X)
        #kmeans = KMeans(n_clusters=num_segments, random_state=42, n_init='auto', max_iter=1000).fit(X)
        segment = kms.labels_ + 1
        data['RFM_Segment'] = segment
        data['RFM_Segment'] = data['RFM_Segment'].astype(str)
        feature_importance = kms.feature_importances_
        return data, feature_importance
    elif technique == 2:
        data = df.copy()
        data['R'] = pd.qcut(df['recency'].rank(method = 'first').values, q=4, labels=[4,3,2,1])
        data['F'] = pd.qcut(df['frequency'].rank(method = 'first').values, q=4, labels=[1,2,3,4])
        data['M'] = pd.qcut(df['monetary'].rank(method = 'first').values, q=4, labels=[1,2,3,4])
        X_ = data[['R', 'F', 'M']]
        X = StandardScaler().fit_transform(X_)
        kms = KMeansInterp(
	    n_clusters=num_segments, random_state=42,
	    ordered_feature_names=X_.columns.tolist(), 
	    n_init = 'auto', max_iter = 1000,
	    feature_importance_method='wcss_min', # or 'unsup2sup'
        ).fit(X)
        #kmeans = KMeans(n_clusters=num_segments, random_state=42, n_init='auto', max_iter=1000).fit(X)
        #segment = kmeans.labels_
        segment = kms.labels_ + 1
        data['RFM_Segment'] = segment
        data['RFM_Segment'] = data['RFM_Segment'].astype(str)
        feature_importance = kms.feature_importances_
        data = data.drop(['R', 'F', 'M'], axis=1)
        return data, feature_importance
    else:
        pass


def encode_units(x):
        if x <= 0:
            return 0
        else:
            return 1

def mba(df, support):
    df1 = df.copy()
    df2 = df1[df1.sum(axis=1) > 1]
    frequent_itemsets = apriori(df2.astype('bool'), min_support=support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1).sort_values(by='lift', ascending =False)
    return rules


#def frequent_bought_recommender(basket_df, item, support):
    basket_sets = basket_df.copy()
    df_item = basket_sets.loc[basket_sets[item] == 1]
    item_frequent = apriori(df_item, min_support=support, use_colnames=True)
    a_rules = association_rules(item_frequent, metric="lift", min_threshold=1)
    # Sorting on lift and support
    a_rules.sort_values(['lift','support'],ascending=False).reset_index(drop=True)
    recommender = a_rules['consequents'].unique()[:5]
    return recommender