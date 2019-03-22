# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 16:12:17 2019

@author: Bolt
"""

# Importing the libraries


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import FastMarkerCluster
from branca.colormap import LinearColormap
import os
from scipy import stats


pd.set_option("display.max_columns",100)
pd.set_option("display.max_rows",1000)


#importing the dataset
reviews = pd.read_csv('reviews.csv')
reviews_details=pd.read_csv('reviews.csv.gz')
neighbourhoods = pd.read_csv('neighbourhoods.csv')
calendar = pd.read_csv('calendar.csv.gz')
listings_details=pd.read_csv('listings.csv.gz')
listings_summary=pd.read_csv('listings.csv')

listings_details.accommodates


#exploring the data


'''The listings_details file contains a total of 96 
variables. I am not going to use all of these, but
selectively joined a number of variables
that seemed useful to me for this EDA'''

target_columns = ["id", "property_type", "accommodates", "first_review", "review_scores_value", "review_scores_cleanliness", "review_scores_location", "review_scores_accuracy", "review_scores_communication", "review_scores_checkin", "review_scores_rating", "maximum_nights", "listing_url", "host_is_superhost", "host_about", "host_response_time", "host_acceptance_rate", "street", "weekly_price", "monthly_price", "market"]
listings = pd.merge(listings_summary, listings_details[target_columns], on='id', how='left')
listings.info()


#dropping acceptance rate
listings = listings.drop(columns=['neighbourhood_group','host_acceptance_rate','weekly_price','monthly_price'])
listings.head()
#********************************************
#SENTIMENT ANALYSIS ON REVIEW DATA
reviews_details.head(4)

import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files

#unpickling the classifier
with open('classifier.pickle','rb') as f:
    clf=pickle.load(f)
    
with open('tfidfmodel.pickle','rb') as f:
    tfidf=pickle.load(f)
    
reviews_details[reviews_details['sentiment']==0]
corpus=[]
for i in range(0,len(reviews_details.comments)):
    review=re.sub(r'\W',' ',str(reviews_details.comments[i]))#removing non words
    review=review.lower()
    review=re.sub(r'\s+[a-z]\s+',' ',review)# removing single words
    review=re.sub(r'^[a-z]\s+',' ',review)#single word at front
    review=re.sub(r'\s+',' ',review)
    corpus.append(review)

corpus=tfidf.transform(corpus).toarray()
reviews_details['sentiment']=clf.predict(corpus)

#groupby listing id and get the sentiment values
review_details1=reviews_details.groupby('listing_id')['sentiment'].mean()
df1 = pd.DataFrame(data=review_details1.index, columns=['listing_id'])
df2 = pd.DataFrame(data=review_details1.values, columns=['sentiment_values'])
df = pd.merge(df1, df2, left_index=True, right_index=True)

#rename column in listing for merging
listings.columns
listings.rename(columns={'id':'listing_id'}, inplace=True)

#lets join the sentiment value with listings

listings=pd.merge(listings,df,on='listing_id',how='outer')

listings.sentiment_values=listings.sentiment_values.fillna(0)


listings.to_csv('finallisting.csv')

listings.head(5)
#**********************************************
#FINDING TREND IN CALENDAR DATA
calendar.available[calendar.available=='t']=1
calendar.available[calendar.available=='f']=0
calendar.available.value_counts()
calendar = calendar.dropna(subset=['available'])
calendar['available']=calendar['available'].astype(int)


forecast=calendar.groupby('date')['available'].sum()


fdf1 = pd.DataFrame(data=forecast.index, columns=['date'])
fdf2 = pd.DataFrame(data=forecast.values, columns=['availablity'])
forecast_data = pd.merge(fdf1, fdf2, left_index=True, right_index=True)

forecast_data.to_csv('finalforecast.csv')

#**********************************************
# unique counts
def unique_counts(sample):
   for i in sample.columns:                #traversing through columns
       count = sample[i].nunique()
       print(i, ": ", count)
unique_counts(listings)


#lets draw histogram for less no of categorical value
listings['host_is_superhost'].value_counts()
listings['market'].value_counts()
listings['room_type'].value_counts
listings.info()


feq=listings['neighbourhood'].value_counts().sort_values(ascending=True)
feq.hist()
ax=feq.plot.barh(figsize=(10, 8), color='b', width=1)
plt.title("Number of listings by neighbourhood", fontsize=20)
plt.xlabel('Number of listings', fontsize=12)
plt.show()

#making superhost value to 0 and 1

listings.host_is_superhost[listings.host_is_superhost=='t']=1
listings.host_is_superhost[listings.host_is_superhost=='f']=0
listings.host_is_superhost.value_counts()
listings = listings.dropna(subset=['host_is_superhost'])

listings['host_is_superhost']=listings['host_is_superhost'].astype(int)


############## Outperforming/underperforming segments - two sample t test #############
def stats_comparison(i):
    listings.groupby(i)['host_is_superhost'].agg({
    'average': 'mean',
    'hosts': 'count'
    }).reset_index()
    cat = listings.groupby(i)['host_is_superhost']\
        .agg({
            'sub_average': 'mean',
            'sub_hosts': 'count'
       }).reset_index()
    cat['overall_average'] = listings['host_is_superhost'].mean()
    cat['overall_hosts'] = listings['host_is_superhost'].count()
    cat['rest_hosts'] = cat['overall_hosts'] - cat['sub_hosts']
    cat['rest_average'] = (cat['overall_hosts']*cat['overall_average'] \
                     - cat['sub_hosts']*cat['sub_average'])/cat['rest_hosts']
    cat['z_score'] = (cat['sub_average']-cat['rest_average'])/\
        np.sqrt(cat['overall_average']*(1-cat['overall_average'])
            *(1/cat['sub_hosts']+1/cat['rest_hosts']))
    cat['prob'] = np.around(stats.norm.cdf(cat.z_score), decimals = 10)
    cat['significant'] = [(lambda x: 1 if x > 0.9 else -1 if x < 0.1 else 0 )(i) for i in cat['prob']]
    print(cat.head)
    
    stats_comparison('market')
    stats_comparison('street')
    stats_comparison('neighbourhood')
    stats_comparison('host_response_time') #with in an hour responce are mostlly not super host
    stats_comparison('room_type')  #entire home/apt are not mostly superhost
    stats_comparison('property_type')
    
#*****************************************************************************
#K=-MEANS CLUSTERING
rev=['review_scores_value',               
'review_scores_cleanliness',         
'review_scores_location',            
'review_scores_accuracy',            
'review_scores_communication',       
'review_scores_checkin',             
'review_scores_rating']

#replacing na values with zero

listings.info() 
listings[rev]=listings[rev].fillna(0)
listings['overall_review'] = listings[rev].apply(lambda row: (row.review_scores_value + row.review_scores_cleanliness + row.review_scores_location + row.review_scores_accuracy + row.review_scores_communication + row.review_scores_checkin + row.review_scores_rating)/7, axis=1)


num_list=['price', 'host_is_superhost',                          
'minimum_nights',                    
'number_of_reviews',
'reviews_per_month',                 
'calculated_host_listings_count',   
'availability_365',                  
'accommodates',                      
'overall_review',             
'maximum_nights']

#grouping based on street
grouping=listings.dropna(axis=0)[num_list + ['street']]
grouping.info()
grouping1 =grouping.groupby('street').mean().reset_index().dropna(axis=0)

#********************************************************
# Step 2: shall I standardise the data?
# What is the magnitude of data range?
from sklearn import preprocessing 
grouping_std = grouping1.copy()

for i in num_list:
    grouping_std[i] = preprocessing.scale(grouping_std[i])


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++') #kmeans ++ used to correctly select the cluster
    kmeans.fit(grouping_std[num_list])
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('number of cluster')
plt.ylabel('wcss')
plt.show()

#applying kmeans
km=KMeans(n_clusters=5,init='k-means++')
grouping_std['cluster']=km.fit_predict(grouping_std[num_list])
                   
grouping.merge(grouping_std[['street','cluster']])\
    .groupby('cluster')\
    .mean() 
    
    
#*******************************************************
#decision tree     

# choose one of the city clusters to analyze
from sklearn.cross_validation import train_test_split
tree_data = listings.dropna(axis = 0)
tree_train, tree_test = train_test_split(tree_data, test_size=0.2, random_state=1, stratify=tree_data['host_is_superhost'])

listings.dtypes
#build the decision tree model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_leaf_nodes=10,min_samples_leaf=200)
clf=clf.fit(tree_train[['price', 'sentiment_values',                           
'minimum_nights',
'accommodates',
                 
'calculated_host_listings_count',   
'availability_365',                  
'accommodates',                      
'overall_review',       
'maximum_nights']],tree_train['host_is_superhost'])

# scoring of the prediction model
clf.score(tree_test[['price',                            
'minimum_nights', 'sentiment_values',
'accommodates',
'calculated_host_listings_count',   
'availability_365',                  
'accommodates',                      
'overall_review',             
'maximum_nights']], tree_test['host_is_superhost'])

# visualize the decision tree
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names =['price',                            
'minimum_nights','sentiment_values',
'accommodates',            
'calculated_host_listings_count',   
'availability_365',                  
'accommodates',                      
'overall_review',             
'maximum_nights'], filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("booking13_tree.pdf")

#******************************************************************
#time series analysis on boarding and accomadation trend

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
import math
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pandas import DataFrame
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf


time_series=pd.read_csv('finalforecast.csv',index_col=[1])

time_series.head()

data = time_series['Occupied']
data.head()

# Function to generate the PDQ Values for ARIMA


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.75)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=False,transparams=True,trend='c',)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = math.sqrt(mean_squared_error(test, predictions))
    return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


p_values = range(0, 10)
d_values = range(0, 10)
q_values = range(0, 10)
warnings.filterwarnings("ignore")
evaluate_models(data.values, p_values, d_values, q_values)


data = data.astype('float32')
mod1 = ARIMA(data,order=(10,2,4))
results = mod1.fit()
print(results.summary())
results.plot_predict()
list1=[results.forecast(steps=365)[0]]


import matplotlib.pyplot as plt
plt.plot(list1)
plt.ylabel('next_year_prediction')
plt.show()

results.forecast(steps=365)[0]
