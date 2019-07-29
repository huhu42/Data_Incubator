#!/usr/bin/env python
# coding: utf-8

# In[273]:


import pandas as pd
from alpha_vantage.timeseries import TimeSeries

import os

import pprint
import datetime as dt

from sklearn.preprocessing import StandardScaler


# In[15]:


ts = TimeSeries(key='AIH485OT2P8QGKIX', output_format = 'pandas')


# In[22]:


df = pd.read_csv('fund_comp.csv', header = 0).set_index('company_id')
df1 = pd.read_csv('fund_by_comp.csv', header = 0).set_index('company_id')


# In[17]:


#data from http://rankandfiled.com/#/data/hedgefunds
df2 = pd.read_csv('cik_ticker_map.csv',header=0, delimiter = '|').set_index('CIK')


# In[ ]:


df2 = pd.read_csv('cik_ticker_map.csv',header=0, delimiter = '|').set_index('CIK')


# In[128]:


#df2
df_map = df.join(df2, how='inner')
#df2, only keep whats needed
df_map = df_map[['Ticker','Exchange','SIC']]
#df_map


# In[133]:


#df2
df_fund = df1.join(df2, how='inner')
#df2, only keep whats needed
df_fund = df_fund[['indicator_id', 'SIC','Ticker','2010','2011','2012','2013','2014','2015','2016']].reset_index()
df_fund


# In[46]:


df_fund.describe()
#get some data for the various years/


# In[103]:


df_fund


# In[130]:


df_fund.groupby('SIC').count()


# In[131]:


grouped = df_fund.groupby('indicator_id').count().sort_values(by= ['SIC'], ascending=False)
#a lot of different indicators

group_ind = grouped.index[:25].tolist()
#just keep the top 50 index for now, use to filter out non-important data
group_ind


# In[125]:


top = grouped[:25]
top


# In[75]:


#graph the top 25 
import matplotlib.pyplot as plt
import numpy as np


# In[150]:


def plot_bar_x():
    # this is for plotting purpose
    label = top.index
    index = np.arange(len(top))
    plt.bar(index, top['SIC'])
    plt.xlabel('Top Features', fontsize=10)
    plt.ylabel('No of Occurance', fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=-90)
    plt.title('Top Fundamental Indicators')
    plt.savefig('Top Fundamental Indicators.png')
    plt.show()
    


# In[152]:


plot_bar_x()
#plt


# In[165]:


map_tik = df2['Ticker']


# In[194]:


#pivot the indicators for one year and try to figure out what are the most important ones
test_=df_fund.loc[df_fund['indicator_id'].isin(group_ind),['index','indicator_id','2015']]
#use only the ones with all values first
test_ = test_.pivot(index='index',columns='indicator_id', values='2015').dropna()
test_=test_.join(map_tik, how = 'inner')

test_


# In[195]:


#take 100 to test it out
test_set = set(test_['Ticker'])


# In[196]:


def alpha_stocks(symbol, start_date=(2018, 1, 1), end_date=None):

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    fails = []
    warrants = []
    for i in symbol:
        #if os.path.exists('stock_dfsa/{}.csv'.format(i)):
            #return
        if '.' in i:
            print (i + " is a warrant")
            warrants.append(i)
            ii = i.split('.')[0]
            try:
                data, meta_data = ts.get_daily_adjusted(symbol = ii, outputsize='full')
            except:
                #print (ii + ' is not available')
                fails.append(i)
        else:
            print (i)
            try:
                data, meta_data = ts.get_daily_adjusted(symbol = i, outputsize='full')
            except:
                #print (i + ' is not available')
                fails.append(i)
        if data.empty == False:
            data.columns = ['Open','High','Low','Close','Adj Close','Volume','Dividend','Split']
            data.index.names = ['Date']

            #print (all_data)
            data.to_csv('stock_dfs/{}.csv'.format(i))
            print ("Downloading data for " + i)


# In[41]:


amzn = pd.read_csv('stock_dfs/AMZN.csv',header=0, date_parser = True)


# In[198]:


#get a list of tickers to download the prices
tickers = test_set
#print ("The number of all stocks is: ")
#print (len(tickers))
alpha_stocks(tickers)
#print(apple_data)
#alpha_stocks(symbol = ['SOHU'])
filenames = os.listdir('stock_dfs')
#print (filenames)


# In[255]:


#initialize a df to remember returns
df_ret = pd.DataFrame(0, index=tickers, columns=['ret'])


# In[259]:



for index, row in df_ret.iterrows():
    #print(index,row)
    pd_temp = pd.read_csv('stock_dfs/{}.csv'.format(index),header=0,date_parser = True)
    pd_temp['Date'] = pd_temp['Date'].astype('datetime64[ns]')
    pd_temp['Year'] = pd_temp['Date'].map(lambda x: x.year)
    #print(pd_temp.head())
    avg_2016 = pd_temp[pd_temp['Year'] == 2016]['Adj Close'].mean()
    #print(avg_2016)
    avg_2015 = pd_temp[pd_temp['Year'] == 2015]['Adj Close'].mean()
    #print(avg_2015)
    ret = (avg_2016/avg_2015)-1
    df_ret.loc[index, 'ret'] = ret
    #print(ret)
    #print(row['ret'])


# In[260]:


df_ret.sort_values(by=['ret'])
#VXN is very high, discard
df_ret = df_ret[df_ret['ret']<=1]
df_ret


# In[265]:



len(test_)


# In[252]:



#reset index to ticker so can join
test_ = test_.set_index('Ticker')


# In[266]:


df_ret


# In[279]:


df_final = test_.join(df_ret, how = 'inner')


# In[291]:


# Separating out the features
x = df_final.loc[:, df_final.columns != 'ret'].values
# Separating out the target
y = df_final.loc[:,['ret']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
x
y


# In[289]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents , columns = ['principal component 1', 'principal component 2'])
principalDf


# In[294]:


df_finall = df_final.reset_index()
df_finall


# In[295]:


df_final = df_final.reset_index()
finalDf = pd.concat([principalDf, df_finall[['ret']]], axis = 1)
finalDf


# In[302]:


from sklearn.decomposition.pca import PCA
#explained variance
print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100))


# In[313]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
jet=plt.get_cmap('jet')
plt.scatter(finalDf['principal component 1'], finalDf['principal component 2'], s=100, c=finalDf['ret'], cmap=jet)
plt.savefig('pca.png')

