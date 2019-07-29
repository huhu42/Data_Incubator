#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
from alpha_vantage.timeseries import TimeSeries

import os

import pprint


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


# In[92]:


def plot_bar_x():
    # this is for plotting purpose
    label = top.index
    index = np.arange(len(top))
    plt.bar(index, top['SIC'])
    plt.xlabel('Top Features', fontsize=10)
    plt.ylabel('No of Occurance', fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=-90)
    plt.title('Top Fundamental Indicators')
    plt.show()


# In[93]:


plot_bar_x()


# In[146]:


test_=df_fund.loc[df_fund['indicator_id'].isin(group_ind),['index','indicator_id','2012']].dropna()

test_ = test_.pivot(index='index',columns='indicator_id', values='2012')
test_['ticker']=test_


# In[ ]:





# In[44]:


#pick one stock to explore first, Amazon
sample = df_fund[df_fund['Ticker']=='AMZN']
sample['indicator_id']
#change these indicators into percentile in the industry


# In[41]:


amzn = pd.read_csv('stock_dfs/AMZN.csv',header=0, date_parser = True)


# In[42]:


amzn
#use the fundamental data as 1 year early, since reporting will usually not be available until the 


# In[20]:


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


# In[21]:


#get a list of tickers to download the prices
tickers = set(df_all['Ticker'])
print ("The number of all stocks is: ")
print (len(tickers))
alpha_stocks(tickers)
#print(apple_data)
#alpha_stocks(symbol = ['SOHU'])
filenames = os.listdir('stock_dfs')
print (filenames)


# In[ ]:





# In[ ]:




