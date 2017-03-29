# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 19:43:34 2017

@author: lduan
"""
#General imports
from sklearn import datasets
from sklearn import metrics
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#import and clean

emails = pd.read_csv('email_table.csv')
opened = pd.read_csv('email_opened_table.csv')
clicked = pd.read_csv('link_clicked_table.csv')

#check for duplicates

emails.describe() #100,000 emails
len(set(emails['email_id'])) #no duplicate emails
pd.unique(emails["email_text"]) #email has either short or long text
np.unique(emails["hour"])

#combine with a second dataset
#join emails with opened and clicked on email_id
opened['opened'] = 1
clicked['clicked'] = 1

s1 = pd.merge(emails, opened, how='left', on=['email_id'])
s2 = pd.merge(s1, clicked, how='left', on=['email_id'])

#replace all NaN with 0
s2=s2.fillna(0)

#create calculated columns (dummy vars, new groupings)

##group by weekday/weekend using 'IsWkday')
def set_wkday(row):
    if row["weekday"] == "Saturday":
        return 0
    elif row["weekday"] == "Sunday":
        return 0
    else:
        return 1

s2 = s2.assign(IsWkday=s2.apply(set_wkday, axis=1))

##create dummies of "user_country"
country_dummies = pd.get_dummies(s2.user_country, prefix='country').iloc[:, 1:]
s2 = pd.concat([s2, country_dummies], axis=1)
s2.head()

#describe the data (statistics, visualizations)

s2['opened'].sum()/len(s2) #10.3% email open rate
s2['clicked'].sum()/s2['opened'].sum() #20.5% click-through rate
##click-through by email_text
s2.loc[s2['email_text']=='long_email', 'clicked'].sum()/len(clicked) 
#44.0% of clicked emails are long emails

##click-through rate by country
100*s2.groupby(['user_country'])['clicked'].sum()/s2.groupby(['user_country'])['opened'].sum()
#user_country
#ES    21.282051
#FR    19.704433
#UK    20.534224
#US    20.466937
#click-through rate averages 20% across all countries

#email open rate by personalization
s2.groupby(['email_version'])['opened'].sum()/len(opened)
#email_version
#generic         0.385114
#personalized    0.614886

#Personalized emails are 1.6x more likely to be opened than generic emails

#email click-through rate by personalization
s2.groupby(['email_version'])['clicked'].sum()/len(clicked)
#email_version
#generic         0.35866
#personalized    0.64134

#Personalization also makes emails 1.8x more likely to be clicked than generic emails

#email open rate by sent hour
openrate=s2.groupby(['hour'])['opened'].sum()/s2.groupby(['hour'])['email_id'].count()

#Plot of open rate
plt.figure();
plt.title('Email Open Rate by Hour Sent')
openrate.plot.bar(alpha=.5)
plt.show()

#Peak email open hours are 9am-5pm, and again from 11pm-midnight

#email open rate by weekday/weekend
s2.groupby(['IsWkday'])['opened'].sum()/len(opened)

#IsWkday
#0    0.245336
#1    0.754664

#Emails sent on weekdays are 3x more likely to be opened


#identify areas for further research
#Relationship between user purchase history and propensity to open email
#Relationship between user purchase history and propensity to click