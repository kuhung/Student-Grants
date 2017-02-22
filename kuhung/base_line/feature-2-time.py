# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

print "Import package OK!"

print "Reading data..."
train = pd.read_table('../train/subsidy_train.txt',sep=',',header=-1)
train.columns = ['id','label']
test = pd.read_table('../test/studentID_test.txt',sep=',',header=-1)
test.columns = ['id']
test['label'] = np.nan

train_test = pd.concat([train,test])
del train
del test

card_train = pd.read_table('../train/card_train.txt',sep=',',header=-1)
card_train.columns = ['id','pos','place','consume','time','price','rest']
card_test = pd.read_table('../test/card_test.txt',sep=',',header=-1)
card_test.columns = ['id','pos','place','consume','time','price','rest']
#
card_train_test = pd.concat([card_train,card_test])
print "All right!"

##release memery
del card_train
del card_test

##
#card_train_test=card_train_test.drop_duplicates()

card_train_test.time = pd.to_datetime(card_train_test.time, format='%Y/%m/%d %H:%M:%S')
card_train_test['month'] = card_train_test.time.dt.month
card_train_test['weekday'] = card_train_test.time.dt.weekday
card_train_test['days_in_month'] = card_train_test.time.dt.day
card_train_test['hour'] = card_train_test.time.dt.hour

#card_train_test.drop(['time'],axis=1,inplace=True)
#card_train_test.drop(['place','time'],axis=1,inplace=True)

print "Feature beginning..."


for m in range(1,13):
    feature=card_train_test[(card_train_test.month == m)&(card_train_test.pos=='POS消费')] 
    if feature.empty:
        pass
    else:
        card = pd.DataFrame(feature.groupby(['id'])['pos'].count())
                    
        card['price_sumM%d'%m] = feature.groupby(['id'])['price'].sum()
        card['price_avgM%d'%m] = feature.groupby(['id'])['price'].mean()
        card['price_maxM%d'%m] = feature.groupby(['id'])['price'].max()
        card['price_minM%d'%m] = feature.groupby(['id'])['price'].min()
        card['price_medianM%d'%m] = feature.groupby(['id'])['price'].median()

        card['rest_sumM%d'%m] = feature.groupby(['id'])['rest'].sum()
        card['rest_avgM%d'%m] = feature.groupby(['id'])['rest'].mean()
        card['rest_maxM%d'%m] = feature.groupby(['id'])['rest'].max()
        card['rest_minM%d'%m] = feature.groupby(['id'])['rest'].min()
        card['rest_medianM%d'%m] = feature.groupby(['id'])['rest'].median() 

        del feature
        
        card.to_csv('../input/featureM%d.csv'%m,index=True)
        card = pd.read_csv('../input/featureM%d.csv'%m) 
        card=card.rename(columns={'pos' : 'countM%d'%m}) 
        train_test = pd.merge(train_test, card, how='left',on='id')
        del card

## hours <7|7|8|9|17|18|19|19>

feature=card_train_test[(card_train_test.hour<7)&(card_train_test.pos=='POS消费')] 
if feature.empty:
    pass
else:
    card = pd.DataFrame(feature.groupby(['id'])['pos'].count())
                    
    card['price_sumH7-'] = feature.groupby(['id'])['price'].sum()
    card['price_avgH7-'] = feature.groupby(['id'])['price'].mean()
    card['price_maxH7-'] = feature.groupby(['id'])['price'].max()
    card['price_minH7-'] = feature.groupby(['id'])['price'].min()
    card['price_medianH7-'] = feature.groupby(['id'])['price'].median()

    card['rest_sumH7-'] = feature.groupby(['id'])['rest'].sum()
    card['rest_avgH7-'] = feature.groupby(['id'])['rest'].mean()
    card['rest_maxH7-'] = feature.groupby(['id'])['rest'].max()
    card['rest_minH7-'] = feature.groupby(['id'])['rest'].min()
    card['rest_medianH7-'] = feature.groupby(['id'])['rest'].median()

    del feature
            
    card.to_csv('../input/featureH7-.csv',index=True)
    card = pd.read_csv('../input/featureH7-.csv') 
    card=card.rename(columns={'pos' : 'countH7-'}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card


h=7
feature=card_train_test[(card_train_test.hour== h)&(card_train_test.pos=='POS消费')] 
if feature.empty:
    pass
else:
    card = pd.DataFrame(feature.groupby(['id'])['pos'].count())
                    
    card['price_sumH%d'%h] = feature.groupby(['id'])['price'].sum()
    card['price_avgH%d'%h] = feature.groupby(['id'])['price'].mean()
    card['price_maxH%d'%h] = feature.groupby(['id'])['price'].max()
    card['price_minH%d'%h] = feature.groupby(['id'])['price'].min()
    card['price_medianH%d'%h] = feature.groupby(['id'])['price'].median()

    card['rest_sumH%d'%h] = feature.groupby(['id'])['rest'].sum()
    card['rest_avgH%d'%h] = feature.groupby(['id'])['rest'].mean()
    card['rest_maxH%d'%h] = feature.groupby(['id'])['rest'].max()
    card['rest_minH%d'%h] = feature.groupby(['id'])['rest'].min()
    card['rest_medianH%d'%h] = feature.groupby(['id'])['rest'].median()

    del feature
            
    card.to_csv('../input/featureH%d.csv'%h,index=True)
    card = pd.read_csv('../input/featureH%d.csv'%h) 
    card=card.rename(columns={'pos' : 'countH%d'%h}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card

h=8
feature=card_train_test[(card_train_test.hour== h)&(card_train_test.pos=='POS消费')] 
if feature.empty:
    pass
else:
    card = pd.DataFrame(feature.groupby(['id'])['pos'].count())
                    
    card['price_sumH%d'%h] = feature.groupby(['id'])['price'].sum()
    card['price_avgH%d'%h] = feature.groupby(['id'])['price'].mean()
    card['price_maxH%d'%h] = feature.groupby(['id'])['price'].max()
    card['price_minH%d'%h] = feature.groupby(['id'])['price'].min()
    card['price_medianH%d'%h] = feature.groupby(['id'])['price'].median()

    card['rest_sumH%d'%h] = feature.groupby(['id'])['rest'].sum()
    card['rest_avgH%d'%h] = feature.groupby(['id'])['rest'].mean()
    card['rest_maxH%d'%h] = feature.groupby(['id'])['rest'].max()
    card['rest_minH%d'%h] = feature.groupby(['id'])['rest'].min()
    card['rest_medianH%d'%h] = feature.groupby(['id'])['rest'].median()

    del feature
            
    card.to_csv('../input/featureH%d.csv'%h,index=True)
    card = pd.read_csv('../input/featureH%d.csv'%h) 
    card=card.rename(columns={'pos' : 'countH%d'%h}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card
                
h=9
feature=card_train_test[(card_train_test.hour== h)&(card_train_test.pos=='POS消费')] 
if feature.empty:
    pass
else:
    card = pd.DataFrame(feature.groupby(['id'])['pos'].count())
                    
    card['price_sumH%d'%h] = feature.groupby(['id'])['price'].sum()
    card['price_avgH%d'%h] = feature.groupby(['id'])['price'].mean()
    card['price_maxH%d'%h] = feature.groupby(['id'])['price'].max()
    card['price_minH%d'%h] = feature.groupby(['id'])['price'].min()
    card['price_medianH%d'%h] = feature.groupby(['id'])['price'].median()

    card['rest_sumH%d'%h] = feature.groupby(['id'])['rest'].sum()
    card['rest_avgH%d'%h] = feature.groupby(['id'])['rest'].mean()
    card['rest_maxH%d'%h] = feature.groupby(['id'])['rest'].max()
    card['rest_minH%d'%h] = feature.groupby(['id'])['rest'].min()
    card['rest_medianH%d'%h] = feature.groupby(['id'])['rest'].median()

    del feature
            
    card.to_csv('../input/featureH%d.csv'%h,index=True)
    card = pd.read_csv('../input/featureH%d.csv'%h) 
    card=card.rename(columns={'pos' : 'countH%d'%h}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card
                
h=17
feature=card_train_test[(card_train_test.hour== h)&(card_train_test.pos=='POS消费')] 
if feature.empty:
    pass
else:
    card = pd.DataFrame(feature.groupby(['id'])['pos'].count())
                    
    card['price_sumH%d'%h] = feature.groupby(['id'])['price'].sum()
    card['price_avgH%d'%h] = feature.groupby(['id'])['price'].mean()
    card['price_maxH%d'%h] = feature.groupby(['id'])['price'].max()
    card['price_minH%d'%h] = feature.groupby(['id'])['price'].min()
    card['price_medianH%d'%h] = feature.groupby(['id'])['price'].median()

    card['rest_sumH%d'%h] = feature.groupby(['id'])['rest'].sum()
    card['rest_avgH%d'%h] = feature.groupby(['id'])['rest'].mean()
    card['rest_maxH%d'%h] = feature.groupby(['id'])['rest'].max()
    card['rest_minH%d'%h] = feature.groupby(['id'])['rest'].min()
    card['rest_medianH%d'%h] = feature.groupby(['id'])['rest'].median()

    del feature
            
    card.to_csv('../input/featureH%d.csv'%h,index=True)
    card = pd.read_csv('../input/featureH%d.csv'%h) 
    card=card.rename(columns={'pos' : 'countH%d'%h}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card
                
h=18
feature=card_train_test[(card_train_test.hour== h)&(card_train_test.pos=='POS消费')] 
if feature.empty:
    pass
else:
    card = pd.DataFrame(feature.groupby(['id'])['pos'].count())
                    
    card['price_sumH%d'%h] = feature.groupby(['id'])['price'].sum()
    card['price_avgH%d'%h] = feature.groupby(['id'])['price'].mean()
    card['price_maxH%d'%h] = feature.groupby(['id'])['price'].max()
    card['price_minH%d'%h] = feature.groupby(['id'])['price'].min()
    card['price_medianH%d'%h] = feature.groupby(['id'])['price'].median()

    card['rest_sumH%d'%h] = feature.groupby(['id'])['rest'].sum()
    card['rest_avgH%d'%h] = feature.groupby(['id'])['rest'].mean()
    card['rest_maxH%d'%h] = feature.groupby(['id'])['rest'].max()
    card['rest_minH%d'%h] = feature.groupby(['id'])['rest'].min()
    card['rest_medianH%d'%h] = feature.groupby(['id'])['rest'].median()

    del feature
            
    card.to_csv('../input/featureH%d.csv'%h,index=True)
    card = pd.read_csv('../input/featureH%d.csv'%h) 
    card=card.rename(columns={'pos' : 'countH%d'%h}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card
                
h=19
feature=card_train_test[(card_train_test.hour== h)&(card_train_test.pos=='POS消费')] 
if feature.empty:
    pass
else:
    card = pd.DataFrame(feature.groupby(['id'])['pos'].count())
                    
    card['price_sumH%d'%h] = feature.groupby(['id'])['price'].sum()
    card['price_avgH%d'%h] = feature.groupby(['id'])['price'].mean()
    card['price_maxH%d'%h] = feature.groupby(['id'])['price'].max()
    card['price_minH%d'%h] = feature.groupby(['id'])['price'].min()
    card['price_medianH%d'%h] = feature.groupby(['id'])['price'].median()

    card['rest_sumH%d'%h] = feature.groupby(['id'])['rest'].sum()
    card['rest_avgH%d'%h] = feature.groupby(['id'])['rest'].mean()
    card['rest_maxH%d'%h] = feature.groupby(['id'])['rest'].max()
    card['rest_minH%d'%h] = feature.groupby(['id'])['rest'].min()
    card['rest_medianH%d'%h] = feature.groupby(['id'])['rest'].median()

    del feature
            
    card.to_csv('../input/featureH%d.csv'%h,index=True)
    card = pd.read_csv('../input/featureH%d.csv'%h) 
    card=card.rename(columns={'pos' : 'countH%d'%h}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card
                

feature=card_train_test[(card_train_test.hour>19)&(card_train_test.pos=='POS消费')] 
if feature.empty:
    pass
else:
    card = pd.DataFrame(feature.groupby(['id'])['pos'].count())
                    
    card['price_sumH19+'] = feature.groupby(['id'])['price'].sum()
    card['price_avgH19+'] = feature.groupby(['id'])['price'].mean()
    card['price_maxH19+'] = feature.groupby(['id'])['price'].max()
    card['price_minH19+'] = feature.groupby(['id'])['price'].min()
    card['price_medianH19+'] = feature.groupby(['id'])['price'].median()

    card['rest_sumH19+'] = feature.groupby(['id'])['rest'].sum()
    card['rest_avgH19+'] = feature.groupby(['id'])['rest'].mean()
    card['rest_maxH19+'] = feature.groupby(['id'])['rest'].max()
    card['rest_minH19+'] = feature.groupby(['id'])['rest'].min()
    card['rest_medianH19+'] = feature.groupby(['id'])['rest'].median()

    del feature
            
    card.to_csv('../input/featureH19+.csv',index=True)
    card = pd.read_csv('../input/featureH19+.csv') 
    card=card.rename(columns={'pos' : 'countH19+'}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card

## week

feature=card_train_test[(card_train_test.weekday<5)&(card_train_test.pos=='POS消费')] 
if feature.empty:
    pass
else:
    card = pd.DataFrame(feature.groupby(['id'])['pos'].count())
                    
    card['price_sumWD'] = feature.groupby(['id'])['price'].sum()
    card['price_avgWD'] = feature.groupby(['id'])['price'].mean()
    card['price_maxWD'] = feature.groupby(['id'])['price'].max()
    card['price_minWD'] = feature.groupby(['id'])['price'].min()
    card['price_medianWD'] = feature.groupby(['id'])['price'].median()

    card['rest_sumWD'] = feature.groupby(['id'])['rest'].sum()
    card['rest_avgWD'] = feature.groupby(['id'])['rest'].mean()
    card['rest_maxWD'] = feature.groupby(['id'])['rest'].max()
    card['rest_minWD'] = feature.groupby(['id'])['rest'].min()
    card['rest_medianWD'] = feature.groupby(['id'])['rest'].median()

    del feature
            
    card.to_csv('../input/featureWD.csv',index=True)
    card = pd.read_csv('../input/featureWD.csv') 
    card=card.rename(columns={'pos' : 'countWD'}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card


feature=card_train_test[(card_train_test.weekday>=5)&(card_train_test.pos=='POS消费')] 
if feature.empty:
    pass
else:
    card = pd.DataFrame(feature.groupby(['id'])['pos'].count())
                    
    card['price_sumWE'] = feature.groupby(['id'])['price'].sum()
    card['price_avgWE'] = feature.groupby(['id'])['price'].mean()
    card['price_maxWE'] = feature.groupby(['id'])['price'].max()
    card['price_minWE'] = feature.groupby(['id'])['price'].min()
    card['price_medianWE'] = feature.groupby(['id'])['price'].median()

    card['rest_sumWE'] = feature.groupby(['id'])['rest'].sum()
    card['rest_avgWE'] = feature.groupby(['id'])['rest'].mean()
    card['rest_maxWE'] = feature.groupby(['id'])['rest'].max()
    card['rest_minWE'] = feature.groupby(['id'])['rest'].min()
    card['rest_medianWE'] = feature.groupby(['id'])['rest'].median()

    del feature
            
    card.to_csv('../input/featureWE.csv',index=True)
    card = pd.read_csv('../input/featureWE.csv') 
    card=card.rename(columns={'pos' : 'countWE'}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card



print "OK!"
##release memery
del card_train_test

print "Feature two end."

