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

score_train = pd.read_table('../train/score_train.txt',sep=',',header=-1)
score_train.columns = ['id','college','rank']
score_test = pd.read_table('../test/score_test.txt',sep=',',header=-1)
score_test.columns = ['id','college','rank']
score_train_test = pd.concat([score_train,score_test])
del score_train
del score_test

college = pd.DataFrame(score_train_test.groupby(['college'])['rank'].max())
college.to_csv('../input/college.csv',index=True)
college = pd.read_csv('../input/college.csv')
college.columns = ['college','total_people']
score_train_test = pd.merge(score_train_test, college, how='left',on='college')
del college

score_train_test['rank_percent'] = score_train_test['rank']/score_train_test['total_people']
train_test = pd.merge(train_test,score_train_test,how='left',on='id')
del score_train_test
print "All right!"


print "Merge beginning..."

for m in range(1,13):
    card = pd.read_csv('../input/featureM%d.csv'%m) 
    card=card.rename(columns={'pos' : 'countM%d'%m}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card

## hours <7|7|8|9|17|18|19|19>

card = pd.read_csv('../input/featureH7-.csv') 
card=card.rename(columns={'pos' : 'countH7-'}) 
train_test = pd.merge(train_test, card, how='left',on='id')
del card


for h in [7,8,9,17,18,19]:        
    card = pd.read_csv('../input/featureH%d.csv'%h) 
    card=card.rename(columns={'pos' : 'countH%d'%h}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card

card = pd.read_csv('../input/featureH19+.csv') 
card=card.rename(columns={'pos' : 'countH19+'}) 
train_test = pd.merge(train_test, card, how='left',on='id')
del card

## week
card = pd.read_csv('../input/featureWD.csv') 
train_test = pd.merge(train_test, card, how='left',on='id')
card=card.rename(columns={'pos' : 'countWD'}) 
del card

card = pd.read_csv('../input/featureWE.csv')     
card=card.rename(columns={'pos' : 'countWE'}) 
train_test = pd.merge(train_test, card, how='left',on='id')
del card


## bash feature
card = pd.read_csv('../input/card_bashfeature.csv') 
card=card.rename(columns={'pos' : 'price_count'}) 
train_test = pd.merge(train_test, card, how='left',on='id') #2512
del card


## consume feature
card_consume = pd.read_csv('../input/card_consumefeature.csv') 
card_consume=card_consume.rename(columns={'pos' : 'consume_count'}) 
train_test = pd.merge(train_test, card_consume, how='left',on='id') 
del card_consume


card_kaihu = pd.read_csv('../input/card_kaihufeature.csv') 
card_kaihu=card_kaihu.rename(columns={'pos' : 'kaihu_count'}) 
train_test = pd.merge(train_test, card_kaihu, how='left',on='id') 
del card_kaihu

card_xiaohu = pd.read_csv('../input/card_xiaohufeature.csv') 
card_xiaohu=card_xiaohu.rename(columns={'pos' : 'xiaohu_count'}) 
train_test = pd.merge(train_test, card_xiaohu, how='left',on='id') 
del card_xiaohu

card_buban = pd.read_csv('../input/card_bubanfeature.csv') 
card_buban=card_buban.rename(columns={'pos' : 'buban_count'}) 
train_test = pd.merge(train_test, card_buban, how='left',on='id') 
del card_buban

card_jiegua = pd.read_csv('../input/card_jieguafeature.csv') 
card_jiegua=card_jiegua.rename(columns={'pos' : 'jiegua_count'}) 
train_test = pd.merge(train_test, card_jiegua, how='left',on='id') 
del card_jiegua

card_change = pd.read_csv('../input/card_changefeature.csv') 
card_change=card_change.rename(columns={'pos' : 'change_count'}) 
train_test = pd.merge(train_test, card_change, how='left',on='id') 
del card_change

## place infor
card_canteen = pd.read_csv('../input/card_canteenfeature.csv')
card_canteen=card_canteen.rename(columns={'pos' : 'canteen_count'}) 
train_test = pd.merge(train_test, card_canteen, how='left',on='id') 
del card_canteen

card_boiled_water = pd.read_csv('../input/card_boiled_waterfeature.csv') 
card_boiled_water=card_boiled_water.rename(columns={'pos' : 'boiled_water_count'}) 
train_test = pd.merge(train_test, card_boiled_water, how='left',on='id') 
del card_boiled_water

card_bathe = pd.read_csv('../input/card_bathefeature.csv') 
card_bathe=card_bathe.rename(columns={'pos' : 'bathe_count'}) 
train_test = pd.merge(train_test, card_bathe, how='left',on='id') 
del card_bathe


card_shool_bus = pd.read_csv('../input/card_shool_busfeature.csv') 
card_shool_bus=card_shool_bus.rename(columns={'pos' : 'shool_bus_count'}) 
train_test = pd.merge(train_test, card_shool_bus, how='left',on='id') 
del card_shool_bus


card_shop = pd.read_csv('../input/card_shopfeature.csv') 
card_shop=card_shop.rename(columns={'pos' : 'shop_count'}) 
train_test = pd.merge(train_test, card_shop, how='left',on='id') 
del card_shop

card_wash_house = pd.read_csv('../input/card_wash_housefeature.csv') 
card_wash_house=card_wash_house.rename(columns={'pos' : 'wash_house_count'}) 
train_test = pd.merge(train_test, card_wash_house, how='left',on='id') 
del card_wash_house


card_library = pd.read_csv('../input/card_libraryfeature.csv') 
card_library=card_library.rename(columns={'pos' : 'library_count'}) 
train_test = pd.merge(train_test, card_library, how='left',on='id') 
del card_library


card_printhouse = pd.read_csv('../input/card_printhousefeature.csv') 
card_printhouse=card_printhouse.rename(columns={'pos' : 'printhouse_count'}) 
train_test = pd.merge(train_test, card_printhouse, how='left',on='id') 
del card_printhouse


card_dean = pd.read_csv('../input/card_deanfeature.csv') 
card_dean=card_dean.rename(columns={'pos' : 'dean_count'}) 
train_test = pd.merge(train_test, card_dean, how='left',on='id') 
del card_dean

card_other = pd.read_csv('../input/card_otherfeature.csv') 
card_other=card_other.rename(columns={'pos' : 'other_count'}) 
train_test = pd.merge(train_test, card_other, how='left',on='id') 
del card_other


card_hospital = pd.read_csv('../input/card_hospitalfeature.csv') 
card_hospital=card_hospital.rename(columns={'pos' : 'hospital_count'}) 
train_test = pd.merge(train_test, card_hospital, how='left',on='id') 
del card_hospital


for var in ['地点21','地点829','地点818','地点213','地点72','地点283','地点91','地点245','地点65','地点161','地点996','地点277','地点842','地点75','地点263','地点840']:

    feature_p=pd.read_csv('../input/card_%sfeature.csv'%var)
    feature_p=feature_p.rename(columns={'pos' : '%s_count'%var})
    train_test = pd.merge(train_test, feature_p, how='left',on='id') 
    del feature_p


print "Merge all right."


train = train_test[train_test['label'].notnull()]
test = train_test[train_test['label'].isnull()]

train = train.fillna(-1)
test = test.fillna(-1)

train.to_csv('../input/train_time.csv',index=False)
test.to_csv('../input/test_time.csv',index=False)


train = pd.read_csv('../input/train_time.csv')
test = pd.read_csv('../input/test_time.csv')

#train=train[train['id']<22500]

nice_feature=pd.read_csv('../input/nice_feature.csv',header=None,index_col=0)
feature_imp_place20=pd.read_csv('../input/feature_imp_place20.csv')



target = 'label'
IDcol = 'id'
ids = test['id'].values

all_feature = [x for x in train.columns if x not in [target,IDcol]]
#predictors = [x for x in train.columns if x in all_feature]
predictors = [ x for x in all_feature if (x in nice_feature.index)|(x in feature_imp_place20.feature.values)]
#predictors = [ x for x in all_feature if (x in nice_feature.index)]


# Oversample
Oversampling1000 = train.loc[train.label == 1000]
Oversampling1500 = train.loc[train.label == 1500]
Oversampling2000 = train.loc[train.label == 2000]
for i in range(7):
    train = train.append(Oversampling1000)
for j in range(10):
    train = train.append(Oversampling1500)
for k in range(9):
    train = train.append(Oversampling2000)

from xgboost import XGBClassifier

'''
# ensemble
from sklearn import ensemble

clf1 = XGBClassifier(max_depth=4,objective='multi:softmax',n_estimators=100,seed=42)
clf2 = XGBClassifier(max_depth=3,objective='multi:softmax',n_estimators=100,seed=42)
clf3 = XGBClassifier(max_depth=4,objective='multi:softmax',n_estimators=100,seed=0)
clf4 = XGBClassifier(max_depth=3,objective='multi:softmax',n_estimators=100,seed=0)
#clf5 = XGBClassifier(max_depth=2,objective='multi:softmax',n_estimators=100,seed=42)


#clf3 = GradientBoostingClassifier(n_estimators=200,random_state=2016)
#clf4 = GradientBoostingClassifier(n_estimators=200,random_state=42)
#clf5 = GradientBoostingClassifier(n_estimators=250,random_state=2016)

clfs=ensemble.VotingClassifier(estimators=[('xgb1',clf1),('xgb2',clf2),('GBM1',clf3),('GBM2',clf4)],voting='soft')

clfs = clfs.fit(train[predictors],train[target])
result = clfs.predict(test[predictors])

'''
# model

param_dist = {
    'n_estimators': 60,
    'max_depth': 5,
    'learning_rate':0.2,
    'min_child_weight':2,
    'gamma':0,
    'subsample':1,
    'colsample_bytree':1,
    'reg_alpha':0.008}
'''
param_dist = {
    'n_estimators': 120,
    'max_depth': 7,
    'min_child_weight':7,
    'gamma':0,
    'subsample':0.4,
    'colsample_bytree':1, 
    'reg_alpha':0.1,
    'learning_rate':0.05}
'''
clf = XGBClassifier(**param_dist).fit(train[predictors],train[target])
#clf=XGBClassifier(max_depth=4,objective='multi:softmax',n_estimators=200,seed=42,learning_rate=0.05)
#clf = GradientBoostingClassifier(n_estimators=200,random_state=2016)
#clf = RandomForestClassifier(n_estimators=500,random_state=2016)


#from sklearn.ensemble import BaggingClassifier
#bg_clf = BaggingClassifier(clfs, max_samples=0.5, max_features=0.5)

#bg_clf = bg_clf.fit(train[predictors],train[target])
#result = bg_clf.predict(test[predictors])

#clf = clf.fit(train[predictors],train[target])
result = clf.predict(test[predictors])

#clf = clf.fit(np.log(train[predictors]),train[target])
#result = clf.predict(np.log(test[predictors]))


# Save results
test_result = pd.DataFrame(columns=["studentid","subsidy"])
test_result.studentid = ids
test_result.subsidy = result
test_result.subsidy = test_result.subsidy.apply(lambda x:int(x))

print '1000--'+str(len(test_result[test_result.subsidy==1000])) + ':741'
print '1500--'+str(len(test_result[test_result.subsidy==1500])) + ':465'
print '2000--'+str(len(test_result[test_result.subsidy==2000])) + ':354'

test_result.to_csv("../output/submitXGBNFeatureNoid.csv",index=False)

