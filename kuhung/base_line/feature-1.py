# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np


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

card_train_test.drop(['place','time'],axis=1,inplace=True)

print "Feature engine begin..."


## bash feature
card = pd.DataFrame(card_train_test.groupby(['id'])['pos'].count())

card['price_sum'] = card_train_test.groupby(['id'])['price'].sum()
card['price_avg'] = card_train_test.groupby(['id'])['price'].mean()
card['price_max'] = card_train_test.groupby(['id'])['price'].max()
card['price_min'] = card_train_test.groupby(['id'])['price'].min()
card['price_median'] = card_train_test.groupby(['id'])['price'].median()

card['rest_sum'] = card_train_test.groupby(['id'])['rest'].sum()
card['rest_avg'] = card_train_test.groupby(['id'])['rest'].mean()
card['rest_max'] = card_train_test.groupby(['id'])['rest'].max()
card['rest_min'] = card_train_test.groupby(['id'])['rest'].min()
card['rest_median'] = card_train_test.groupby(['id'])['rest'].median()

card.to_csv('../input/card_bashfeature.csv',index=True)
card = pd.read_csv('../input/card_bashfeature.csv') 
card=card.rename(columns={'pos' : 'price_count'}) 

train_test = pd.merge(train_test, card, how='left',on='id') #2512
del card

## consume feature
consume=card_train_test[card_train_test.pos == 'POS消费']

card_consume = pd.DataFrame(consume.groupby(['id'])['pos'].count())

card_consume['consume_sum']=consume.groupby(['id'])['price'].sum()
card_consume['consume_avg']=consume.groupby(['id'])['price'].mean()
card_consume['consume_max']=consume.groupby(['id'])['price'].max()
card_consume['consume_min']=consume.groupby(['id'])['price'].min()
card_consume['consume_median']=consume.groupby(['id'])['price'].median()

del consume
card_consume.to_csv('../input/card_consumefeature.csv',index=True)
card_consume = pd.read_csv('../input/card_consumefeature.csv') 
card_consume=card_consume.rename(columns={'pos' : 'consume_count'}) 

train_test = pd.merge(train_test, card_consume, how='left',on='id') 
del card_consume


kapiankaihu=card_train_test[card_train_test.pos=='卡片开户']
card_kaihu = pd.DataFrame(kapiankaihu.groupby(['id'])['pos'].count())
del kapiankaihu
card_kaihu.to_csv('../input/card_kaihufeature.csv',index=True)
card_kaihu = pd.read_csv('../input/card_kaihufeature.csv') 
card_kaihu=card_kaihu.rename(columns={'pos' : 'kaihu_count'}) 

train_test = pd.merge(train_test, card_kaihu, how='left',on='id') 
del card_kaihu

kapianxiaohu=card_train_test[card_train_test.pos=='卡片销户']
card_xiaohu = pd.DataFrame(kapianxiaohu.groupby(['id'])['pos'].count())
del kapianxiaohu
card_xiaohu.to_csv('../input/card_xiaohufeature.csv',index=True)
card_xiaohu = pd.read_csv('../input/card_xiaohufeature.csv') 
card_xiaohu=card_xiaohu.rename(columns={'pos' : 'xiaohu_count'}) 

train_test = pd.merge(train_test, card_xiaohu, how='left',on='id') 
del card_xiaohu

kapianbuban=card_train_test[card_train_test.pos=='卡补办']
card_buban = pd.DataFrame(kapianbuban.groupby(['id'])['pos'].count())
del kapianbuban
card_buban.to_csv('../input/card_bubanfeature.csv',index=True)
card_buban = pd.read_csv('../input/card_bubanfeature.csv') 
card_buban=card_buban.rename(columns={'pos' : 'buban_count'}) 

train_test = pd.merge(train_test, card_buban, how='left',on='id') 
del card_buban

kapianjiegua=card_train_test[card_train_test.pos=='卡解挂']
card_jiegua = pd.DataFrame(kapianjiegua.groupby(['id'])['pos'].count())
del kapianjiegua
card_jiegua.to_csv('../input/card_jieguafeature.csv',index=True)
card_jiegua = pd.read_csv('../input/card_jieguafeature.csv') 
card_jiegua=card_jiegua.rename(columns={'pos' : 'jiegua_count'}) 

train_test = pd.merge(train_test, card_jiegua, how='left',on='id') 
del card_jiegua

kapianchange=card_train_test[card_train_test.pos=='换卡']
card_change = pd.DataFrame(kapianchange.groupby(['id'])['pos'].count())
del kapianchange
card_change.to_csv('../input/card_changefeature.csv',index=True)
card_change = pd.read_csv('../input/card_changefeature.csv') 
card_change=card_change.rename(columns={'pos' : 'change_count'}) 

train_test = pd.merge(train_test, card_change, how='left',on='id') 
del card_change



canteen=card_train_test[card_train_test.consume=='食堂']

card_canteen = pd.DataFrame(canteen.groupby(['id'])['pos'].count())
card_canteen['canteen_sum']=canteen.groupby(['id'])['price'].sum()
card_canteen['canteen_avg']=canteen.groupby(['id'])['price'].mean()
card_canteen['canteen_max']=canteen.groupby(['id'])['price'].max()
card_canteen['canteen_min']=canteen.groupby(['id'])['price'].min()
card_canteen['canteen_median']=canteen.groupby(['id'])['price'].median()

del canteen
card_canteen.to_csv('../input/card_canteenfeature.csv',index=True)
card_canteen = pd.read_csv('../input/card_canteenfeature.csv') 
card_canteen=card_canteen.rename(columns={'pos' : 'canteen_count'}) 

train_test = pd.merge(train_test, card_canteen, how='left',on='id') 
del card_canteen

boiled_water=card_train_test[card_train_test.consume=='开水']

card_boiled_water = pd.DataFrame(boiled_water.groupby(['id'])['pos'].count())
card_boiled_water['boiled_water_sum']=boiled_water.groupby(['id'])['price'].sum()
card_boiled_water['boiled_water_avg']=boiled_water.groupby(['id'])['price'].mean()
card_boiled_water['boiled_water_max']=boiled_water.groupby(['id'])['price'].max()
card_boiled_water['boiled_water_min']=boiled_water.groupby(['id'])['price'].min()
card_boiled_water['boiled_water_median']=boiled_water.groupby(['id'])['price'].median()

del boiled_water
card_boiled_water.to_csv('../input/card_boiled_waterfeature.csv',index=True)
card_boiled_water = pd.read_csv('../input/card_boiled_waterfeature.csv') 
card_boiled_water=card_boiled_water.rename(columns={'pos' : 'boiled_water_count'}) 

train_test = pd.merge(train_test, card_boiled_water, how='left',on='id') 
del card_boiled_water

bathe=card_train_test[card_train_test.consume=='淋浴']

card_bathe = pd.DataFrame(bathe.groupby(['id'])['pos'].count())
card_bathe['bathe_sum']=bathe.groupby(['id'])['price'].sum()
card_bathe['bathe_avg']=bathe.groupby(['id'])['price'].mean()
card_bathe['bathe_max']=bathe.groupby(['id'])['price'].max()
card_bathe['bathe_min']=bathe.groupby(['id'])['price'].min()
card_bathe['bathe_median']=bathe.groupby(['id'])['price'].median()

del bathe
card_bathe.to_csv('../input/card_bathefeature.csv',index=True)
card_bathe = pd.read_csv('../input/card_bathefeature.csv') 
card_bathe=card_bathe.rename(columns={'pos' : 'bathe_count'}) 

train_test = pd.merge(train_test, card_bathe, how='left',on='id') 
del card_bathe

shool_bus=card_train_test[card_train_test.consume=='校车']

card_shool_bus = pd.DataFrame(shool_bus.groupby(['id'])['pos'].count())
card_shool_bus['shool_bus_sum']=shool_bus.groupby(['id'])['price'].sum()
card_shool_bus['shool_bus_avg']=shool_bus.groupby(['id'])['price'].mean()
card_shool_bus['shool_bus_max']=shool_bus.groupby(['id'])['price'].max()
card_shool_bus['shool_bus_min']=shool_bus.groupby(['id'])['price'].min()
card_shool_bus['shool_bus_median']=shool_bus.groupby(['id'])['price'].median()

del shool_bus
card_shool_bus.to_csv('../input/card_shool_busfeature.csv',index=True)
card_shool_bus = pd.read_csv('../input/card_shool_busfeature.csv') 
card_shool_bus=card_shool_bus.rename(columns={'pos' : 'shool_bus_count'}) 

train_test = pd.merge(train_test, card_shool_bus, how='left',on='id') 
del card_shool_bus


shop=card_train_test[card_train_test.consume=='超市']

card_shop = pd.DataFrame(shop.groupby(['id'])['pos'].count())
card_shop['shop_sum']=shop.groupby(['id'])['price'].sum()
card_shop['shop_avg']=shop.groupby(['id'])['price'].mean()
card_shop['shop_max']=shop.groupby(['id'])['price'].max()
card_shop['shop_min']=shop.groupby(['id'])['price'].min()
card_shop['shop_median']=shop.groupby(['id'])['price'].median()

del shop
card_shop.to_csv('../input/card_shopfeature.csv',index=True)
card_shop = pd.read_csv('../input/card_shopfeature.csv') 
card_shop=card_shop.rename(columns={'pos' : 'shop_count'}) 

train_test = pd.merge(train_test, card_shop, how='left',on='id') 
del card_shop


wash_house=card_train_test[card_train_test.consume=='洗衣房']

card_wash_house = pd.DataFrame(wash_house.groupby(['id'])['pos'].count())
card_wash_house['wash_sum']=wash_house.groupby(['id'])['price'].sum()
card_wash_house['wash_avg']=wash_house.groupby(['id'])['price'].mean()
card_wash_house['wash_max']=wash_house.groupby(['id'])['price'].max()
card_wash_house['wash_min']=wash_house.groupby(['id'])['price'].min()
card_wash_house['wash_median']=wash_house.groupby(['id'])['price'].median()

del wash_house
card_wash_house.to_csv('../input/card_wash_housefeature.csv',index=True)
card_wash_house = pd.read_csv('../input/card_wash_housefeature.csv') 
card_wash_house=card_wash_house.rename(columns={'pos' : 'wash_house_count'}) 

train_test = pd.merge(train_test, card_wash_house, how='left',on='id') 
del card_wash_house


library=card_train_test[card_train_test.consume=='图书馆']

card_library= pd.DataFrame(library.groupby(['id'])['pos'].count())
card_library['library_sum']=library.groupby(['id'])['price'].sum()
card_library['library_avg']=library.groupby(['id'])['price'].mean()
card_library['library_max']=library.groupby(['id'])['price'].max()
card_library['library_min']=library.groupby(['id'])['price'].min()
card_library['library_median']=library.groupby(['id'])['price'].median()

del library
card_library.to_csv('../input/card_libraryfeature.csv',index=True)
card_library = pd.read_csv('../input/card_libraryfeature.csv') 
card_library=card_library.rename(columns={'pos' : 'library_count'}) 

train_test = pd.merge(train_test, card_library, how='left',on='id') 
del card_library

printhouse=card_train_test[card_train_test.consume=='文印中心']

card_printhouse= pd.DataFrame(printhouse.groupby(['id'])['pos'].count())
card_printhouse['print_sum']=printhouse.groupby(['id'])['price'].sum()
card_printhouse['print_avg']=printhouse.groupby(['id'])['price'].mean()
card_printhouse['print_max']=printhouse.groupby(['id'])['price'].max()
card_printhouse['print_min']=printhouse.groupby(['id'])['price'].min()
card_printhouse['print_median']=printhouse.groupby(['id'])['price'].median()

del printhouse
card_printhouse.to_csv('../input/card_printhousefeature.csv',index=True)
card_printhouse = pd.read_csv('../input/card_printhousefeature.csv') 
card_printhouse=card_printhouse.rename(columns={'pos' : 'printhouse_count'}) 

train_test = pd.merge(train_test, card_printhouse, how='left',on='id') 
del card_printhouse

dean=card_train_test[card_train_test.consume=='教务处']

card_dean= pd.DataFrame(dean.groupby(['id'])['pos'].count())
card_dean['dean_sum']=dean.groupby(['id'])['price'].sum()
card_dean['dean_avg']=dean.groupby(['id'])['price'].mean()
card_dean['dean_max']=dean.groupby(['id'])['price'].max()
card_dean['dean_min']=dean.groupby(['id'])['price'].min()
card_dean['dean_median']=dean.groupby(['id'])['price'].median()

del dean
card_dean.to_csv('../input/card_deanfeature.csv',index=True)
card_dean = pd.read_csv('../input/card_deanfeature.csv') 
card_dean=card_dean.rename(columns={'pos' : 'dean_count'}) 

train_test = pd.merge(train_test, card_dean, how='left',on='id') 
del card_dean

other=card_train_test[card_train_test.consume=='其他']

card_other= pd.DataFrame(other.groupby(['id'])['pos'].count())
card_other['other_sum']=other.groupby(['id'])['price'].sum()
#
card_other['other_avg']=other.groupby(['id'])['price'].mean()
#
card_other['other_max']=other.groupby(['id'])['price'].max()
#
card_other['other_min']=other.groupby(['id'])['price'].min()
#
card_other['other_median']=other.groupby(['id'])['price'].median()

del other
card_other.to_csv('../input/card_otherfeature.csv',index=True)
card_other = pd.read_csv('../input/card_otherfeature.csv') 
card_other=card_other.rename(columns={'pos' : 'other_count'}) 

train_test = pd.merge(train_test, card_other, how='left',on='id') 
del card_other

hospital=card_train_test[card_train_test.consume=='校医院']

card_hospital= pd.DataFrame(hospital.groupby(['id'])['pos'].count())
card_hospital['hospital_sum']=hospital.groupby(['id'])['price'].sum()
card_hospital['hospital_avg']=hospital.groupby(['id'])['price'].mean()
#
card_hospital['hospital_max']=hospital.groupby(['id'])['price'].max()
card_hospital['hospital_min']=hospital.groupby(['id'])['price'].min()
#
card_hospital['hospital_median']=hospital.groupby(['id'])['price'].median()

del hospital
card_hospital.to_csv('../input/card_hospitalfeature.csv',index=True)
card_hospital = pd.read_csv('../input/card_hospitalfeature.csv') 
card_hospital=card_hospital.rename(columns={'pos' : 'hospital_count'}) 

train_test = pd.merge(train_test, card_hospital, how='left',on='id') 
del card_hospital


print "OK!"
##release memery
del card_train_test

print "Feature one end."
