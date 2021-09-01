import numpy as np
import pandas as pd
from __future__ import division, print_function, unicode_literals
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets 
from collections import Counter
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import learning_curve 
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import os
import datetime
from pylab import rcParams
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

df_expedia = pd.read_csv("train.csv")

df_expedia_temp = df_expedia.groupby("srch_id").agg({"booking_bool": np.sum, "srch_id": lambda x: x.nunique()})
df_expedia_temp = df_expedia_temp[df_expedia_temp.booking_bool != 0]
list_of_searh_id_with_booking = df_expedia_temp.iloc[:, np.r_[0,1]]
list_search_id_with_bookings = list_of_searh_id_with_booking.index.tolist()

df_expedia_clean = df_expedia[df_expedia['srch_id'].isin(list_search_id_with_bookings)] 

df_expedia_clean.isnull().sum(axis = 0) 

df = df_expedia_clean

df['date_time'] = pd.to_datetime(df['date_time'])
df['month'] = df['date_time'].dt.month

df['click_bool'] = df['click_bool'].astype('bool')
df['booking_bool'] = df['booking_bool'].astype('bool')
df['visitor_location_country_id'] = df['visitor_location_country_id'].astype('category')
df['prop_country_id'] = df['prop_country_id'].astype('category')
df['prop_id'] = df['prop_id'].astype('category')
df['prop_brand_bool'] = df['prop_brand_bool'].astype('bool')
df['promotion_flag'] = df['promotion_flag'].astype('bool')
df['srch_destination_id'] = df['srch_destination_id'].astype('category')
df['srch_saturday_night_bool'] = df['srch_saturday_night_bool'].astype('bool')
df['month'] = df['month'].astype('category')
df['position'] = df['position'].astype('category')

for k, v in df.nunique().to_dict().items():
    print('{}={}'.format(k,v))


rcParams['figure.figsize'] = 8,8
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

df = df.drop(['date_time', 'position'], axis=1)

col_encode = list(df.select_dtypes(exclude=['number','bool']))

df_encode = df

df_encode[col_encode] = df_encode[col_encode].apply(LabelEncoder().fit_transform)

for col in col_encode:
    df_encode[col] = df_encode[col].astype('category')

df_encode.price_usd += 0.000001
df_encode['log_price'] = np.log(df_encode.price_usd)


df_encode['log_price'].hist(bins=100)

df = df_encode.drop(['price_usd'], axis=1)

sample = pd.Series(df['srch_id'].unique()).sample(60000, random_state = 64)
df1 = df.loc[df['srch_id'].isin(sample)]

from sklearn.model_selection import GroupShuffleSplit

df1.reset_index()['srch_id']
group = GroupShuffleSplit(n_splits=1, test_size=0.3)
train_dataset,test_dataset = next(group.split(X=df1, y=df1['booking_bool'], groups=df1.index.values))

train_id = pd.Series(df1['srch_id'].unique()).sample(42000)
train = df1.loc[df1['srch_id'].isin(train_id)]
test = df1.loc[~df1['srch_id'].isin(train_id)]

train_majority = train[train.click_bool==0]
train_minority = train[train.click_bool==1]

train_majority_downsampled = train_majority.groupby('srch_id').apply(lambda x: x.sample(5, replace = True)).reset_index(drop=True)

train_downsampled = pd.concat([train_minority, train_majority_downsampled])
train_downsampled.booking_bool.value_counts()

X_train = train_downsampled.iloc[:, np.r_[ :19,20,21]]
y_train = train_downsampled.iloc[:,19]
X_test = test.iloc[:, np.r_[ :19,20,21]]
y_test = test.iloc[:,19]


y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

X_train['click_bool'] = X_train['click_bool'].astype('bool')
X_train['visitor_location_country_id'] = X_train['visitor_location_country_id'].astype('category')
X_train['prop_country_id'] = X_train['prop_country_id'].astype('category')
X_train['prop_id'] = X_train['prop_id'].astype('category')
X_train['prop_brand_bool'] = X_train['prop_brand_bool'].astype('bool')
X_train['promotion_flag'] = X_train['promotion_flag'].astype('bool')
X_train['srch_destination_id'] = X_train['srch_destination_id'].astype('category')
X_train['site_id'] = X_train['site_id'].astype('category')
X_train['srch_saturday_night_bool'] = X_train['srch_saturday_night_bool'].astype('bool')
X_train['month'] = X_train['month'].astype('category')

X_test['click_bool'] = X_test['click_bool'].astype('bool')
X_test['visitor_location_country_id'] = X_test['visitor_location_country_id'].astype('category')
X_test['prop_country_id'] = X_test['prop_country_id'].astype('category')
X_test['prop_id'] = X_test['prop_id'].astype('category')
X_test['prop_brand_bool'] = X_test['prop_brand_bool'].astype('bool')
X_test['promotion_flag'] = X_test['promotion_flag'].astype('bool')
X_test['srch_destination_id'] = X_test['srch_destination_id'].astype('category')
X_test['site_id'] = X_test['site_id'].astype('category')
X_test['srch_saturday_night_bool'] = X_test['srch_saturday_night_bool'].astype('bool')
X_test['month'] = X_test['month'].astype('category')


col_encode = list(X_train.select_dtypes(exclude=['number','bool']))

X_train[col_encode] = X_train[col_encode].apply(LabelEncoder().fit_transform)

for col in col_encode:
    X_train[col] = X_train[col].astype('category')
    

X_test[col_encode] = X_test[col_encode].apply(LabelEncoder().fit_transform)

for col in col_encode:
    X_test[col] = X_test[col].astype('category')

click_list = X_test['click_bool']
srchid_list = X_test['srch_id']

X_train = X_train.drop(['click_bool', 'srch_id'], axis=1)

X_test = X_test.drop(['click_bool', 'srch_id'], axis=1)

np.random.seed(42)

gs_dt1 = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [5,7,9,11], 
                               'criterion':['gini','entropy'], 
                              'min_samples_leaf':[4,6,8,10],
                              'min_samples_split':[2,4,6,8,10]}],
                  scoring='roc_auc',
                  cv=5,
                  n_jobs=-1)

gs_dt1 = gs_dt1.fit(X_train,y_train)
print(gs_dt1.best_score_)
print(gs_dt1.best_params_)

dt = DecisionTreeClassifier(criterion= 'gini', max_depth= 7, min_samples_leaf= 4, min_samples_split= 10)
dt_model = dt.fit(X_train, y_train)
y_dt_pred = dt_model.predict_proba(X_test)

from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss

dt_auc = roc_auc_score(y_test,y_dt_pred[:,1])
dt_log_loss = log_loss(y_test, y_dt_pred)


from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report

y_pred = gs_dt1.predict(X_test)
y_pred_insample = gs_dt1.predict(X_train)

from sklearn.metrics import confusion_matrix

Confusion_Matrix = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['Predicted no', 'Predicted yes'],
    index=['True no', 'True yes'])

from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

gs_rm = GridSearchCV(estimator=RandomForestClassifier(random_state=64),
                  param_grid=[{'max_depth': [6,8,10,12], 'criterion':['gini','entropy'], 
                              'min_samples_leaf':[4,5,6,7,8,9],
                              'min_samples_split':[2,3,4,5],
                              'n_estimators':[100]}],
                  scoring='roc_auc', cv=5, n_jobs=-1)

gs_rm = gs_rm.fit(X_train, y_train)

rm_param = gs_rm.best_params_

rm = RandomForestClassifier(**rm_param)

rm_model = rm.fit(X_train, y_train)
y_rm_pred = rm_model.predict_proba(X_test)
rm = RandomForestClassifier(**gs_rm.best_params_)
rm_model = rm.fit(X_train, y_train)
y_rm_pred = rm_model.predict_proba(X_test)

from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss

rm_auc = roc_auc_score(y_test,y_rm_pred[:,1])
rm_log_loss = log_loss(y_test, y_rm_pred)

y_pred = gs_rm.predict(X_test)
from sklearn.metrics import confusion_matrix

Confusion_Matrix = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['Predicted no', 'Predicted yes'],
    index=['True no', 'True yes'])


rm_importance = pd.DataFrame({'feature':X_train.columns,'importance':(rm_model.feature_importances_).ravel()})
rm_importance.sort_values(by='importance', ascending=False)

feats = {}
for feature, importance in zip(X_train.columns, gs_rm.best_estimator_.feature_importances_):
    feats[feature] = importance
    
importances_rf = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance_rf'})
importances_rf = importances_rf.sort_values('Gini-importance_rf', ascending = False)

importances_rf.plot(kind='bar', figsize=(10,5), rot=70)

from sklearn.model_selection import GridSearchCV, cross_val_score

gs_grb = GridSearchCV(estimator=GradientBoostingClassifier(loss='deviance', n_estimators=300, random_state=43),
                  param_grid=[{'learning_rate': [0.05], 'max_depth': [7],'min_samples_leaf': [4], 'min_samples_split': [10]}],
                  scoring='roc_auc', cv=5, n_jobs=-1)

gs_grb = gs_grb.fit(X_train, y_train)

gs_grb = GridSearchCV(estimator=GradientBoostingClassifier(loss='deviance', n_estimators=100, random_state=43),
                  param_grid=[{'learning_rate': [0.1], 'max_depth': [7],'min_samples_leaf': [4], 'min_samples_split': [10]}],
                  scoring='roc_auc', cv=5, n_jobs=-1)

gs_grb = gs_grb.fit(X_train, y_train)

grb = GradientBoostingClassifier(**gs_grb.best_params_)

grb_model = grb.fit(X_train, y_train)
y_grb_pred = grb_model.predict_proba(X_test)

y_pred = pd.DataFrame({'prob':y_grb_pred[:,1],'click_bool':click_list,'booking_bool':y_test, 'srch_id':srchid_list})
y_pred['position'] = 0
y_pred = y_pred.reset_index(0, drop=True)

rank_total = y_pred
rank_total['score'] = rank_total['click_bool']*1 + rank_total['booking_bool']*5

rank = rank_total.groupby('srch_id', group_keys=False).apply(lambda x: x.sort_values('prob', ascending=False))

rank_order = rank.groupby('srch_id').head(10).reset_index(drop=True)
rank_order['position'] = rank_order.groupby(['srch_id'])['position'].rank(method='first').astype(int).astype(str)

ndgc_input = rank_order.pivot(index='srch_id', columns='position', values='score').reset_index()

del ndgc_input.columns.name
new_names = [(i, 'rank_' + i) for i in ndgc_input.iloc[:,1:].columns.values]
ndgc_input.rename(columns = dict(new_names), inplace=True)

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
     
    else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

ndgc_input['ndcg'] = ndgc_input.iloc[:,1:].apply(lambda x: ndcg_at_k(x, 10), axis=1)
dt_score = ndgc_input['ndcg'].mean()
dt_score

ndgc_input['ndcg'] = ndgc_input.iloc[:,1:].apply(lambda x: ndcg_at_k(x, 10), axis=1)
rm_score = ndgc_input['ndcg'].mean()
rm_score
ndgc_input['ndcg'] = ndgc_input.iloc[:,1:].apply(lambda x: ndcg_at_k(x, 10), axis=1)
gb_score = ndgc_input['ndcg'].mean()
gb_score



X_total = pd.concat([X_test, y_test], axis=1)
X_total = X_total[['srch_id','click_bool','booking_bool']]
X_total['position'] = 0
X_total['score'] = X_total['click_bool']*1 + X_total['booking_bool']*5

X_order = X_total.groupby('srch_id').head(10).reset_index(drop=True)

X_order = X_order.groupby('srch_id').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)

X_order['position'] = X_order.groupby(['srch_id'])['position'].rank(method='first').astype(int).astype(str)

random_input = X_order.pivot(index='srch_id', columns='position', values='score').reset_index()

new_names = [(i, 'rank_' + i) for i in random_input.iloc[:,1:].columns.values]
random_input.rename(columns = dict(new_names), inplace=True)
random_input['ndcg'] = random_input.iloc[:,1:].apply(lambda x: ndcg_at_k(x, 10), axis=1)
random_score = random_input['ndcg'].mean()
random_score

