import pandas as pd
import numpy as np
import pickle
import time
from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns

import os
import gc
import zipfile
import io

train = pd.read_csv('train.csv')

train.head()


train.drop(columns=['Unnamed: 0','date_time', 'check_in','check_out',], inplace=True)
train.drop(columns=['is_booking','short_trip','room_count','solo_travel','is_mobile','weekend_trip','posa_continent','similar_events',\
                    'is_package','adult_count','child_count','channel','user_location_country','price_compare','biz_trip'], inplace=True)

train['hotel_cluster'].replace([91, 41, 48, 64, 5, 65, 98, 59, 70, 42], \
                               ['ninety-one','forty-one','forty-eight','sixty-four', 'five',\
                               'sixty-five','ninety-eight','fifty-nine','seventy','forty-two'], inplace=True)
train['hotel_cluster'].value_counts()

train.shape

from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score, confusion_matrix, make_scorer, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, log_loss
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

X = train.drop(['hotel_cluster'], axis=1)
y = train['hotel_cluster']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

tree = DecisionTreeClassifier(max_depth=12, criterion='entropy', max_features=12)

tree = tree.fit(X_train, y_train)
y_pred = tree.predict(X_val)

print("Accuracy:", metrics.accuracy_score(y_val, y_pred))
print("The score is")
print("Training: {:6.2f}%".format(100*tree.score(X_train, y_train)))
print("Validate: {:6.2f}%".format(100*tree.score(X_val, y_val)))
print("Testing: {:6.2f}%".format(100*tree.score(X_test, y_test)))

tree_pred = tree.predict(X_val)
print('Tree Results:')
print(confusion_matrix(y_val, tree_pred))
print(classification_report(y_val, tree_pred))


feature_importance = abs(tree.feature_importances_)
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

featfig = plt.figure(1,figsize=(17,10))
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center', color='#eca02c')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=14)

featax.set_xlabel('Relative Feature Importance', fontsize=20, weight='bold')
featax.tick_params(axis="x", labelsize=16)


featax.spines['right'].set_visible(False)
featax.spines['top'].set_visible(False)
featax.spines['bottom'].set_visible(True)
featax.spines['left'].set_visible(True)

featax.patch.set_visible(False)

plt.savefig('feature_importance.jpg', transparent=True);
# plt.show()


files.download('all_feature_importance.jpg')

est = DecisionTreeClassifier()

rf_p_dist={
           'criterion':['gini','entropy'],
           'max_depth':[3,6,9,12],
           'max_features':[3,6,9,12],
          }

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,
                                  n_jobs=-1, n_iter=nbr_iter, cv=5)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score

hypertuning_rscv(est, rf_p_dist, 40, X, y)

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier

X = train.drop(['hotel_cluster'], axis=1)
y = train['hotel_cluster']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

model = XGBClassifier(max_depth=12)

model = model.fit(X_train, y_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

print("Validate accuracy: %.2f" % (accuracy_score(y_val, y_pred_val) * 100))
print("Test accuracy: %.2f" % (accuracy_score(y_test, y_pred_test) * 100))

feature_importance = abs(model.feature_importances_)
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5


from sklearn.model_selection import GridSearchCV

est = xgb.XGBClassifier()

rf_p_dist = {
     "eta"    : [0.01, 0.1, 0.3] ,
     "max_depth"        : [ 6, 9, 12],
     "min_child_weight" : [ 3, 5, 7 ],
     "gamma"            : [ 0.1, 0.3, 0.4 ],
     }

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,
                                  n_jobs=-1, n_iter=nbr_iter, cv=5)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score

hypertuning_rscv(est, rf_p_dist, 1, X, y)

from numpy import loadtxt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

X = train.drop(['hotel_cluster'], axis=1)
y = train['hotel_cluster']

# CV model
model = XGBClassifier(max_depth=12)
kfold = KFold(n_splits=5, random_state=7)
results = cross_val_score(model, X, y, cv=kfold)
print(results)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

