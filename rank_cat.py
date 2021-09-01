import numpy as np
import pandas as pd
from random import randint
from catboost import CatBoostClassifier, Pool, cv, CatBoost, Pool, MetricVisualizer

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GroupKFold, GroupShuffleSplit, cross_val_score, TimeSeriesSplit, cross_validate
from pprint import pprint
from catboost import cv
import zipfile
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report
from copy import deepcopy
import os


get_ipython().system('unzip /kaggle/input/expedia-personalized-sort/data.zip ')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def get_target(row):
    if row.booking_bool>0:
        return 1
    if row.click_bool>0 :
        return 0.2
    return 0


def featurize_df(df:pd.DataFrame) ->pd.DataFrame:

    df["weekday"] = df["date_time"].dt.weekday
    df["week_of_year"] = df["date_time"].dt.week

    df["hour"] = df["date_time"].dt.hour
    df["minute"] = df["date_time"].dt.minute
    df["time_epoch"] = df["date_time"].astype('int64')//1e9
    df["early_night"] = ((df["hour"]>19) | (df["hour"]<3)) 
    df["nans_count"] = df.isna().sum(axis=1)
    return df

df = pd.read_csv('train.csv',nrows=30123456)



df["date_time"] = pd.to_datetime(df["date_time"],infer_datetime_format=True)
df["target"] = df.apply(get_target,axis=1)

df_test = pd.read_csv('test.csv')

cols = df_test.columns.drop(['date_time'])
float_cols = df_test.columns[df_test.dtypes.eq('float')]
for c in float_cols:
    df_test[c] = pd.to_numeric(df_test[c], errors="ignore",downcast="integer") 
df_test["date_time"] = pd.to_datetime(df_test["date_time"],infer_datetime_format=True)

df.drop_duplicates(['click_bool','booking_bool','random_bool'])

drop_cols = []

drop_unary_cols = [c for c
             in list(df)
             if df[c].nunique(dropna=False) <= 1]
target_cols = ["gross_bookings_usd","click_bool","booking_bool"] 
drop_cols.extend(drop_unary_cols)
drop_cols.extend(target_cols) 


df = df.drop(columns=drop_cols,errors="ignore")
df_test = df_test.drop(columns=drop_cols,errors="ignore")

df = featurize_df(df)
df_test = featurize_df(df_test)

df.drop(['comp3_rate',
       'comp3_inv', 'comp3_rate_percent_diff', 'comp4_inv', 'comp5_rate',
       'comp5_inv', 'comp5_rate_percent_diff', 'comp8_rate', 'comp8_inv',
       'comp8_rate_percent_diff'],axis=1).groupby(df["target"]>0).mean()


cutoff_id = df["srch_id"].quantile(0.94) # 90/10 split
X_train = df.loc[df.srch_id< cutoff_id].drop(["target"],axis=1)
X_eval = df.loc[df.srch_id>= cutoff_id].drop(["target"],axis=1)
y_train = df.loc[df.srch_id< cutoff_id]["target"]
y_eval = df.loc[df.srch_id>= cutoff_id]["target"]



df["target"].value_counts()



categorical_cols = ['prop_id',"srch_destination_id", "weekday"] 


df.tail()

set(X_train.columns).symmetric_difference(set(df_test.columns))



train_pool = Pool(data=X_train,
                  label = y_train,
                  cat_features=categorical_cols,
                  group_id=X_train["srch_id"]
                 )

eval_pool = Pool(data=X_eval,
                  label = y_eval,
                  cat_features=categorical_cols,
                  group_id=X_eval["srch_id"]
                 )





## default_parameters  = {
    'iterations': 2000,
    'custom_metric': ['NDCG', "AUC:type=Ranking"], 
    'random_seed': 42,
#     "task_type":"GPU",
    "has_time":True,
    "metric_period":4,
    "save_snapshot":False,
    "use_best_model":True,
} 

parameters = {}

def fit_model(loss_function, additional_params=None, train_pool=train_pool, test_pool=eval_pool):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function
    
    if additional_params is not None:
        parameters.update(additional_params)
        
    model = CatBoost(parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)
    print("best results (train on train):")
    print(model.get_best_score()["learn"])
    print("best results (on validation set):")
    print(model.get_best_score()["validation"])
    
    print("(Default) Feature importance (on train pool)")
    display(model.get_feature_importance(data=train_pool,prettified=True).head(15))
    
    try:
        print("SHAP features importance, on all data:")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(pd.concat([X_train,X_eval]),
                                            y=pd.concat([y_train,y_eval]))

        shap.summary_plot(shap_values, pd.concat([X_train,X_eval]))
    finally:
        return model

