# coding=utf8
# Based on yibo's R script and JianXiao's Python script
from scipy import sparse
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
import pandas as pd
import numpy as np
from scipy import sparse as ssp
import pylab as plt
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,MinMaxScaler,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.base import BaseEstimator

from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from sklearn.decomposition import TruncatedSVD, SparsePCA
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_classif, chi2
from sklearn.linear_model import Ridge, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import xgboost as xgb
import datetime

seed = 1024

path = "data/"

# import code; code.interact(local=dict(globals(), **locals()))

##################
#  Train and Test
##################
print("# Generate Train and Test")

train = pd.read_csv(path+"gender_age_train.csv",
                    dtype={'device_id': np.str})

train.drop(["age", "gender"], axis=1, inplace=True)

test = pd.read_csv(path+"gender_age_test.csv",
                   dtype={'device_id': np.str})
test["group"] = np.nan

split_len = len(train)

# Group Labels
Y = train["group"]
lable_group = LabelEncoder()
Y = lable_group.fit_transform(Y)
device_id = test["device_id"]

FLS = pd.read_csv(path+"fls.csv",
                  dtype={'device_id': np.str})


###################
# User-Item Feature
###################
print("# User-Item-Feature")

device_ids = FLS["device_id"].unique()
feature_cs = FLS["feature"].unique()

data = np.ones(len(FLS))
dec = LabelEncoder().fit(FLS["device_id"])
row = dec.transform(FLS["device_id"])
col = LabelEncoder().fit_transform(FLS["feature"])
sparse_matrix = sparse.csr_matrix(
    (data, (row, col)), shape=(len(device_ids), len(feature_cs)))

sparse_matrix = sparse_matrix[:, sparse_matrix.getnnz(0) > 0]

##################
#      Data
##################

train_row = dec.transform(train["device_id"])
train_sp = sparse_matrix[train_row, :]

test_row = dec.transform(test["device_id"])
test_sp = sparse_matrix[test_row, :]

X_train = pd.read_csv(path+'gender_age_train.csv')
group_le = LabelEncoder()
group_lb = LabelBinarizer()
labels = group_le.fit_transform(X_train['group'].values)
labels = group_lb.fit_transform(labels)

def compute_val_loss(percent_features=0.8, max_depth=6, eta=0.07, alpha=3, gamma=0, subsample=1, min_child_weight=1, n_folds=7, num_rounds=60):
    losses = []

    skf = StratifiedKFold(Y, n_folds=n_folds, shuffle=True)
    # skf = KFold(train.shape[0],n_folds=5, shuffle=True, random_state=seed)
    for ind_tr, ind_te in skf:
        X_train = train_sp[ind_tr]
        X_val = train_sp[ind_te]
        y_train = Y[ind_tr]
        y_val = Y[ind_te]

        selector = SelectPercentile(f_classif, percentile=100)

        selector.fit(X_train, y_train)

        X_train = selector.transform(X_train).toarray()
        X_val = selector.transform(X_val).toarray()

        temp_train_sp = selector.transform(train_sp)
        temp_test_sp = selector.transform(test_sp).toarray()

        print("# Num of Features: ", X_train.shape[1])

        params = {
            "objective": "multi:softprob",
            "num_class": 12,
            "booster": "gblinear",
            "max_depth": max_depth,
            "eval_metric": "mlogloss",
            "eta": eta,
            "silent": 1,
            "alpha": alpha,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "subsample": subsample,
            "colsample_bytree": percent_features
        }

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_val, y_val)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        progress = dict()

        gbm = xgb.train(params, dtrain, num_rounds, evals=watchlist,
                        early_stopping_rounds=25, verbose_eval=True, evals_result=progress)

        val_loss = float(min(progress['eval']['mlogloss']))

        losses.append(val_loss)

    print(losses)
    mean = reduce(lambda x, y: x + y, losses) / len(losses)
    return mean

import code; code.interact(local=dict(globals(), **locals()))

depths = [2, 3, 4, 5, 6, 7, 8, 10, 12]
val_losses = []

for depth in depths:
    val_loss = compute_val_loss(max_depth=depth)
    val_losses.append(val_loss)
    print("depth: " + str(depth) + " val_loss: " + str(val_loss))

import code; code.interact(local=dict(globals(), **locals()))
# Write results
submission = pd.DataFrame(y_preds, columns=group_le.classes_)
submission["device_id"] = device_id
submission = submission.set_index("device_id")
now = datetime.datetime.now()
submission.to_csv('submission_mlp_sparse_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv', index=True, index_label='device_id')
