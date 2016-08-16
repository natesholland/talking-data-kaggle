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

# Create bag-of-apps in character string format
# first by event
# then merge to generate larger bags by device

##################
#   App Events
##################
print("# Read App Events")
app_ev = pd.read_csv(path+"app_events.csv", dtype={'device_id': np.str})
# remove duplicates(app_id)

print("# Read App Labels and map")
app_lab = pd.read_csv(path+"app_labels.csv", dtype={'device_id': np.str})
app_lab = app_lab.groupby('app_id')['label_id'].apply(lambda x: " ".join(set("label_id:" + str(s) for s in x)))
app_ev['labels'] = app_ev['app_id'].map(app_lab)
events_to_labels = app_ev.groupby("event_id")["labels"].apply(lambda x: " ".join(set(" ".join(set(s for s in x)).split(' '))))

print("# Map App Events")
app_ev = app_ev.groupby("event_id")["app_id"].apply(
    lambda x: " ".join(set("app_id:" + str(s) for s in x)))

##################
#     Events
##################
print("# Read Events")
events = pd.read_csv(path+"events.csv", dtype={'device_id': np.str})
events["app_id"] = events["event_id"].map(app_ev)
events["labels"] = events["event_id"].map(events_to_labels)

events = events.dropna()

del app_ev
del app_lab

events = events[["device_id", "app_id", "labels"]]

# remove duplicates(app_id)
events_to_app_ids = events.groupby("device_id")["app_id"].apply(
    lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" "))))
events_to_app_ids = events_to_app_ids.reset_index(name="app_id")

# expand to multiple rows
events_to_app_ids = pd.concat([pd.Series(row['device_id'], row['app_id'].split(' '))
                    for _, row in events_to_app_ids.iterrows()]).reset_index()
events_to_app_ids.columns = ['app_id', 'device_id']

events_to_label_ids = events.groupby("device_id")["labels"].apply(
    lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" "))))
events_to_label_ids = events_to_label_ids.reset_index(name="labels")

# expand to multiple rows
events_to_label_ids = pd.concat([pd.Series(row['device_id'], row['labels'].split(' '))
                    for _, row in events_to_label_ids.iterrows()]).reset_index()
events_to_label_ids.columns = ['labels', 'device_id']

del events
del events_to_labels

##################
#   Phone Brand
##################
print("# Read Phone Brand")
pbd = pd.read_csv(path+"phone_brand_device_model.csv",
                  dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)


##################
#  Train and Test
##################
print("# Generate Train and Test")

train = pd.read_csv(path+"gender_age_train.csv",
                    dtype={'device_id': np.str})

train["gender"][train["gender"]=='M']=1
train["gender"][train["gender"]=='F']=0
Y_gender = train["gender"]
Y_age = train["age"]
Y_age = np.log(Y_age)

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

# Concat
Df = pd.concat((train, test), axis=0, ignore_index=True)

Df = pd.merge(Df, pbd, how="left", on="device_id")
Df["phone_brand"] = Df["phone_brand"].apply(lambda x: "phone_brand:" + str(x))
Df["device_model"] = Df["device_model"].apply(
    lambda x: "device_model:" + str(x))

###################
#  Concat Feature
###################

f1 = Df[["device_id", "phone_brand"]]   # phone_brand
f2 = Df[["device_id", "device_model"]]  # device_model
f3 = events_to_app_ids[["device_id", "app_id"]]    # app_id
f4 = events_to_label_ids[["device_id", "labels"]] # label ids

del Df

f1.columns.values[1] = "feature"
f2.columns.values[1] = "feature"
f3.columns.values[1] = "feature"
f4.columns.values[1] = "feature"

FLS = pd.concat((f1, f2, f3, f4), axis=0, ignore_index=True)


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

def compute_val_loss(percent_features=25, max_depth=6):
    losses = []

    skf = StratifiedKFold(Y, n_folds=10, shuffle=True)
    # skf = KFold(train.shape[0],n_folds=5, shuffle=True, random_state=seed)
    for ind_tr, ind_te in skf:
        X_train = train_sp[ind_tr]
        X_val = train_sp[ind_te]
        y_train = Y[ind_tr]
        y_val = Y[ind_te]
        y_train_gender = Y_gender[ind_tr]
        y_val_gender = Y_gender[ind_te]
        y_train_age = Y_age[ind_tr]
        y_val_age = Y_age[ind_te]


        ##################
        #   Feature Sel
        ##################
        print("# Feature Selection")
        selector = SelectPercentile(f_classif, percentile=percent_features)

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
            "max_depth": 6,
            "eval_metric": "mlogloss",
            "eta": 0.07,
            "silent": 1,
            "alpha": 3,
        }

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_val, y_val)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        progress = dict()

        gbm = xgb.train(params, dtrain, 60, evals=watchlist,
                        early_stopping_rounds=25, verbose_eval=True, evals_result=progress)

        val_loss = float(min(progress['eval']['mlogloss']))

        losses.append(val_loss)

    print(losses)
    mean = reduce(lambda x, y: x + y, losses) / len(losses)
    return mean

compute_val_loss()


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
