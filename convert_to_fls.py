# coding=utf8
# Based on yibo's R script and JianXiao's Python script
from scipy import sparse
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
import pandas as pd
import numpy as np
from scipy import sparse as ssp
import datetime
from sklearn.preprocessing import LabelEncoder

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

FLS.to_csv('data/fls.csv')
