# Notice: This is partially copied from  ZFTurbo at:
# https://www.kaggle.com/zfturbo/talkingdata-mobile-user-demographics/xgboost-simple-starter/discussion

import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

# Helpful debugger line copied from here: https://gist.github.com/obfusk/208597ccc64bf9b436ed
# import code; code.interact(local=dict(globals(), **locals()))

def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table

print("importing training data...")

events = pd.read_csv("data/events.csv", dtype={'device_id': np.str})
events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')

pbd = pd.read_csv("data/phone_brand_device_model.csv", dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)
pbd = map_column(pbd, 'phone_brand')
pbd = map_column(pbd, 'device_model')

train = pd.read_csv("data/gender_age_train.csv", dtype={'device_id': np.str})
labels = sorted(train['group'].unique())
print(labels)
train = map_column(train, 'group')
train = train.drop(['gender'], axis=1)
train = train.drop(['age'], axis=1)
train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)
train.fillna(-1, inplace=True)

print("importing training data...")

test = pd.read_csv("data/gender_age_test.csv", dtype={'device_id': np.str})
test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
test = pd.merge(test, events_small, how='left', on='device_id', left_index=True)
test.fillna(-1, inplace=True)

X_test = test[['phone_brand', 'device_model', 'counts']].values

# This is what we can use for cross validation
result = train_test_split(train)

train_data = result[0]
val_data = result[1]
X_tr = train_data[['phone_brand', 'device_model', 'counts']].values
y_tr = train_data['group'].values
X_val = val_data[['phone_brand', 'device_model', 'counts']].values
y_val = val_data['group'].values

print('fitting data...')
recognizer = RandomForestClassifier(10, max_depth=5)
recognizer.fit(X_tr, y_tr)

print('prediting data...')
prediction = recognizer.predict_proba(X_val)

ll = log_loss(y_val, prediction)

print "Log loss: " + str(ll)

X_tr = train[['phone_brand', 'device_model', 'counts']].values
y_tr = train['group'].values

recognizer = RandomForestClassifier(10, max_depth=5)
recognizer.fit(X_tr, y_tr)
prediction = recognizer.predict_proba(X_test)

def write_submission_file(test, prediction):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    f = open(sub_file, 'w')
    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    total = 0
    test_val = test['device_id'].values
    for i in range(len(test_val)):
        str1 = str(test_val[i])
        for j in range(12):
            str1 += ',' + str(prediction[i][j])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()

# uncomment this line if you want to write out a predition
# write_submission_file(test, prediction)

# import code; code.interact(local=dict(globals(), **locals()))
