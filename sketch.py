import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

# Helpful debugger line copied from here: https://gist.github.com/obfusk/208597ccc64bf9b436ed
# import code; code.interact(local=dict(globals(), **locals()))

def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table

events = pd.read_csv("data/events.csv", dtype={'device_id': np.str})
events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')

pbd = pd.read_csv("data/phone_brand_device_model.csv", dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)
pbd = map_column(pbd, 'phone_brand')
pbd = map_column(pbd, 'device_model')

train = pd.read_csv("data/gender_age_train.csv", dtype={'device_id': np.str})
train = map_column(train, 'group')
train = train.drop(['gender'], axis=1)
train = train.drop(['age'], axis=1)
train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)
train.fillna(-1, inplace=True)

X_tr = train[['phone_brand', 'device_model', 'counts']].values
y_tr = train['group'].values


recognizer = RandomForestClassifier(4, max_depth=5)
recognizer.fit(X_tr, y_tr)



