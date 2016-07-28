import pandas as pd
import numpy as np
import datetime
import os
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

params = {
    "objective": "multi:softprob",
    "num_class": 12,
    "booster" : "gbtree",
    "eval_metric": "mlogloss",
    "eta": 0.3,
    "max_depth": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "silent": 1
}

early_stopping_rounds = 20
num_boost_round = 500

# import code; code.interact(local=dict(globals(), **locals()))

train = pd.read_csv("data/train_features.csv", dtype={'device_id': np.str})
test = pd.read_csv("data/test_features.csv", dtype={'device_id': np.str})

(train_data, val_data) = train_test_split(train)

X_tr = train_data.values[:, 3:]
y_tr = list(train_data.values[:, 2])
X_val = val_data.values[:, 3:]
y_val = list(val_data.values[:, 2])

dtrain = xgb.DMatrix(X_tr, y_tr)
dvalid = xgb.DMatrix(X_val, y_val)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

prediction = gbm.predict(xgb.DMatrix(X_val), ntree_limit=gbm.best_iteration)

if (os.name == 'nt'):
    ll = log_loss(y_val.tolist(), prediction)
else:
    ll = log_loss(y_val, prediction)

print("Log loss: " + str(ll))

X_tr = train_data.values[:, 3:]
y_tr = list(train_data.values[:, 2])

X_test = test.values[:, 2:]

prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration)

def write_submission_file(test, prediction, log_loss):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(log_loss) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
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
write_submission_file(test, prediction, ll)
