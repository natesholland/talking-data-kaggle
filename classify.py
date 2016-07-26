import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

NUM_TREES = 500
TREE_DEPTH = 20
NUM_THREADS = 4

# import code; code.interact(local=dict(globals(), **locals()))

train = pd.read_csv("data/train_features.csv", dtype={'device_id': np.str})
test = pd.read_csv("data/test_features.csv", dtype={'device_id': np.str})

(train_data, val_data) = train_test_split(train)

X_tr = train_data.values[:, 3:]
y_tr = list(train_data.values[:, 2])
X_val = val_data.values[:, 3:]
y_val = list(val_data.values[:, 2])

recognizer = RandomForestClassifier(NUM_TREES, max_depth=TREE_DEPTH, verbose=1, n_jobs=NUM_THREADS)
recognizer.fit(X_tr, y_tr)

prediction = recognizer.predict_proba(X_val)

if (os.name == 'nt'):
    ll = log_loss(y_val.tolist(), prediction)
else:
    ll = log_loss(y_val, prediction)

print("Log loss: " + str(ll))

X_tr = train_data.values[:, 3:]
y_tr = list(train_data.values[:, 2])

X_test = test.values[:, 2:]

recognizer = RandomForestClassifier(NUM_TREES, max_depth=TREE_DEPTH, verbose=1, n_jobs=NUM_THREADS)
recognizer.fit(X_tr, y_tr)
prediction = recognizer.predict_proba(X_test)

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
# write_submission_file(test, prediction, ll)
