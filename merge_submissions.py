import pandas as pd
import numpy as np
import sys
import datetime

files = sys.argv[1:]

print(len(files))

panda_stuff = []

for f in files:
    panda_stuff.append(pd.read_csv(f, dtype={'device_id': np.str}))

concat = pd.concat(tuple(panda_stuff))

means = concat.groupby('device_id').mean()

now = datetime.datetime.now()

means.to_csv('submission_stacked_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv', index=True, index_label='device_id')
