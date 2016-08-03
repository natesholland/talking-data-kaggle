import pandas as pd
import numpy as np
from tzwhere import tzwhere
import matplotlib.pyplot as plt

# import code; code.interact(local=dict(globals(), **locals()))

print('initializing timezones...')

tz = tzwhere.tzwhere(shapely=True, forceTZ=True)

print('importing events...')

events = pd.read_csv("data/events.csv", dtype={'device_id': np.str})

print('converting timezones...')

events['timezone'] = events[['latitude', 'longitude']].apply( lambda x: tz.tzNameAt(x['latitude'], x['longitude']), axis=1)

TIMEZONE_MAP = {
    'None': 0,
    'Asia/Shanghai': 0,
    'Asia/Chongqing': 0,
    'Asia/Harbin': 0,
    'Asia/Urumqi': 0,
    'Asia/Kashgar': 0,
    'Asia/Colombo': -3, # this should really be -3.5 but I didn't feel like messing with that
    'Asia/Rangoon': -2, # this should really be -2.5 but I didn't feel like messing with that
    'Asia/Dubai': -4,
    'America/Denver': -14,
    'Asia/Macau': 0,
    'America/Montreal': -12,
    'Europe/Zurich': -7,
    'Europe/Amsterdam': -7,
    'Asia/Hong_Kong': 0,
    'America/New_York': -12,
    'Asia/Bangkok': -1,
    'America/Mexico_City': -13,
    'Asia/Makassar': 0,
    'Asia/Kuwait': -5,
    'America/Toronto': -12,
    'Africa/Windhoek': -7,
    'Europe/London': -7,
    'Europe/Berlin': -7,
    'Asia/Manila': 0,
    'Australia/Sydney': 2,
    'Asia/Thimphu': -2,
    'Asia/Seoul': 1,
    'Africa/Khartoum': -5,
    'Asia/Almaty': -2,
    'Australia/Perth': 0,
    'Asia/Taipei': 0,
    'Asia/Phnom_Penh': -1,
    'Asia/Kuala_Lumpur': 0,
    'Asia/Pyongyang': 1, # this one might be 30 minutes off. But it's North Korea so who knows
    'Australia/Melbourne': 2,
    'Europe/Athens': -5,
    'Indian/Mauritius': -4,
    'Asia/Ho_Chi_Minh': -1,
    'Asia/Riyadh': -5,
    'Asia/Dhaka': -2,
    'Asia/Kolkata': -3, # another one that's off by 30 minutes
    'Asia/Singapore': 0,
    'Asia/Tokyo': 1,
    'Europe/Prague': -6,
    'Pacific/Auckland': 4,
    'Pacific/Honolulu': -17,
    'Europe/Bucharest': -6,
    'America/Los_Angeles': -15,
    'Asia/Vladivostok': 2,
    'Europe/Paris': -7,
    'Asia/Choibalsan': 0
}

events['timezone'] = events.groupby(['device_id'])['timezone'].transform( lambda x: x.fillna(x.unique()[x.unique() != np.array(None)][0] if x.unique()[x.unique() != np.array(None)].any() else 'None'))

print('localizing time...')

events['local_hour'] = events[['timezone', 'timestamp']].apply(lambda x: (int(x.timestamp[11:13]) + TIMEZONE_MAP[x.timezone]) % 24, axis=1)

events.to_csv('data/events_localized.csv')
