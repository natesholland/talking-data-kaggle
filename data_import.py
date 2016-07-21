import psycopg2
import os

appPath = os.path.dirname(os.path.realpath(__file__))

conn = psycopg2.connect("host='localhost' port='5432' dbname='kaggle'")

cur = conn.cursor()

# -------------------------phone_brand_device_model----------------------------
dropTableString = """
DROP TABLE IF EXISTS phone_brand_device_model;
"""

cur.execute(dropTableString)

createTableString = """
CREATE TABLE phone_brand_device_model (
    device_id       varchar(40),
    phone_brand     varchar(40),
    device_model    varchar(40)
);
"""

cur.execute(createTableString)

f = open(appPath + '/data/phone_brand_device_model.csv', 'r')
f.readline()

cur.copy_from(f, 'phone_brand_device_model', sep=',')
# cur.execute('select * from phone_brand_device_model limit 1')
# print(cur.fetchall())

# --------------------------------events---------------------------------------

dropTableString = """
DROP TABLE IF EXISTS events;
"""

cur.execute(dropTableString)

createTableString = """
CREATE TABLE events (
    event_id        integer,
    device_id       varchar(40),
    timestamp       timestamp,
    longitude       float,
    latitude        float
);
"""

cur.execute(createTableString)

f = open(appPath + '/data/events.csv', 'r')
f.readline()

cur.copy_from(f, 'events', sep=',')



conn.commit()
conn.close()
f.close()
