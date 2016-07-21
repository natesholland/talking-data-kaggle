import psycopg2
import os

appPath = os.path.dirname(os.path.realpath(__file__))

conn = psycopg2.connect("host='localhost' port='5432' dbname='kaggle'")

cur = conn.cursor()

dropTableString = """
DROP TABLE phone_brand_device_model;
"""

cur.execute(dropTableString)

databaseSelectString = """
SELECT table_schema,table_name
FROM information_schema.tables
ORDER BY table_schema,table_name;
"""

# cur.execute('SELECT current_database();')
# print(cur.fetchall())

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
cur.execute('select * from phone_brand_device_model limit 1')
print(cur.fetchall())

conn.commit()
conn.close()
f.close()
