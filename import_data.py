import psycopg2
import os

appPath = os.path.dirname(os.path.realpath(__file__))

conn = psycopg2.connect("host='localhost' port='5432' dbname='kaggle'")

cur = conn.cursor()

# -------------------------phone_brand_device_model----------------------------
print("phone_brand_device_model importing...")

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
print("events importing...")

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


# ------------------------gender_age_test---------------------------------------
print("gender_age_test importing...")

dropTableString = """
DROP TABLE IF EXISTS gender_age_test;
"""

cur.execute(dropTableString)

createTableString = """
CREATE TABLE gender_age_test (
    device_id       varchar(40)
);
"""

cur.execute(createTableString)

f = open(appPath + '/data/gender_age_test.csv', 'r')
f.readline()

cur.copy_from(f, 'gender_age_test', sep=',')

# ----------------------------app_labels---------------------------------------
print("app_labels importing...")

dropTableString = """
DROP TABLE IF EXISTS app_labels;
"""

cur.execute(dropTableString)

createTableString = """
CREATE TABLE app_labels (
    app_id      varchar(40),
    label_id    integer
);
"""

cur.execute(createTableString)

f = open(appPath + '/data/app_labels.csv', 'r')
f.readline()

cur.copy_from(f, 'app_labels', sep=',')

# ----------------------------label_categories----------------------------------
print("label_categories importing...")

dropTableString = """
DROP TABLE IF EXISTS label_categories;
"""

cur.execute(dropTableString)

createTableString = """
CREATE TABLE label_categories (
    label_id    integer,
    category    varchar(40)
);
"""

cur.execute(createTableString)

f = open(appPath + '/data/app_labels.csv', 'r')
f.readline()

cur.copy_from(f, 'app_labels', sep=',')

# ------------------------------app_events-------------------------------------
print("app_events importing...")

dropTableString = """
DROP TABLE IF EXISTS app_events;
"""

cur.execute(dropTableString)

createTableString = """
CREATE TABLE app_events (
    event_id        integer,
    app_id          varchar(40),
    is_installed    boolean,
    is_active       boolean
);
"""

cur.execute(createTableString)

f = open(appPath + '/data/app_events.csv', 'r')
f.readline()

cur.copy_from(f, 'app_events', sep=',')

# ------------------------------gender_age_train-------------------------------------
print("gender_age_train importing...")

dropTableString = """
DROP TABLE IF EXISTS gender_age_train;
"""

cur.execute(dropTableString)

createTableString = """
CREATE TABLE gender_age_train (
    device_id       varchar(40),
    gender          char(1),
    age             integer,
    grouping        varchar(10)
);
"""

cur.execute(createTableString)

f = open(appPath + '/data/gender_age_train.csv', 'r')
f.readline()

cur.copy_from(f, 'gender_age_train', sep=',')

#------------------------------------------------------------------------------

conn.commit()
conn.close()
f.close()
