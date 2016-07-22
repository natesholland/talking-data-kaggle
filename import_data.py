import psycopg2
import os
import fileinput
import re

from tempfile import mkstemp
from shutil import move
from os import remove, close


def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    close(fh)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

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
    category    varchar(80)
);
"""

cur.execute(createTableString)

file_path = appPath + '/data/label_categories.csv'

replace(file_path, '"online shopping by group, like groupon"', 'online shopping by group like groupon')
replace(file_path, '"online shopping, price comparing"', 'online shopping price comparing')
replace(file_path, '"Jewelry, jewelry"', 'Jewelry jewelry')
replace(file_path, '"Europe, the United States and Macao (aviation)"', 'Europe the United States and Macao (aviation)')
replace(file_path, '"Hong Kong, Macao and Taiwan (aviation)"', '"Hong Kong Macao and Taiwan (aviation)"')
replace(file_path, '"Europe, the United States and Macao (Travel)"', 'Europe the United States and Macao (Travel)')
replace(file_path, '"Hong Kong, Macao and Taiwan (Travel)"', '"Hong Kong Macao and Taiwan (Travel)"')


f = open(file_path, 'r')
f.readline()

cur.copy_from(f, 'label_categories', sep=',')

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
