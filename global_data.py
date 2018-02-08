import numpy as np
import logging
from logging.config import fileConfig
from sklearn.datasets import fetch_20newsgroups

# create logger
fileConfig('logging_config.ini')
logger = logging.getLogger()
logger.setLevel("WARNING")

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

cat_comp = categories[:4]   # Computer Technologies
cat_rec = categories[4:]    # Recreational Activities
CAT = ["Computer Technologies", "Recreational Activities"]

logging.info("loading data")
# all_data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# create labels
# 0 for computer technology, 1 for recreational activities
train_label = [(x//4) for x in train_data.target]
test_label = [(x//4) for x in test_data.target]

'''
comp_data_test = fetch_20newsgroups(subset='test', categories=cat_comp, shuffle=True, random_state=42)
comp_data_train = fetch_20newsgroups(subset='train', categories=cat_comp, shuffle=True, random_state=42)
rec_data_test = fetch_20newsgroups(subset='test', categories=cat_rec, shuffle=True, random_state=42)
rec_data_train = fetch_20newsgroups(subset='train', categories=cat_rec, shuffle=True, random_state=42)
'''

logging.info("loading finished")

