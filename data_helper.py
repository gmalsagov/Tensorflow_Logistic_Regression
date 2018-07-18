import os
import csv
import matplotlib.pyplot as plt
import string
import numpy as np
from collections import Counter
import pandas as pd
from keras.utils.np_utils import to_categorical
import re
import random

directory = 'data/'

def load_data(file_name):

    save_file_name = os.path.join(directory,str(file_name))

    # Copy rows into array
    if os.path.isfile(save_file_name):
        text_data = []
        with open(save_file_name, 'r') as temp_output_file:
            reader = csv.reader(temp_output_file)
            for row in reader:
                text_data.append(row)

            data_array = np.array(text_data)

    texts = np.array([x[1] for x in data_array])
    targets = np.array([x[0] for x in data_array])

    return texts, targets

def normalize_data(texts, targets):

    # Count number of occurences for each target
    count = Counter(targets)
    emotions = []
    for k, v in count.items():
        emotions.append(k)

    # Normalize text
    # Lower case
    texts = [x.lower() for x in texts]

    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]


    # # Plot histogram of text lengths
    # text_lengths = [len(x.split()) for x in texts]
    # text_lengths = [x for x in text_lengths if x < 50]
    # plt.hist(text_lengths, bins=25)
    # plt.title('Histogram of # of Words in Texts')
    # plt.show()

    return texts, targets

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def loadData(filename):
    df = pd.read_csv(directory + filename)
    selected = ['label', 'text']
    non_selected = list(set(df.columns) - set(selected))
    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    labels = sorted(list(set(df[selected[0]].tolist())))
    dict.fromkeys(set(df[selected[0]].tolist()))
    label_dict = {}
    for i in range(len(labels)):
        label_dict[labels[i]] = i
    x_train = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    # y_train = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    # y_train = to_categorical(np.asarray(y_train))
    y_train = df[selected[0]].tolist()

    count = Counter(y_train)
    categories = []
    for k, v in count.items():
        categories.append(k)

    dictionary = dict(zip(x_train, y_train))

    return dictionary, categories


def randomize_data(dictionary, categories):
    # Count number of occurences for each target

    joy_data = {}
    shame_data= {}
    sadness_data= {}
    guilt_data = {}
    disgust_data = {}
    anger_data = {}
    fear_data = {}

    # Splitting data into emotion categories
    for k, v in dictionary.items():
        if v in categories[0]:
            joy_data.update({k:v})
        elif v in categories[1]:
            shame_data.update({k: v})
        elif v in categories[2]:
            sadness_data.update({k: v})
        elif v in categories[3]:
            guilt_data.update({k: v})
        elif v in categories[4]:
            disgust_data.update({k: v})
        elif v in categories[5]:
            anger_data.update({k: v})
        elif v in categories[6]:
            fear_data.update({k: v})

    train_data = {}
    test_data = {}
    np.random.seed(5)
    count = 0

    # Splitting dataset into train and test data randomly
    # JOY
    joy_count = len(joy_data)
    shame_count = len(shame_data)
    sadness_count = len(sadness_data)
    guilt_count = len(guilt_data)
    disgust_count = len(disgust_data)
    anger_count = len(anger_data)
    fear_count = len(fear_data)

    while count < 0.8 * joy_count:
        x, y = random.choice(list(joy_data.items()))
        train_data.update({x: y})
        del joy_data[x]
        count = count + 1
    count = 0
    while count < 0.2 * joy_count:
        x, y = random.choice(list(joy_data.items()))
        test_data.update({x: y})
        del joy_data[x]
        count = count + 1
    count = 0
    # SHAME
    while count < 0.8 * shame_count:
        x, y = random.choice(list(shame_data.items()))
        train_data.update({x: y})
        del shame_data[x]
        count = count + 1
    count = 0
    while count < int(0.2 * shame_count):
        x, y = random.choice(list(shame_data.items()))
        test_data.update({x: y})
        del shame_data[x]
        count = count + 1
    count = 0
    # SADNESS
    while count < 0.8 * sadness_count:
        x, y = random.choice(list(sadness_data.items()))
        train_data.update({x: y})
        del sadness_data[x]
        count = count + 1
    count = 0
    while count < int(0.2 * sadness_count):
        x, y = random.choice(list(sadness_data.items()))
        test_data.update({x: y})
        del sadness_data[x]
        count = count + 1
    count = 0
    # Guilt
    while count < 0.8 * guilt_count:
        x, y = random.choice(list(guilt_data.items()))
        train_data.update({x: y})
        del guilt_data[x]
        count = count + 1
    count = 0
    while count < int(0.2 * guilt_count):
        x, y = random.choice(list(guilt_data.items()))
        test_data.update({x: y})
        del guilt_data[x]
        count = count + 1
    count = 0
    # disgust
    while count < 0.8 * disgust_count:
        x, y = random.choice(list(disgust_data.items()))
        train_data.update({x: y})
        del disgust_data[x]
        count = count + 1
    count = 0
    while count < int(0.2 * disgust_count):
        x, y = random.choice(list(disgust_data.items()))
        test_data.update({x: y})
        del disgust_data[x]
        count = count + 1
    count = 0
    # anger
    while count < 0.8 * anger_count:
        x, y = random.choice(list(anger_data.items()))
        train_data.update({x: y})
        del anger_data[x]
        count = count + 1
    count = 0
    while count < int(0.2 * anger_count):
        x, y = random.choice(list(anger_data.items()))
        test_data.update({x: y})
        del anger_data[x]
        count = count + 1
    count = 0
    # fear
    while count < 0.8 * fear_count:
        x, y = random.choice(list(fear_data.items()))
        train_data.update({x: y})
        del fear_data[x]
        count = count + 1
    count = 0
    while count < int(0.2 * fear_count):
        x, y = random.choice(list(fear_data.items()))
        test_data.update({x: y})
        del fear_data[x]
        count = count + 1
    #
    # count = Counter(test_data.values())
    # print(count)
    return train_data, test_data
