import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import logging

from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras
from sklearn.model_selection import train_test_split
from ast import literal_eval
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV

import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier

import os

# This script will  a multilabel classification task with a keras implementation of a CNN
# We will be using the rolled common input as this will restrict labeling to 10 labels at most.

# Set Logging -- basic configuration
logging.basicConfig(format='%(asctime)s -- %(levelname)s: %(message)s',
                    level=logging.NOTSET,
                    datefmt='%Y-%m-%d %H:%M:%S')
my_logger = logging.getLogger('classifier_project')

# Define the file path to be using
input_file = "./cleansed_data/rolled_common_input.txt"

# create DF list
df_list = []

my_logger.info("Reading input file into memory ... ")

# Read the file into memory
with open("./cleansed_data/rolled_common_input.txt", "r") as f:
    # Python 3.10 and up
    while line := f.readline():
        labels = []
        note = ''
        for item in line.rstrip('\n').split(' '):
            # not found
            if item.find('__label__') == -1:
                note = note + str(item) + ' '
            else:
                labels.append(str(item))

        labels.sort()
        df_list.append([str(labels), note])

# Convert input to dataframe
mimic_data = pd.DataFrame(df_list, columns=['labels', 'notes'])

my_logger.info(f"Input file loaded ... There are {len(mimic_data)} rows in the cleansed dataset.")

# Remove the lowest occurrence classes for stratification
# There are some terms with occurrence as low as 1.
my_logger.info("    Labels with an occurrence of only one: " + str(sum(mimic_data["labels"].value_counts() == 1)))

# Filter out the low occurrence classes
mimic_data_filtered = mimic_data.groupby("labels").filter(lambda x: len(x) > 1)

# Convert Label literals to lists
mimic_data_filtered["labels"] = mimic_data_filtered["labels"].apply(lambda x: literal_eval(x))

# Create a Stratified test split
# Initial train and test split. hold back 10% for test per paper
train_df, test_df = train_test_split(mimic_data_filtered, test_size=.2, stratify=mimic_data_filtered["labels"].values)
# further filtering
val_df = test_df.sample(frac=0.5)
test_df.drop(val_df.index, inplace=True)

my_logger.info("Validation sets created ...")
my_logger.info(f"   Number of rows in training set: {len(train_df)}")
my_logger.info(f"   Number of rows in validation set: {len(val_df)}")
my_logger.info(f"   Number of rows in test set: {len(test_df)}")

# Preprocess the labels - using multi-label binarization
labels = tf.ragged.constant(train_df["labels"].values)
# lookup type, create lookup
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
lookup.adapt(labels)
vocab = lookup.get_vocabulary()

# list labels, debug only
# my_logger.info(vocab)
# get stats, debug only -- need this for sizing of our params, we set max seq length to the 50% per keras tutorial
my_logger.info("Cleanse dataset statistics for word tokens ...")
my_logger.info(train_df["notes"].apply(lambda x: len(x.split(" "))).describe())

max_seqlen = 1750
batch_size = 32  # per report
padding_token = "<pad>"
auto = tf.data.AUTOTUNE


# helper functions for making a dataset for trainer
def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["labels"].values)
    labels_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["notes"].values, labels_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)


# Taken from KERAS tutorial
# reverses single multi-hot encoded label to a tuple of vocab terms
def invert_multi_hot(encoded_labels):
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)


# per keras tutorial
train_dataset = make_dataset(train_df, is_train=True)
validation_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)

# preview - DEBUG ONLY
"""
text_batch, label_batch = next(iter(train_dataset))

for i, text in enumerate(text_batch[:1]):
    label = label_batch[i].numpy()[None, ...]
    my_logger.info(f"Abstract: {text}")
    my_logger.info(f"Label(s): {invert_multi_hot(label[0])}")
"""

# Source: https://stackoverflow.com/a/18937309/7636462
vocabulary = set()
train_df["notes"].str.lower().str.split().apply(vocabulary.update)
vocabulary_size = len(vocabulary)
my_logger.info("Vocabulary size: " + str(vocabulary_size))

# Vectorize the notes information
my_logger.info("Vectorizing the notes ...")
text_vectorizer = layers.TextVectorization(max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf")

# `TextVectorization` layer needs to be adapted as per the vocabulary from our
# training set.
with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

train_dataset = train_dataset.map(lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto).prefetch(
    auto)
validation_dataset = validation_dataset.map(lambda text, label: (text_vectorizer(text), label),
                                            num_parallel_calls=auto).prefetch(auto)
test_dataset = test_dataset.map(lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto).prefetch(
    auto)

my_logger.info("Vectorizing the notes COMPLETE ...")


# Define the model

def make_model():
    model_cnn_multi = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(lookup.vocabulary_size(), activation="sigmoid"),
        ]  # More on why "sigmoid" has been used here in a moment.
    )
    return model_cnn_multi


# Train the model
epochs = 20

model = make_model()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[metrics.Precision(), metrics.Recall()])
# history if i want to plot
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)

# Evaluate the model
_, precision, recall = model.evaluate(test_dataset)
my_logger.info("Precision: " + str(precision))
my_logger.info("Recall: " + str(recall))
