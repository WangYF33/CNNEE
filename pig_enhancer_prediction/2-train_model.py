# -*- coding: utf-8 -*-
import numpy as np
import os
import joblib
import argparse
import matplotlib.pyplot as plt  # 1
from datetime import datetime
from sklearn import metrics
from sklearn.utils import shuffle, class_weight
from matplotlib import pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten, \
    GlobalAveragePooling2D, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.python.keras.engine
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, Callback, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Import custom packages
from create_model import model_create
from custom_metrics import *

def get_train_data(species, inputpath, seq_types):
    print(species)
    pos = []
    neg = []
    first_seq = True
    for Type in seq_types:
        print("-" + Type)
        postive_name = args.inputpath + args.species + "." + args.tissue_name + "." + Type + ".pos.tsv"
        postive_matrix = np.loadtxt(postive_name, delimiter='\t')
        negative_name = args.inputpath + args.species + "." + args.tissue_name + "." + Type + ".neg.tsv"
        negative_matrix = np.loadtxt(negative_name, delimiter='\t')
        if first_seq == True:
            for i in postive_matrix:
                pos.append(np.array([n]))
            for i in negative_matrix:
                neg.append(np.array([n]))
            first_seq = False
        else:
            for i in range(len(pos)):
                pos[n] = np.vstack((pos[n], postive_matrix[n,]))
            for i in range(len(neg)):
                neg[n] = np.vstack((neg[n], negative_matrix[n,]))

    X_pos = np.array(pos)
    X_neg = np.array(neg)
    X = np.vstack((X_pos, X_neg))
    y = np.array([1 for i in range(X_pos.shape[0])] + [0 for i in range(X_neg.shape[0])]).reshape(-1, 1)
    return X, y


# creat_model

# custom_metrics
 
data_analysis = argparse.ArgumentParser(
    description='Training CNN model to predict pig enhancers based on chromatin accessbility, histone and transcriptome marks')
data_analysis.add_argument('--species', type=str, help='comma separated string of species')
data_analysis.add_argument('--inputpath', type=str, help='input data (tsv files) directory')
data_analysis.add_argument('--outpath', type=str, help='output_directory')
data_analysis.add_argument('--tissue_name', type=str, help='name of the tissue')


argument_str = ' --species ' + "Duroc" + \
              ' --inputpath ' + "/user/xx/input/"
              ' --outpath ' + "/user/xx/output/" + \
              ' --tissue_name ' + "Muscle" + \

              
seq_types = ["H3K27ac", "H3K4me3", "ATAC", "RNA"]

args = data_analysis.parse_args(argument_str.split())

for seq in seq_types:
    postive_file = args.inputpath + args.species + "." + args.tissue_name + "." + seq + ".pos.tsv"
    if not os.path.exists(postive_file):
        print(postive_file + " file does not exist")
        exit(1)
    negative_file = args.inputpath + args.species + "." + args.tissue_name + "." + seq + ".neg.tsv"
    if not os.path.exists(negative_file):
        print(negative_file + " file does not exist")
        exit(1)
print("all files found!")



X, y = get_train_data(args.species, args.inputpath, seq_types)
print(X.shape)
print(y.shape)
with open(args.outpath + "susScr11_signals.pickle", 'wb') as f:
    joblib.dump((X, y), f)
with open(args.outpath + "susScr11_signals.pickle", 'rb') as f:
    X, Y = joblib.load(f)
window_size = int(X.shape[2] * 10)
print(window_size)
X, Y = shuffle(X, Y, random_state=0)
x_train = np.expand_dims(X, axis=3)
y_train = Y
print(x_train.shape)
print(y_train.shape)

class_weights = {0: 5.5, 1: 0.55}
model = model_create(width=int(window_size / 10))
es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, mode='auto')
adam = Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=9e-5)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', f1_m, recall_m, precision_m])
pre_model = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.1, shuffle=True, callbacks=[es], class_weight=class_weights)
model.save_weights(args.species + "." + args.tissue_name+".h5")
print("done training")
