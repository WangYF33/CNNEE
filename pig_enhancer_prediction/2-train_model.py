# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from tempfile import TemporaryFile
from sklearn import metrics
from sklearn.utils import shuffle, class_weight


import tensorflow as tf
import tensorflow.keras.backend
import tensorflow.python.keras.engine
from tensorflow.keras.layers import Input, Layer, InputSpec, Reshape, LeakyReLU, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, \
       BatchNormalization, Flatten, Multiply
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras import regularizers
from tensorflow.keras import metrics

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from create_model import model_create
from custom_metrics import *

def get_train_data(species, inputpath, seq_types):
    print(species)
    data_pos = []
    data_neg = []
    fir_seq_types = True
    for Type in seq_types:
        postive_name = args.inputpath + args.species + "_" + args.tissue_name + "_" + Type + "_postive.tsv"
        postive_matrix = pd.read_table(postive_name, sep='\t',header=None)
        negative_name = args.inputpath + args.species + "_" + args.tissue_name + "_" + Type + "_negative.tsv"
        negative_matrix = pd.read_table(negative_name, sep='\t',header=None)
        if fir_seq_types == True:
            for n in postive_matrix:
                data_pos.append(np.array([n]))
            for n in negative_matrix:
                data_neg.append(np.array([n]))
            fir_seq_types = False
        else:
            for n in range(len(data_pos)):
                data_pos[n] = np.vstack((data_pos[n], postive_matrix[n,]))
            for n in range(len(data_neg)):
                data_neg[n] = np.vstack((data_neg[n], negative_matrix[n,]))

    postive_mat = np.array(data_pos)
    negative_mat = np.array(data_neg)
    x = np.vstack((postive_mat, negative_mat))
    y = np.array([1 for i in range(postive_mat.shape[0])] + [0 for i in range(negative_mat.shape[0])]).reshape(-1, 1)
    return x, y


data_analysis = argparse.ArgumentParser()
data_analysis.add_argument('--species', type=str)
data_analysis.add_argument('--inputpath', type=str)
data_analysis.add_argument('--outpath', type=str)
data_analysis.add_argument('--tissue_name')


argument_str = ' --species ' + "xxx" + \
              ' --inputpath ' + "/user/xx/input/"
              ' --outpath ' + "/user/xx/output/" + \
              ' --tissue_name ' + "Muscle" + \

              
seq_types = ["H3K27ac", "H3K4me3", "ATAC", "RNA"]

args = data_analysis.parse_args(argument_str.split())

x, y = get_train_data(args.species, args.inputpath, seq_types)

with open(args.outpath + "susScr11_signals.pickle", 'wb') as f:
    joblib.dump((x, y), f)
with open(args.outpath + "susScr11_signals.pickle", 'rb') as f:
    train1, train2 = joblib.load(f)
window_size = int(train1.shape[2] * 10)
print(window_size)
trainX, trainY = shuffle(train1, train2, random_state=0)
Train_X = np.expand_dims(trainX, axis=3)
Train_y = trainY
print(Train_X.shape)
print(Train_y.shape)

class_weights = {0: 5.5, 1: 0.55}
model = model_create(width=int(window_size / 10))
earlystop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, mode='auto')
ADam = Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=9e-5)
model.compile(loss='binary_crossentropy', optimizer=ADam, metrics=['accuracy', Precision_s, Recall_s, f1_source])
pre_model = model.fit(Train_X, Train_y, batch_size=32, epochs=100, validation_split=0.1, shuffle=True, callbacks=[earlystop], class_weight=class_weights)
model.save_weights(args.species + "." + args.tissue_name+".h5")
print("train finished")
