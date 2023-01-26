import numpy as np
import os
import argparse
import multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import datetime
from tempfile import TemporaryFile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['KERAS_BACKEND'] = 'tensorflow'

import pybedtools
from pybedtools import featurefuncs
import pyBigWig

import sklearn
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats

import tensorflow as tf
from tensorflow.python.framework import ops
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
from tensorflow.keras.metrics import AUC
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
import cv2
# 导入自定义的包
from create_model import model_create



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pred_pos_regionitives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_m_precision = true_positives / (pred_pos_regionitives + K.epsilon())
    return precision_m_precision


def f1_m(y_true, y_pred):
    f1_m_precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((f1_m_precision * recall) / (f1_m_precision + recall + K.epsilon()))


refine_parser = argparse.ArgumentParser(
    description='Training CNN model to predict STARR-seq enhancers based on chromatin accessbility and histone marks')
refine_parser.add_argument('--features1_peaks', type=str, help='chromatin accessibility peak')
refine_parser.add_argument('--features2_peaks', type=str, help='ChIP-seq H3K27ac peak')
refine_parser.add_argument('--features3_peaks', type=str, help='ChIP-seq H3K4me3 peak')
refine_parser.add_argument('--features4_peaks', type=str, help='RNA-seq  peak')
refine_parser.add_argument('--features1_bw', type=str, help='chromatin accessibility bigWig')
refine_parser.add_argument('--features2_bw', type=str, help='ChIP-seq H3K27ac bigWig')
refine_parser.add_argument('--features3_bw', type=str, help='ChIP-seq H3K4me3 bigWig')
refine_parser.add_argument('--features4_bw', type=str, help='RNA-seq  bigWig')
refine_parser.add_argument('--out_dir', type=str, help='output_directory')
refine_parser.add_argument('--species', type=str, help='name of the species')
refine_parser.add_argument('--tissue_name', type=str, help='name of the tissue')
refine_parser.add_argument('--window_size', type=int, help='prediction window size')
refine_parser.add_argument('--model_name', type=str, help='name of the model')

window_size = 4000

args = refine_parser.parse_args()
for key, value in vars(args).items():
    if key == "species" or key == "in_dir" or key == "out_dir" or key == "tissue_name" or key == "window_size":
        continue
    else:
        if not os.path.exists(value):
            print(key + " argument file does not exist")
            exit(1)
print("check finished")

chrom_list = []
for i in range(1, 19):
    chrom_list.append("chr" + str(i))
chrom_list.append("chrX")
chrom_list.append("chrY")
print(chrom_list)

for target_chrom in chrom_list:

    ATAC_bigwig = pyBigWig.open(args.features1_bw)
    Chip1_bigwig = pyBigWig.open(args.features2_bw)
    Chip2_bigwig = pyBigWig.open(args.features3_bw)
    RNA_bigwig = pyBigWig.open(args.features4_bw)

    def bigWigAverageOverBed(x, bigwig):
        return bigwig.stats(x.chrom, x.start, x.stop, nBins=int(window_size / 10))


    def get_signal(input_list):
        print(input_list)
        sys.stdout.flush()
        return [bigWigAverageOverBed(x, pyBigWig.open(input_list[0])) for x in pybedtools.BedTool(input_list[1])]


    ATAC_sig_pred = get_signal(pybedtools.BedTool(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "pred0.5_pos_regions.bed"),
                                      Chromatin_bigwig)
    print("Chromatin is ok")
    chip1_sig_pred = get_signal(pybedtools.BedTool(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "pred0.5_pos_regions.bed"),
                                  Chip1_bigwig)
    print("Chip1 is ok")
    chip2_sig_pred = get_signal(pybedtools.BedTool(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "pred0.5_pos_regions.bed"),
                                  Chip2_bigwig)
    print("Chip2 is ok")
    RNA_sig_pred = get_signal(pybedtools.BedTool(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "pred0.5_pos_regions.bed"),
                                  RNA_bigwig)
    print("RNA is ok")

    ATAC_sig_pred = [np.array(i) for i in ATAC_sig_pred]
    chip1_sig_pred = [np.array(i) for i in chip1_sig_pred]
    chip2_sig_pred = [np.array(i) for i in chip2_sig_pred]
    RNA_sig_pred = [np.array(i) for i in RNA_sig_pred]

    pred_pos_region = pybedtools.BedTool().window_maker(
        b=args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "pred0.5_pos_regions.bed",
        n=200)
    print(pred_pos_region.count())
    pred_pos_region.saveas(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "Sep_pred0.5_pos_regions.bed")

    x_valid_region = []

    Pre_pred_pos_regions = pybedtools.BedTool(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "pred0.5_pos_regions.bed")
    for i in range(Pre_pred_pos_regions.count()):
        x_valid_region.append(
            np.array([ATAC_sig_pred[i], chip1_sig_pred[i], chip2_sig_pred[i], RNA_sig_pred[i]]))

    x_valid_region = np.nan_to_num(np.array(x_valid_region, dtype=float))
    print(x_valid_region.shape)

    x_valid_region = np.expand_dims(np.array(x_valid_region), axis=3)

    print(x_valid_region.shape)
    model = model_create(width=int(window_size / 10))
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    adam = Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=9e-5)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', f1_m, recall_m, precision_m],
                  experimental_run_tf_function=False)
    model.load_weights(args.model_name)


    def normalize(x):
        return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

    tf.compat.v1.disable_eager_execution()

    def cam_pred_multi_1d(model, X):
        layer_name = "conv2d_4"

        y_c = model.output
        print(y_c.shape)
        conv_output = model.get_layer(layer_name).output
        print(conv_output.shape)
        grads = K.gradients(y_c, conv_output)[0]
        print(grads)
        grads = normalize(grads)
        gradient_function = K.function([model.input], [conv_output, grads])

        output, grads_val = gradient_function([X])
        weights = np.mean(grads_val, axis=(1, 2))

        all_cam = []
        for i in range(len(weights)):
            cam = np.dot(output[i], weights[i])
            cam = cv2.resize(cam, (200, 1), cv2.INTER_LINEAR)
            cam = np.maximum(cam, 0)
            all_cam.append(cam[0])

        all_cam = np.array(all_cam)
        return all_cam
 

    print(len(x_valid_region))

    raw_pred_1d = []
    stride = 1000
    for i in range(int(len(x_valid_region) / stride) + 1):
        if (i % 10 == 0):
            print(i, ",000")
        raw_pred_1d.extend(cam_pred_multi_1d(model, x_valid_region[i * stride: (i + 1) * stride]))
    raw_pred_1d = np.array(raw_pred_1d)

    print(stats.describe(raw_pred_1d.flatten()))

    pd_file = pd.read_csv(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "Sep_pred0.5_pos_regions.bed",
        sep="\t", header=None)
    pd_file[4] = "bin"
    pd_file[5] = raw_pred_1d.flatten()
    pd_file.to_csv(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "all.pred_pos_regions.bed",
        sep="\t", header=None, index=False)

    grad_cam_filter = np.mean(raw_pred_1d.flatten())
    pd_file[pd_file[5] > grad_cam_filter].to_csv(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "filter.all.pred_pos_regions.bed",
        sep="\t", header=None, index=False)

    pybedtools.BedTool(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "all.pred_pos_regions.bed").sort().merge(
        c=5, o="mean").saveas(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "all.merge.pred_pos_regions.bed")
    pybedtools.BedTool(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "filter.all.pred_pos_regions.bed").sort().merge(
        c=5, o="mean").saveas(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "filter.merge.pred_pos_regions.bed")

    pd_file = pd.read_csv(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "filter.merge.pred_pos_regions.bed",
        sep="\t", header=None)
    pd_file[3] = pd_file.index
    pd_file.to_csv(
        args.out_dir + args.species + "." + args.tissue_name + "." + target_chrom + "." + "Final_pred_pos_regions.bed",
        sep="\t", header=None, index=False)
