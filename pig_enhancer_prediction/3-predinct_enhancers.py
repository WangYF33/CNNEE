import numpy as np
import os
import argparse
import sys
import multiprocessing as mp
from datetime import datetime
from tempfile import TemporaryFile

import pybedtools
from pybedtools import featurefuncs
import pyBigWig

import sklearn
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten, \
    GlobalAveragePooling2D, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import backend as K
# from tensorflow.keras.engine.topology import Layer, InputSpec
import tensorflow.python.keras.engine
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, Callback, TensorBoard, ReduceLROnPlateau
# import keras_metrics as km
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
# 导入自定义的包

from create_model import model_create
from custom_metrics import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['chr_num']='0'

window_size = 4000


def bigWigAverageOverBed(x, bigwig):
    return bigwig.stats(x.chrom, x.start, x.stop, nBins=int(window_size / 10))


def get_signal(input_list):
    print(input_list)
    sys.stdout.flush()
    return [bigWigAverageOverBed(x, pyBigWig.open(input_list[0])) for x in pybedtools.BedTool(input_list[1])]
    

 
data_analysis = argparse.ArgumentParser(
    description='Training CNN model to predict STARR-seq enhancers based on chromatin accessbility and histone marks')
data_analysis.add_argument('--features1_bw', type=str, help='ATAC-seq bigWig')
data_analysis.add_argument('--features2_bw', type=str, help='ChIP-seq H3K27ac bigWig')
data_analysis.add_argument('--features3_bw', type=str, help='ChIP-seq H3K4me3 bigWig')
data_analysis.add_argument('--features4_bw', type=str, help='RNA-seq  bigWig')
data_analysis.add_argument('--outpath', type=str, help='output_directory')
data_analysis.add_argument('--species', type=str, help='name of the species')
data_analysis.add_argument('--tissue_name', type=str, help='name of the tissue')
data_analysis.add_argument('--window_size', type=int, help='prediction window size')


seq_names = ["ATAC", "H3K27ac", "H3K4me3", "RNA"]
window_size = 4000

tissue_name = "Muscle"

argument_str = ' --featrue1_bw ' + "/BIGDATA2/scau_xlyuan_1/wyf/DECODE/Pig-Enhancer/encode/dataest/Duroc/Duroc.Muscle.ATAC-seq.bigWig" + \
              ' --featrue2_bw ' + "/BIGDATA2/scau_xlyuan_1/wyf/DECODE/Pig-Enhancer/encode/dataest/Duroc/Duroc.Muscle.ChIP-seq.H3K27ac.bigWig" + \
              ' --featrue3_bw ' + "/BIGDATA2/scau_xlyuan_1/wyf/DECODE/Pig-Enhancer/encode/dataest/Duroc/Duroc.Muscle.ChIP-seq.H3K4me3.bigWig" + \
              ' --featrue4_bw ' + "/BIGDATA2/scau_xlyuan_1/wyf/DECODE/Pig-Enhancer/encode/dataest/Duroc/Duroc.Muscle.RNA-seq.bigWig" + \
              ' --outpath ' + "/BIGDATA2/scau_xlyuan_1/wyf/DECODE/Pig-Enhancer/Dataest/Muscle/" + \
              ' --tissue_name ' + tissue_name + \
              ' --species ' + "Duroc" + \
              ' --window_size ' + "4000"

args = data_analysis.parse_args(argument_str.split())


# args.species = args.species.split(",")
for key, value in vars(args).items():
    if key == "species" or key == "in_dir" or key == "outpath" or key == "tissue_name" or key == "window_size":
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

    susScr11_windows = pybedtools.BedTool().window_maker(genome="susScr11", w=window_size, s=2000)
    susScr11_windows = susScr11_windows.filter(pybedtools.featurefuncs.greater_than, window_size - 1)
    print(target_chrom)
    susScr11_windows = susScr11_windows.filter(lambda x: x.chrom == target_chrom).sort()
    candidate_regions = susScr11_windows

    candidate_regions.saveas(
        args.outpath + args.species + "." + args.tissue_name + "." + target_chrom + ".valid_regions.bed")
    print("candidate_regions: " + str(candidate_regions.count()))

    ATAC_bigwig = pyBigWig.open(args.features1_bw)
    Chip1_bigwig = pyBigWig.open(args.features2_bw)
    Chip2_bigwig = pyBigWig.open(args.features3_bw)
    RNA_bigwig = pyBigWig.open(args.features4_bw)

    ATAC_sig_region = get_signal(pybedtools.BedTool(
        args.outpath + args.species + "." + args.tissue_name + "." + target_chrom + ".valid_regions.bed"),
                                      ATAC_bigwig)
    print("ATAC is ok")
    chip1_sig_region = get_signal(pybedtools.BedTool(
        args.outpath + args.species + "." + args.tissue_name + "." + target_chrom + ".valid_regions.bed"), Chip1_bigwig)
    print("Chip1 is ok")
    chip2_sig_region = get_signal(pybedtools.BedTool(
        args.outpath + args.species + "." + args.tissue_name + "." + target_chrom + ".valid_regions.bed"), Chip2_bigwig)
    print("Chip2 is ok")
    RNA_sig_region = get_signal(pybedtools.BedTool(
        args.outpath + args.species + "." + args.tissue_name + "." + target_chrom + ".valid_regions.bed"), RNA_bigwig)
    print("RNA is ok")

    # reformat the validationation values
    ATAC_sig_region = [np.array(i) for i in ATAC_sig_region]
    chip1_sig_region = [np.array(i) for i in chip1_sig_region]
    chip2_sig_region = [np.array(i) for i in chip2_sig_region]
    RNA_sig_region = [np.array(i) for i in RNA_sig_region]

    X_sig_region = []
    for i in range(candidate_regions.count()):
        X_sig_region.append(
            np.array([ATAC_sig_region[i], chip1_sig_region[i], chip2_sig_region[i], RNA_sig_region[i]]))
    X_sig_region = np.nan_to_num(np.array(X_sig_region, dtype=float))

    print(X_sig_region.shape)

    valid_X = np.expand_dims(valid_X, axis=3)
    print(valid_X.shape)

    model = model_create(width=int(window_size / 10))
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    adam = Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=9e-5)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', f1_m, recall_m, precision_m])
    model.load_weights(args.model_name)

    valid_Y = model.predict(valid_X).ravel()
    print(valid_Y.shape)

    # format into bed with proper regions
    pd_file = pd.read_csv(
        args.outpath + args.species + "." + args.tissue_name + "." + target_chrom + "." + "valid_regions.bed", sep="\t",
        header=None)
    pd_file[4] = "predicton"
    pd_file[5] = valid_Y
    pd_file.to_csv(
        args.outpath + args.species + "." + args.tissue_name + "." + target_chrom + "." + "prediction_regions.bed",
        sep="\t", header=None, index=False)

    # filter for positive predictions
    pd_file[pd_file[5] > 0.5].to_csv(
        args.outpath + args.species + "." + args.tissue_name + "." + target_chrom + "." + "pred0.5_pos_regions.bed",
        sep="\t", header=None, index=False)

    pybedtools.BedTool(
        args.outpath + args.species + "." + args.tissue_name + "." + target_chrom + "." + "pred0.5_pos_regions.bed").sort().merge(
        c=5, o="mean").saveas(
        args.outpath + args.species + "." + args.tissue_name + "." + target_chrom + "." + "merge_pred0.5_pos_regions.bed")
    pd_file = pd.read_csv(
        args.outpath + args.species + "." + args.tissue_name + "." + target_chrom + "." + "merge_pred0.5_pos_regions.bed",
        sep="\t", header=None)
    pd_file[3] = pd_file.index
    pd_file.to_csv(
        args.outpath + args.species + "." + args.tissue_name + "." + target_chrom + "." + "all.merge_pred0.5_pos_regions.bed",
        sep="\t", header=None, index=False)

