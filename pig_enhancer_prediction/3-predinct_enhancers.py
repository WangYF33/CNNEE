# -*- coding: utf-8 -*-
import argparse
import numpy as np
from datetime import datetime
from tempfile import TemporaryFile
import os
import pybedtools
from pybedtools import BedTool
from pybedtools import featurefuncs
import pyBigWig


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

def bigwig_aver_peak(x, bw_file):
    return bw_file.stats(x.chrom, x.start, x.stop, nBins=int(WindowSize / 10))


def sig_dataset(input_list):
    print(input_list)
    sys.stdout.flush()
    return [bigwig_aver_peak(x, pyBigWig.open(input_list[0])) for x in pybedtools.BedTool(input_list[1])]

 
data_analysis = argparse.ArgumentParser()
data_analysis.add_argument('--features1_bigwig', type=str, help='ATAC-seq bigWig')
data_analysis.add_argument('--features2_bigwig', type=str, help='ChIP-seq H3K27ac bigWig')
data_analysis.add_argument('--features3_bigwig', type=str, help='ChIP-seq H3K4me3 bigWig')
data_analysis.add_argument('--features4_bigwig', type=str, help='RNA-seq  bigWig')
data_analysis.add_argument('--outpath', type=str, help='output_directory')
data_analysis.add_argument('--species', type=str, help='name of the species')
data_analysis.add_argument('--tissue_name', type=str, help='name of the tissue')
data_analysis.add_argument('--WindowSize', type=int, help='prediction window size')


seq_names = ["ATAC", "H3K27ac", "H3K4me3", "RNA"]

tissue_name = "Muscle"

argument_str = ' --featrue1_bigwig ' + "/xxxx/xxx/xx.ATAC.bigwig" + \
              ' --featrue2_bigwig ' + "/xxxx/xxx/xx.H3K27ac.bigwig" + \
              ' --featrue3_bigwig ' + "/xxxx/xxx/xx.H3K4me3.bigwig" + \
              ' --featrue4_bigwig ' + "/xxxx/xxx/xx.RNA.bigwig" + \
              ' --outpath ' + "/user/xx/output/" + \
              ' --tissue_name ' + tissue_name + \
              ' --species ' + "xxx" + \
              ' --WindowSize ' + "4000"

args = data_analysis.parse_args(argument_str.split())

chrom_list = []
for i in range(1, 19):
    chrom_list.append("chr" + str(i))
chrom_list.append("chrX")
chrom_list.append("chrY")
print(chrom_list)

for target_chrom in chrom_list:

    susScr11_windows = pybedtools.BedTool().window_maker(genome="susScr11", w=WindowSize, s=2000)
    susScr11_windows = susScr11_windows.filter(pybedtools.featurefuncs.greater_than, WindowSize - 1)
    print(target_chrom)
    susScr11_windows = susScr11_windows.filter(lambda x: x.chrom == target_chrom).sort()
    candidate_regions = susScr11_windows
    with open(args.outpath + args.species + "_" + args.tissue_name + "_" + target_chrom + ".candidate_regions.bed","w") as cand_region:
        np.savetxt(cand_region, candidate_regions, fmt='%s', delimiter='\t')

  
    candidate_region = args.outpath + args.species + "_" + args.tissue_name + "_" + target_chrom + ".candidate_regions.bed"
    
    input_list_ATAC=[args.features1_bigwig, candidate_region]
    ATAC_sig_region = sig_dataset(input_list_ATAC)
    print("ATAC_sig finished")
  
    input_list_H3K27ac=[args.features2_bigwig, candidate_region]
    H3K27ac_sig_region = sig_dataset(input_list_H3K27ac)
    print("H3K27ac_sig finished")
    
    input_list_H3K4me3=[args.features3_bigwig, candidate_region]
    H3K4me3_sig_region = sig_dataset(input_list_H3K4me3)
    print("H3K4me3_sig finished")
    
    input_list_RNA=[args.features4_bigwig, candidate_region]
    RNA_sig_region = sig_dataset(input_list_RNA)
    print("RNA_sig finished")
    
    ATAC_sig_region = [np.array(i) for i in ATAC_sig_region]
    H3K27ac_sig_region = [np.array(i) for i in H3K27ac_sig_region]
    H3K4me3_sig_region = [np.array(i) for i in H3K4me3_sig_region]
    RNA_sig_region = [np.array(i) for i in RNA_sig_region]

    X_sig_region = []
    for i in range(candidate_regions.count()):
        X_sig_region.append(
            np.array([ATAC_sig_region[i], H3K27ac_sig_region[i], H3K4me3_sig_region[i], RNA_sig_region[i]]))
    X_sig_region = np.nan_to_num(np.array(X_sig_region, dtype=float))

    print(X_sig_region.shape)

    valid_X = np.expand_dims(valid_X, axis=3)
    print(valid_X.shape)

    model = model_create(width=int(WindowSize / 10))
    earlystop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, mode='auto')
    ADam = Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=9e-5)
    model.compile(loss='binary_crossentropy', optimizer=ADam, metrics=['accuracy', Precision_s, Recall_s, f1_source])
    model.load_weights(args.species + "." + args.tissue_name+".h5")

    valid_Y = model.predict(valid_X).ravel()
    print(valid_Y.shape)

    pd_file = np.loadtxt(
        args.outpath + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "candidate_regions.bed", delimiter='\t')
    pd_file[4] = "predicton"
    pd_file[5] = valid_Y
    with open(args.outpath + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "prediction_regions.bed","w") as pred_region:
        np.savetxt(pred_region, pd_file, fmt='%s', delimiter='\t')
        
    pd_file[pd_file[5] > 0.5].to_csv(
        args.outpath + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "pred0.5_pos_regions.bed",
        sep="\t", header=None, index=False)
    
    tmpe1 = pybedtools.BedTool(
        args.outpath + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "pred0.5_pos_regions.bed").sort().merge(c=5, o="mean").
    with open(args.outpath + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "merge_pred0.5_pos_regions.bed","w") as merge_pred50_region:
        np.savetxt(merge_pred50_region, tmpe1, fmt='%s', delimiter='\t')
    pd_file = pd.read_csv(
        args.outpath + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "merge_pred0.5_pos_regions.bed",
        sep="\t", header=None)
    pd_file[3] = pd_file.index
    with open(args.outpath + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "all.merge_pred0.5_pos_regions.bed","w") as all_merge_pred50_region:
        np.savetxt(all_merge_pred50_region, pd_file, fmt='%s', delimiter='\t')

