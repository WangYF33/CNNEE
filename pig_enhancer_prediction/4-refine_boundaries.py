
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from datetime import datetime
import os
import pybedtools
from pybedtools import BedTool
from pybedtools import featurefuncs
import pyBigWig
from tempfile import TemporaryFile

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
disable_eager_execution()


data_analysis = argparse.ArgumentParser()
data_analysis.add_argument('--features1_bigwig', type=str, help='ATAC-seq bigWig')
data_analysis.add_argument('--features2_bigwig', type=str, help='ChIP-seq H3K27ac bigWig')
data_analysis.add_argument('--features3_bigwig', type=str, help='ChIP-seq H3K4me3 bigWig')
data_analysis.add_argument('--features4_bigwig', type=str, help='RNA-seq  bigWig')
data_analysis.add_argument('--outpath', type=str, help='output_directory')
data_analysis.add_argument('--species', type=str, help='name of the species')
data_analysis.add_argument('--tissue_name', type=str, help='name of the tissue')
data_analysis.add_argument('--WindowSize', type=int, help='prediction window size')

tissue_name = "Muscle"

argument_str = ' --featrue1_bigwig ' + "/BIGDATA2/scau_xlyuan_1/wyf/DECODE/Pig-Enhancer/encode/dataest/Duroc/Duroc.Muscle.ATAC-seq.bigWig" + \
              ' --featrue2_bigwig ' + "/BIGDATA2/scau_xlyuan_1/wyf/DECODE/Pig-Enhancer/encode/dataest/Duroc/Duroc.Muscle.ChIP-seq.H3K27ac.bigWig" + \
              ' --featrue3_bigwig ' + "/BIGDATA2/scau_xlyuan_1/wyf/DECODE/Pig-Enhancer/encode/dataest/Duroc/Duroc.Muscle.ChIP-seq.H3K4me3.bigWig" + \
              ' --featrue4_bigwig ' + "/BIGDATA2/scau_xlyuan_1/wyf/DECODE/Pig-Enhancer/encode/dataest/Duroc/Duroc.Muscle.RNA-seq.bigWig" + \
              ' --outpath ' + "/BIGDATA2/scau_xlyuan_1/wyf/DECODE/Pig-Enhancer/Dataest/Muscle/" + \
              ' --tissue_name ' + tissue_name + \
              ' --species ' + "Duroc" + \
              ' --WindowSize ' + "4000"

args = data_analysis.parse_args(argument_str.split())

chrom_list = []
for i in range(1, 19):
    chrom_list.append("chr" + str(i))
chrom_list.append("chrX")
chrom_list.append("chrY")
print(chrom_list)

for target_chrom in chrom_list:
    
    def bigwig_aver_peak(x, bw_file):
        return bw_file.stats(x.chrom, x.start, x.stop, nBins=int(WindowSize / 10))

    def sig_dataset(input_list):
        print(input_list)
        sys.stdout.flush()
        return [bigwig_aver_peak(x, pyBigWig.open(input_list[0])) for x in pybedtools.BedTool(input_list[1])]

    raw_pre_region = args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "pred0.5_pos_regions.bed"
    
    input_list_ATAC=[args.features1_bigwig, raw_pre_region]
    ATAC_sig_pred = sig_dataset(input_list_ATAC)
    print("ATAC_sig finished")
    
    input_list_H3K27ac=[args.features2_bigwig, raw_pre_region]
    H3K27ac_sig_pred = sig_dataset(input_list_H3K27ac)
    print("H3K27ac_sig finished")
    
    input_list_H3K4me3=[args.features3_bigwig, raw_pre_region]
    H3K4me3_sig_pred = sig_dataset(input_list_H3K4me3)
    print("H3K4me3_sig finished")
    
    input_list_RNA=[args.features4_bigwig, raw_pre_region]
    RNA_sig_pred = sig_dataset(input_list_RNA)
    print("RNA_sig finished")

    ATAC_sig_pred = [np.array(i) for i in ATAC_sig_pred]
    H3K27ac_sig_pred = [np.array(i) for i in H3K27ac_sig_pred]
    H3K4me3_sig_pred = [np.array(i) for i in H3K4me3_sig_pred]
    RNA_sig_pred = [np.array(i) for i in RNA_sig_pred]

    pred_pos_region = pybedtools.BedTool().window_maker(
        b=args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "pred0.5_pos_regions.bed",
        n=200)
    print(pred_pos_region.count())
    pred_pos_region.to_csv(
        args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "Sep_pred0.5_pos_regions.bed", sep= "\t", header = False, index = False)

    x_valid_region = []

    Pre_pred_pos_regions = pybedtools.BedTool(
        args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "pred0.5_pos_regions.bed")
    for i in range(Pre_pred_pos_regions.count()):
        x_valid_region.append(
            np.array([ATAC_sig_pred[i], H3K27ac_sig_pred[i], H3K4me3_sig_pred[i], RNA_sig_pred[i]]))

    x_valid_region = np.nan_to_num(np.array(x_valid_region, dtype=float))
    print(x_valid_region.shape)

    x_valid_region = np.expand_dims(np.array(x_valid_region), axis=3)

    print(x_valid_region.shape)
    model = model_create(width=int(window_size / 10))
    earlystop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, mode='auto')
    ADam = Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=9e-5)
    model.compile(loss='binary_crossentropy', optimizer=ADam, metrics=['accuracy', Precision_s, Recall_s, f1_source], experimental_run_tf_function=False)
    model.load_weights(args.species + "." + args.tissue_name+".h5")

    def normalize(x):
        return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

    tf.compat.v1.disable_eager_execution()

    def pred_grad_cam(model, X):
        last_conv_layer = "conv2d_4"

        pred_y = model.output
        print(pred_y.shape)
        Conv_layer_out = model.get_layer(last_conv_layer).output
        print(Conv_layer_out.shape)
        Gradients = K.gradients(pred_y, Conv_layer_out)[0]
        print(Gradients)
        Gradients = normalize(Gradients)
        Gradient_func = K.function([model.input], [Conv_layer_out, Gradients])

        output, Gradients_val = Gradient_func([X])
        weights = np.mean(Gradients_val, axis=(1, 2))

        Cam_res = []
        for i in range(len(weights)):
            temp = np.dot(output[i], weights[i])
            temp = cv2.resize(temp, (200, 1), cv2.INTER_LINEAR)
            temp = np.maximum(temp, 0)
            Cam_res.append(temp[0])

        Cam_res = np.array(Cam_res)
        return Cam_res
 
    print(len(x_valid_region))

    pred_soure_raw = []
    stride = 1000
    for i in range(int(len(x_valid_region) / stride) + 1):
        if (i % 10 == 0):
            print(i, ",000")
        pred_soure_raw.extend(pred_grad_cam(model, x_valid_region[i * stride: (i + 1) * stride]))
    pred_soure_raw = np.array(pred_soure_raw)

    print(stats.describe(pred_soure_raw.flatten()))

    pd_file = pd.read_csv(
        args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "Sep_pred0.5_pos_regions.bed",
        sep="\t", header=None)
    pd_file[4] = "region"
    pd_file[5] = pred_soure_raw.flatten()
    with open(args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "all.pred_pos_regions.bed","w") as all_predpos_region:
        np.savetxt(all_predpos_region, pd_file, fmt='%s', delimiter='\t')    

    Grad_cam_mean = np.mean(pred_soure_raw.flatten())
    pd_file_filter = pd_file[pd_file[5] > Grad_cam_mean]
    with open(args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "filter.all.pred_pos_regions.bed","w") as filter_all_predpos_region:
        np.savetxt(filter_all_predpos_region, pd_file_filter, fmt='%s', delimiter='\t')  

    filter_merge_region = pybedtools.BedTool(
        args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "filter.all.pred_pos_regions.bed").sort().merge(c=5, o="mean")
    with open(args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "filter.merge.pred_pos_regions.bed","w") as filter_merge_predpos_region:
        np.savetxt(filter_merge_predpos_region, filter_merge_region, fmt='%s', delimiter='\t')
 
    all_merge_region = pybedtools.BedTool(
        args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "all.pred_pos_regions.bed").sort().merge(c=5, o="mean")
    with open(args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "all.merge.pred_pos_regions.bed","w") as all_merge_predpos_region:
        np.savetxt(all_merge_predpos_region, all_merge_region, fmt='%s', delimiter='\t')   
        
    pd_file = pd.read_csv(
        args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "filter.merge.pred_pos_regions.bed",
        sep="\t", header=None)
    pd_file[3] = pd_file.index
    with open(args.out_dir + args.species + "_" + args.tissue_name + "_" + target_chrom + "_" + "Final_pred_pos_regions.bed","w") as final_pred_region:
        np.savetxt(final_pred_region, pd_file, fmt='%s', delimiter='\t')
