# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import pandas as pd
import datetime
from tempfile import TemporaryFile
import os
import pybedtools
from pybedtools import BedTool
from pybedtools import featurefuncs
import pyBigWig

data_analysis = argparse.ArgumentParser()
data_analysis.add_argument('--featrue1_peaks', type=str)
data_analysis.add_argument('--featrue2_peaks', type=str)
data_analysis.add_argument('--featrue3_peaks', type=str)
data_analysis.add_argument('--featrue4_peaks', type=str)
data_analysis.add_argument('--featrue1_bigwig', type=str)
data_analysis.add_argument('--featrue2_bigwig', type=str)
data_analysis.add_argument('--featrue3_bigwig', type=str)
data_analysis.add_argument('--featrue4_bigwig', type=str)
data_analysis.add_argument('--outpath', type=str)
data_analysis.add_argument('--species', type=str)
data_analysis.add_argument('--tissue_name', type=str)
data_analysis.add_argument('--WindowSize', type=int)

tissue_name = "Muscle"

argumentStr = ' --featrue1_peaks ' + "xx.ATAC.narrowPeak" + \
              ' --featrue2_peaks ' + "xx.H3K27ac.narrowPeak" + \
              ' --featrue3_peaks ' + "xx.H3K4me3.narrowPeak" + \
              ' --featrue4_peaks ' + "xx.RNA.bed" + \
              ' --featrue1_bigwig ' + "xx.ATAC.bigwig" + \
              ' --featrue2_bigwig ' + "xx.H3K27ac.bigwig" + \
              ' --featrue3_bigwig ' + "xx.H3K4me3.bigwig" + \
              ' --featrue4_bigwig ' + "xx.RNA.bigwig" + \
              ' --outpath ' + "/user/xx/output/" + \
              ' --tissue_name ' + tissue_name + \
              ' --species ' + "xxx" + \
              ' --windowSize ' + "4000"

args = data_analysis.parse_args(argumentStr.split())
chrom_list = []
for i in range(1,19):
    chrom_list.append("chr" + str(i))
chrom_list.append("chrX")
chrom_list.append("chrY")
print(chrom_list)

ATAC_signal = pybedtools.BedTool(args.featrue1_peaks).sort().merge()
H3K27ac_signal = pybedtools.BedTool(args.featrue2_peaks).sort().merge()
H3K4me3_signal = pybedtools.BedTool(args.featrue3_peaks).sort().merge()
RNA_signal = pybedtools.BedTool(args.featrue4_peaks).sort().merge()
ATAC_signal_and_H3K27ac_signal = ATAC_signal.intersect(H3K27ac_signal, sorted=True)
ATAC_signal_and_H3K4me3_signal = ATAC_signal.intersect(H3K4me3_signal, sorted=True)
ATAC_signal_and_RNA_signal = ATAC_signal.intersect(RNA_signal, sorted=True)

raw_trainset = ATAC_signal_and_H3K27ac_signal.cat(ATAC_signal_and_H3K4me3_signal).cat(ATAC_signal_and_RNA_signal).filter(
    lambda x: x.chrom in chrom_list)
positive_trainset = raw_trainset.each(pybedtools.featurefuncs.midpoint).slop(b=args.WindowSize / 2,
                                                                                genome="susScr11").sort()

with open(args.outpath + args.species + "." + args.tissue_name + ".pos.txt","w") as Sus_pos:
        np.savetxt(Sus_pos, postive_trainset, fmt='%s', delimiter='\t')  
                                                                               

susScr11_windows = pybedtools.BedTool().window_maker(genome="susScr11", w=args.WindowSize).filter(lambda x: x.chrom in chrom_list)
susScr11_windows = susScr11_windows - positive_trainset

with open(args.outpath + args.species + "." + args.tissue_name + ".susScr11_ws.txt","w") as Sus_ws:
        np.savetxt(Sus_ws, susScr11_windows, fmt='%s', delimiter='\t')  

negative_trainset = susScr11_windows.random_subset(positive_trainset.count() * 10).sort()
with open(args.outpath + args.species + "." + args.tissue_name + ".neg.txt","w") as Sus_neg:
        np.savetxt(Sus_neg, negative_trainset, fmt='%s', delimiter='\t')  

def bigwig_aver_peak(x, bw_file):
    return bw_file.stats(x.chrom, x.start, x.stop, nBins=400)

def sig_dataset(input_list):
    print(input_list)
    sys.stdout.flush()
    return np.array([np.nan_to_num(np.array(bigwig_aver_peak(x, pyBigWig.open(input_list[0])), dtype=float)) for x in pybedtools.BedTool(input_list[1])])


positive_region = args.outpath + args.species + "_" + args.tissue_name + "_" + "positive.bed"
negative_region = args.outpath + args.species + "_" + args.tissue_name + "_" + "negative.bed"


input_list_ATAC_pos=[args.featrue1_bigwig, positive_region]
input_list_ATAC_neg=[args.featrue1_bigwig, negative_region]
pos_matrix = sig_dataset(input_list_ATAC_pos)
neg_matrix = sig_dataset(input_list_ATAC_neg)
with open(args.outpath + args.species + "_" + args.tissue_name + "_" + "ATAC" + "_postive.tsv","w") as ATAC_pos:
    np.savetxt(ATAC_pos, pos_matrix, fmt='%s', delimiter='\t')
with open(args.outpath + args.species + "_" + args.tissue_name + "_" + "ATAC" + "_negative.tsv","w") as ATAC_neg:
    np.savetxt(ATAC_neg, neg_matrix, fmt='%s', delimiter='\t')


input_list_H3K27ac_pos=[args.featrue2_bigwig, positive_region]
input_list_H3K27ac_neg=[args.featrue2_bigwig, negative_region]
pos_matrix = sig_dataset(input_list_H3K27ac_pos)
neg_matrix = sig_dataset(input_list_H3K27ac_neg)
with open(args.outpath + args.species + "_" + args.tissue_name + "_" + "H3K27ac" + "_postive.tsv","w") as H3K27ac_pos:
    np.savetxt(H3K27ac_pos, pos_matrix, fmt='%s', delimiter='\t')
with open(args.outpath + args.species + "_" + args.tissue_name + "_" + "H3K27ac" + "_negative.tsv","w") as H3K27ac_neg:
    np.savetxt(H3K27ac_neg, neg_matrix, fmt='%s', delimiter='\t')


input_list_H3K4me3_pos=[args.featrue3_bigwig, positive_region]
input_list_H3K4me3_neg=[args.featrue3_bigwig, negative_region]
pos_matrix = sig_dataset(input_list_H3K4me3_pos)
neg_matrix = sig_dataset(input_list_H3K4me3_neg)
with open(args.outpath + args.species + "_" + args.tissue_name + "_" + "H3K4me3" + "_postive.tsv","w") as H3K4me3_pos:
    np.savetxt(H3K4me3_pos, pos_matrix, fmt='%s', delimiter='\t')
with open(args.outpath + args.species + "_" + args.tissue_name + "_" + "H3K27ac" + "_negative.tsv","w") as H3K4me3_neg:
    np.savetxt(H3K4me3_neg, neg_matrix, fmt='%s', delimiter='\t')


input_list_RNA_pos=[args.featrue4_bigwig, positive_region]
input_list_RNA_neg=[args.featrue4_bigwig, negative_region]
pos_matrix = sig_dataset(input_list_RNA_pos)
neg_matrix = sig_dataset(input_list_RNA_neg)
with open(args.outpath + args.species + "_" + args.tissue_name + "_" + "RNA" + "_postive.tsv","w") as RNA_pos:
    np.savetxt(RNA_pos, pos_matrix, fmt='%s', delimiter='\t')
with open(args.outpath + args.species + "_" + args.tissue_name + "_" + "H3K27ac" + "_negative.tsv","w") as RNA_neg:
    np.savetxt(RNA_neg, neg_matrix, fmt='%s', delimiter='\t')
    