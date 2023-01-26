# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import pybedtools
from pybedtools import featurefuncs
import pyBigWig

data_analysis = argparse.ArgumentParser(description='Preprocessing data and constructing the raw input data matrix')
data_analysis.add_argument('--featrue1_peaks', type=str, help='chromatin accessibility peak')
data_analysis.add_argument('--featrue2_peaks', type=str, help='ChIP-seq H3K27ac peak')
data_analysis.add_argument('--featrue3_peaks', type=str, help='ChIP-seq H3K4me3 peak')
data_analysis.add_argument('--featrue4_peaks', type=str, help='RNA-seq  peak')
data_analysis.add_argument('--featrue1_bw', type=str, help='chromatin accessibility bigWig')
data_analysis.add_argument('--featrue2_bw', type=str, help='ChIP-seq H3K27ac bigWig')
data_analysis.add_argument('--featrue3_bw', type=str, help='ChIP-seq H3K4me3 bigWig')
data_analysis.add_argument('--featrue4_bw', type=str, help='RNA-seq  bigWig')
data_analysis.add_argument('--outpath', type=str, help='output_directory')
data_analysis.add_argument('--species', type=str, help='name of the cell')
data_analysis.add_argument('--tissue_name', type=str, help='name of the tissue')
data_analysis.add_argument('--window_size', type=int, help='prediction window size')

tissue_name = "Muscle"

argument_str = ' --featrue1_peaks ' + "Duroc.Muscle.ATAC-seq.narrowPeak" + \
              ' --featrue2_peaks ' + "Duroc.Muscle.ChIP-seq.H3K27ac.narrowPeak" + \
              ' --featrue3_peaks ' + "Duroc.Muscle.ChIP-seq.H3K4me3.narrowPeak" + \
              ' --featrue4_peaks ' + "Duroc.Muscle.RNA_TMP.bed" + \
              ' --featrue1_bw ' + "Duroc.Muscle.ATAC-seq.bigWig" + \
              ' --featrue2_bw ' + "Duroc.Muscle.ChIP-seq.H3K27ac.bigWig" + \
              ' --featrue3_bw ' + "Duroc.Muscle.ChIP-seq.H3K4me3.bigWig" + \
              ' --featrue4_bw ' + "Duroc.Muscle.RNA-seq.bigWig" + \
              ' --outpath ' + "/user/xx/output/" + \
              ' --tissue_name ' + tissue_name + \
              ' --species ' + "Duroc" + \
              ' --window_size ' + "4000"

args = data_analysis.parse_args(argument_str.split())

for key, value in vars(args).items():
    if type(value) is list:
        for v in value:
            if not os.path.exists(v):
                print(key + " argument file does not exist")
                exit(1)
    elif key == "outpath" or key == "cell_name" or key == "pos_neg_ratio" or key == "window_size":
        continue
    else:
        if not os.path.exists(value):
            print(key + " argument file does not exist")
            exit(1)
print("all files found!")
chrom_list = []
for i in range(1, 19):
    chrom_list.append("chr" + str(i))
chrom_list.append("chrX")
chrom_list.append("chrY")
print(chrom_list)

os.system("mkdir -p " + args.outpath)

ATAC_signal = pybedtools.BedTool(args.featrue1_peaks).sort().merge()
chip1 = pybedtools.BedTool(args.featrue2_peaks).sort().merge()
chip2 = pybedtools.BedTool(args.featrue3_peaks).sort().merge()
RNA = pybedtools.BedTool(args.featrue4_peaks).sort().merge()
ATAC_signal_and_chip1 = ATAC_signal.intersect(chip1, sorted=True).filter(pybedtools.featurefuncs.greater_than, 20).sort()
ATAC_signal_and_chip2 = ATAC_signal.intersect(chip2, sorted=True).filter(pybedtools.featurefuncs.greater_than, 20).sort()
ATAC_signal_and_RNA = ATAC_signal.intersect(RNA, sorted=True).filter(pybedtools.featurefuncs.greater_than, 20).sort()

raw_trainset = ATAC_signal_and_chip1.cat(ATAC_signal_and_chip2).cat(ATAC_signal_and_RNA).filter(
    lambda x: x.chrom in chrom_list)
positive_trainset = raw_trainset.each(pybedtools.featurefuncs.midpoint).slop(b=args.window_size / 2,
                                                                                genome="susScr11").filter(
    pybedtools.featurefuncs.greater_than, args.window_size - 1).sort()

susScr11_windows = pybedtools.BedTool().window_maker(genome="susScr11", w=args.window_size).filter(
    pybedtools.featurefuncs.greater_than, args.window_size - 1).filter(lambda x: x.chrom in chrom_list)
susScr11_windows = susScr11_windows - positive_trainset
print("original negative window: " + str(susScr11_windows.count()))
susScr11_windows.saveas(args.outpath + args.species + "." + args.tissue_name + ".susScr11_windows.bed")

# downsample negative to 10x of positive
negative_trainset = susScr11_windows.random_subset(positive_trainset.count() * 10).sort()
negative_trainset.saveas(args.outpath + args.species + "." + args.tissue_name + ".negative.bed")

# IO the bigwig signals
ATAC_signal_bw = pyBigWig.open(args.featrue1_bw)
chip1_bw = pyBigWig.open(args.featrue2_bw)
chip2_bw = pyBigWig.open(args.featrue3_bw)
RNA_bw = pyBigWig.open(args.featrue4_bw)

def bigWigAverageOverBed(x, bigwig):
    return bigwig.stats(x.chrom, x.start, x.stop, nBins=400)

def get_signal(region, bigwig):
    return np.array([np.nan_to_num(np.array(bigWigAverageOverBed(x, bigwig), dtype=float)) for x in region])

# generate some random data
data = np.random.randn(200)
d = [data, data]

pos_matrix = get_signal(
    pybedtools.BedTool(args.outpath + args.species + "." + args.tissue_name + "." + "positive.bed"), ATAC_signal_bw)
neg_matrix = get_signal(
    pybedtools.BedTool(args.outpath + args.species + "." + args.tissue_name + "." + "negative.bed"), ATAC_signal_bw)
np.savetxt(args.outpath + args.species + "." + args.tissue_name + "." + "ATAC" + ".pos.tsv", pos_matrix, fmt='%s',
           delimiter='\t')
np.savetxt(args.outpath + args.species + "." + args.tissue_name + "." + "ATAC" + ".neg.tsv", neg_matrix, fmt='%s',
           delimiter='\t')

pos_matrix = get_signal(
    pybedtools.BedTool(args.outpath + args.species + "." + args.tissue_name + "." + "positive.bed"), chip1_bw)
neg_matrix = get_signal(
    pybedtools.BedTool(args.outpath + args.species + "." + args.tissue_name + "." + "negative.bed"), chip1_bw)
np.savetxt(args.outpath + args.species + "." + args.tissue_name + "." + "H3K27ac" + ".pos.tsv", pos_matrix, fmt='%s',
           delimiter='\t')
np.savetxt(args.outpath + args.species + "." + args.tissue_name + "." + "H3K27ac" + ".neg.tsv", neg_matrix, fmt='%s',
           delimiter='\t')

pos_matrix = get_signal(
    pybedtools.BedTool(args.outpath + args.species + "." + args.tissue_name + "." + "positive.bed"), chip2_bw)
neg_matrix = get_signal(
    pybedtools.BedTool(args.outpath + args.species + "." + args.tissue_name + "." + "negative.bed"), chip2_bw)
np.savetxt(args.outpath + args.species + "." + args.tissue_name + "." + "H3K4me3" + ".pos.tsv", pos_matrix, fmt='%s',
           delimiter='\t')
np.savetxt(args.outpath + args.species + "." + args.tissue_name + "." + "H3K4me3" + ".neg.tsv", neg_matrix, fmt='%s',
           delimiter='\t')

pos_matrix = get_signal(
    pybedtools.BedTool(args.outpath + args.species + "." + args.tissue_name + "." + "positive.bed"), RNA_bw)
neg_matrix = get_signal(
    pybedtools.BedTool(args.outpath + args.species + "." + args.tissue_name + "." + "negative.bed"), RNA_bw)
np.savetxt(args.outpath + args.species + "." + args.tissue_name + "." + "RNA" + ".pos.tsv", pos_matrix, fmt='%s',
           delimiter='\t')
np.savetxt(args.outpath + args.species + "." + args.tissue_name + "." + "RNA" + ".neg.tsv", neg_matrix, fmt='%s',
           delimiter='\t')
