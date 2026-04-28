#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------
# Read tab-separated RF model gene expression data file into dataframe
RF_expr = pd.read_csv("data_mrna_seq_v2_rsem.txt", sep="\t")

# Select first 500 genes expression data 
rawRF_expr = RF_expr.iloc[:, 2:502]

# Apply log transformation to expression columns
logRF_expr = np.log2(rawRF_expr + 1)

# ------------------------------------------------------------------------------------
# Read CL gene expression data file into dataframe 
CL_expr = pd.read_csv(
	"OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
)

# Select first 500 gene expression data 
rawCL_expr = CL_expr.iloc[:, 6:506]

# -------------------------------------------------------------------------------------
# Make the histograms 

# Set font sizes
SMALL = 13
MEDIUM = 15
BIG = 18

plt.rc('axes', titlesize=BIG)
plt.rc('axes', labelsize=MEDIUM)
plt.rc('xtick', labelsize=SMALL)
plt.rc('ytick', labelsize=SMALL)

# Histogram to show distribution of the data (first 500 values)
fig, axs = plt.subplots(2,2, figsize=(12,10), dpi=300)

# A = Raw RF data
axs[0,0].hist(rawRF_expr, bins=100)
axs[0,0].set_title('A. Raw RSEM Gene Expression', pad=25)
axs[0,0].set_xlabel('Expression (RSEM)', labelpad=15)
axs[0,0].set_ylabel('Frequency (Count)', labelpad=15)

# B = Log-transformed CL data 
axs[0,1].hist(logRF_expr, bins=100)
axs[0,1].set_title('B. Log-Transformed RSEM Gene Expression', pad=25)
axs[0,1].set_xlabel('Expression (Log2(RSEM + 1))', labelpad=15)
axs[0,1].set_ylabel('Frequency (Count)', labelpad=15)

# C = CL data - already log-transformed 
axs[1,0].hist(rawCL_expr, bins=100)
axs[1,0].set_title('C. Log-Transformed TPM Gene Expression', pad=25)
axs[1,0].set_xlabel('Expression (Log2(TPM + 1))', labelpad=15)
axs[1,0].set_ylabel('Frequency (Count)', labelpad=15)

# D = remove empty histogram 
fig.delaxes(axs[1,1])

# Adjust spacing
plt.subplots_adjust(
	left=0.1, 
	bottom=0.1, 
	right=0.9, 
	top=0.9, 
	wspace=0.3, 
	hspace=0.7
)

plt.savefig('Expression_Hist.png')
print("Histogram saved as Expression_Hist.png")


