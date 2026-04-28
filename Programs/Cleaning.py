#!/usr/bin/env python

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------------
# Make pandas data frame

# Read gene expression data file into dataframe
df_expr = pd.read_csv(
	"data_mrna_seq_v2_rsem.txt", sep="\t"
)

# ------------------------------------------------------------------------------------
# Load and clean data

# Remove Entrez_Gene_Id column
df_expr = df_expr.drop(
	columns=["Entrez_Gene_Id"],
	errors="ignore"
)

# Set Hugo Symbol as index and transpose data
# Rows = patients, columns = genes
df_expr = df_expr.set_index("Hugo_Symbol").T

# Make patients IDs a column, instead of an index
df_expr.index.name = "Patient"
df_expr = df_expr.reset_index()

# Remove leftover column index name
df_expr.columns.name = None

# Read clinical data into dataframe, skip first 4 rows
df_clin = pd.read_csv(
        "data_clinical_sample.txt", sep="\t", skiprows=4
)

# Keep only sample id and detailed cancer type data
df_clin = df_clin[["SAMPLE_ID", "CANCER_TYPE_DETAILED"]]

# Remove any hidden whitespace
df_clin["SAMPLE_ID"] = df_clin["SAMPLE_ID"].str.strip()
df_expr["Patient"] = df_expr["Patient"].str.strip()

# Merge expression and clinical data
df_final = df_expr.merge(
        df_clin,
        left_on="Patient",
        right_on="SAMPLE_ID",
        how="left"
)

# Drop duplicate sample id column after merge
df_final = df_final.drop(columns=["SAMPLE_ID"])

# ----------------------------------------------------------------------------
# Preprocess data

# Keep only ductal and lobular subtypes
subtypes = [
        "Breast Invasive Ductal Carcinoma",
        "Breast Invasive Lobular Carcinoma"
]
df_subset = df_final[
        df_final["CANCER_TYPE_DETAILED"].isin(subtypes)
].copy()

# Create binary labels
df_subset["Subtype_Label"] = df_subset["CANCER_TYPE_DETAILED"].map({
        "Breast Invasive Ductal Carcinoma": 0,
        "Breast Invasive Lobular Carcinoma": 1
})

# Remove duplicate gene columns
df_subset = df_subset.loc[:, ~df_subset.columns.duplicated()]

# Select gene expression values
gene_cols = df_subset.columns.difference([
        "Patient", "CANCER_TYPE_DETAILED", "Subtype_Label"])

# Log transform gene expression values
df_subset[gene_cols] = np.log2(
        df_subset[gene_cols].astype(float) + 1
).fillna(0)

# Verify counts
print(df_subset["Subtype_Label"].value_counts())


# Save cleaned data set 
df_subset.to_csv("cleaned_gene_expression.csv", index=False)

print("Cleaned data saved as cleaned_gene_expression.csv")


# Check the data frame 
df = pd.read_csv("cleaned_gene_expression.csv")
print(df.shape)
print(df.columns)
print(df.head())
