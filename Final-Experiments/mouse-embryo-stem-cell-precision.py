import sys
import os

# Add parent directory to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(
                __file__
            )
        )
    )
)

# Data processing
import numpy as np
import pandas as pd
from Backends.GmGM import GmGM
from Backends.utilities import shrink_per_row, shrink_sparsities
import igraph as ig

# Plotting
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(
    description=\
        "Creates a video of the duck from the E-MTAB-2805 dataset"
)

parser.add_argument(
    "-v",
    "--verbose",
    type=int,
    default=1,
    help="Increase verbosity (0-1)"
)

parser.add_argument(
    "-p",
    "--path",
    type=str,
    default="Data/",
    help="Path to the data directory"
)

parser.add_argument(
    "--show",
    action="store_true",
    help="Shows plots that are created"
)

parser.add_argument(
    "--dont-save",
    action="store_false",
    help="If passed, don't save data",
    dest="save"
)

args = parser.parse_args()

# Read the data, limiting ourselves to the same genes
# as in the scBiGLasso paper
mitosis_genes = pd.read_csv('./Data/Nmythosis.txt')
G1_df = pd.read_csv(
    './Data/G1_singlecells_counts.txt',
    sep='\t'
).dropna()
G2M_df = pd.read_csv(
    './Data/G2M_singlecells_counts.txt',
    sep='\t'
).dropna()
S_df = pd.read_csv(
    './Data/S_singlecells_counts.txt',
    sep='\t'
).dropna()
G1_df = G1_df[G1_df['EnsemblGeneID'].isin(
    mitosis_genes['Genes related to mitosis']
)]
G2M_df = G2M_df[G2M_df['EnsemblGeneID'].isin(
    mitosis_genes['Genes related to mitosis']
)]
S_df = S_df[S_df['EnsemblGeneID'].isin(
    mitosis_genes['Genes related to mitosis']
)]

# Get matrix of counts
G1_mat = G1_df.iloc[:, 4:].to_numpy()
G2M_mat = G2M_df.iloc[:, 4:].to_numpy()
S_mat = S_df.iloc[:, 4:].to_numpy()
counts = np.concatenate([S_mat, G1_mat, G2M_mat], axis=1).T[np.newaxis, ...]
log_counts = np.log(counts + 1)

precisions = GmGM()(
    {('cells', 'genes'): log_counts}
)

precisions_shrunk = shrink_per_row(
    precisions,
    ns={
        "cells": 100,
        "genes": 100,
    },
    safe=True
)

fig, ax = plt.subplots(ncols=1)
ax.imshow(
    precisions_shrunk["cells"] != 0
)
ax.set_title("Mouse Embryo Stem Cells", fontsize=24)
ax.set_axis_off()

if args.save:
    plt.savefig(
        "Final-Plots/mouse-embryo-cells-precision.svg",
        dpi=300
    )

if args.show:
    plt.show()