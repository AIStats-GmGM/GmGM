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

# Process data
import numpy as np
import pandas as pd
import igraph as ig
from Backends.GmGM import GmGM
from Backends.utilities import shrink_per_row, shrink_sparsities
import muon as mu
import scanpy as sc
from anndata import AnnData

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Read in from Data/LL-Deep\ Data\ -\ Processed
# The files Map.csv, Metabolomics.csv, and MetagenomicsShotgun.csv

# Read in the map
map_df = pd.read_csv(
    'Data/LL-Deep Data - Processed/Map.csv',
    index_col=0
)

# Read in the metabolomics data
metabolomics_df = pd.read_csv(
    'Data/LL-Deep Data - Processed/Metabolomics.csv',
    index_col=0
)

# Read in the metagenomics shotgun data
metagenomics_shotgun_df = pd.read_csv(
    'Data/LL-Deep Data - Processed/MetagenomicsShotgun.csv',
    index_col=0
)

# Recall that the nth row in one df is for the same
# patient as the nth row in the other df, by construction
# in `Map-Data.ipynb`

metabol_ann = AnnData(
    X = metabolomics_df.to_numpy(),
)
metabol_ann.obs_names = metabolomics_df.index
metabol_ann.var_names = metabolomics_df.columns

metagen_ann = AnnData(
    X = metagenomics_shotgun_df.to_numpy(),
)
metagen_ann.obs_names = metabolomics_df.index
metagen_ann.var_names = metagenomics_shotgun_df.columns

mudata = mu.MuData({
    'metabolomics': metabol_ann,
    'metagenomics_shotgun': metagen_ann
})

# Add the gender data
mudata.obs["Gender"] = map_df["Gender"].to_numpy()

# Log transform the data
# (These modify in-place)
sc.pp.log1p(mudata['metabolomics'])
sc.pp.log1p(mudata['metagenomics_shotgun'])

# Load file Data/LL-Deep Data - Processed/kept-species-ziln.csv
kept_species_ziln_df = pd.read_csv(
    "Data/LL-Deep Data - Processed/kept-species-ziln.csv",
    index_col=0,
    header=0,
)
shotgun_array = \
    mudata["metagenomics_shotgun"][:, kept_species_ziln_df["x"].values].X
metabol_array = mudata["metabolomics"].X

for_comparison_with_ziln = GmGM()(
    {
        #("people", "metabolomics"): metabol_array[np.newaxis, ...],
        ("people", "species"): shotgun_array[np.newaxis, ...],
    },
    verbose=True,
    verbose_every=1_000,
    max_iter=10_000,
    tol=1e-8,
)

for_comparison = for_comparison_with_ziln["species"]

# Save for_comparison to csv
np.savetxt(
    "Data/LL-Deep Data - Processed/GmGM-Microbes-Subset.csv",
    for_comparison,
    delimiter=","
)

# We found we had to scale the metagenomics data for this to work
# The running theory is that the metabolomics data is not very helpful
# for the species graph, and "distracts" the optimization method.
# Numerical precision issues are also likely involved, since it sometimes
# works without scaling too - just not consistently.
for_comparison_with_ziln_2 = GmGM()(
    {
        ("people", "metabolomics"): metabol_array[np.newaxis, ...],
        ("people", "species"): 10*shotgun_array[np.newaxis, ...],
    },
    verbose=True,
    verbose_every=1000,
    max_iter=10_000,
    tol=1e-8,
)

for_comparison = for_comparison_with_ziln_2["species"]

# Save for_comparison to csv
np.savetxt(
    "Data/LL-Deep Data - Processed/GmGM-Microbes-Subset-Metabolites.csv",
    for_comparison,
    delimiter=","
)