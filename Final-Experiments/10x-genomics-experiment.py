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

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Data manipulation
import scanpy as sc
import muon as mu
import numpy as np
import pandas as pd
import sklearn.cluster as clust
from scipy import sparse, io
import igraph as ig
import gseapy

# Utilities
import os

# Custom functions
from Backends.GmGM import GmGM
from Backends.utilities import shrink_sparsities
from Backends.utilities import shrink_per_row

import argparse

import random
random.seed(0)

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
    default="/home/bailey/Desktop/10x-scRNA-ATAC/",
    help="Path to the data directory"
)

parser.add_argument(
    "--show",
    action="store_true",
    help="Shows plots that are created"
)

args = parser.parse_args()

# Load the data
mudata = mu.read_10x_h5(
    os.path.join(
        args.path,              
        "lymph_node_lymphoma_14k_filtered_feature_bc_matrix.h5"
    ),
)
if args.verbose:
    print("Unprocessed data:")
    print(mudata)

rna = mudata["rna"]
rna.var_names_make_unique()

atac = mudata["atac"]
atac.var_names_make_unique()
mudata.update()

##############################
### Preprocessing
##############################

# Find the mitochondrial genes
# We should expect these to be lowly expressed, because
# this is a single-nucleus experiment and the
# mitochondria are in the cytoplasm, not the nucleus
rna.var['mt'] = rna.var_names.str.startswith("MT-")

#===#
# Filter out low quality cells
#===#

# QC = quality control
sc.pp.calculate_qc_metrics(
    rna,
    qc_vars=['mt'],
    percent_top=None,
    log1p=False,
    inplace=True
)

rna.obs["log_genes_by_counts"] = np.log10(
    rna.obs["n_genes_by_counts"] + 1
)

# Calculate median absolute deviation of n_genes_by_counts
med = np.median(rna.obs['log_genes_by_counts'])
mad = np.median(
    np.abs(
        rna.obs['log_genes_by_counts'] - med
    )
)

# Filter out cells with any mitochondrial genes
# Because this is a single-nucleus experiment, and we expect
# the mitochondria to be in the cytoplasm (not the nucleus),
# any presence is likely to be a technical artifact.
mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x == 0)

# Filter out cells with too many or too few genes
# (Based on 3 median absolute deviations from the median)
mu.pp.filter_obs(
    rna,
    'log_genes_by_counts',
    lambda x: (x >= med - 3*mad) & (x < med + 3*mad)
)

#===#
# Filter out low quality genes
#===#

rna.var["log_cells_by_counts"] = np.log10(
    rna.var["n_cells_by_counts"]+1
)
med = np.median(rna.var["log_cells_by_counts"])
mad = np.median(
    np.abs(
        rna.var["log_cells_by_counts"] - med
    )
)

# Filter out genes with less than 3 cells
mu.pp.filter_var(
    rna,
    'n_cells_by_counts',
    lambda x: x >= 3
)

# Filter out genes outside of 3 median absolute deviations
# Note that this won't do anything, all are within 3 MADs
mu.pp.filter_var(
    rna,
    'log_cells_by_counts',
    lambda x: (x >= med - 3*mad) & (x < med + 3*mad)
)

#===#
# Normalize genes
#===#

sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)

# Find and label highly variable genes
sc.pp.highly_variable_genes(
    rna,
    min_mean=0.02,
    max_mean=4,
    min_disp=0.5
)

#===#
# Filter out low quality peaks
#===#

# Note, because we're using `scanpy`, our peaks
# will be referred to as "genes" here.
sc.pp.calculate_qc_metrics(
    atac,
    percent_top=None,
    log1p=False,
    inplace=True
)

atac.obs['log_total_counts'] = np.log10(
    atac.obs['total_counts']+1
)
atac.obs['log_genes_by_counts'] = np.log10(
    atac.obs['n_genes_by_counts']+1
)
atac.var['log_cells_by_counts'] = np.log10(
    atac.var['n_cells_by_counts']+1
)

# Filter out total_counts within 3 MADs of the median
med = np.median(atac.obs['log_total_counts'])
mad = np.median(np.abs(atac.obs['log_total_counts'] - med))
mu.pp.filter_obs(
    atac,
    'log_total_counts',
    lambda x: (x >= med - 3*mad) & (x < med + 3*mad)
)

# Filter out n_genes_by_counts within 3 MADs of the median
med = np.median(atac.obs['log_genes_by_counts'])
mad = np.median(np.abs(atac.obs['log_genes_by_counts'] - med))
mu.pp.filter_obs(
    atac,
    'log_genes_by_counts',
    lambda x: (x >= med - 3*mad) & (x < med + 3*mad)
)

# Filter out n_cells_by_counts within 3 MADs of the median
med = np.median(atac.var['log_cells_by_counts'])
mad = np.median(np.abs(atac.var['log_cells_by_counts'] - med))
mu.pp.filter_var(
    atac,
    'log_cells_by_counts',
    lambda x: (x >= med - 3*mad) & (x < med + 3*mad)
)

#===#
# Normalize peaks
#===#

mu.atac.pp.tfidf(atac, scale_factor=1e4)
sc.pp.normalize_per_cell(atac, counts_per_cell_after=1e4)
sc.pp.log1p(atac)

# Find and label highly variable peaks ("genes")
sc.pp.highly_variable_genes(
    atac,
    min_mean=0.05,
    max_mean=1.5,
    min_disp=0.5
)

mudata.update()

##############################
### Run GmGM
##############################

cells_that_passed_both_tests = np.intersect1d(
    rna.obs_names,
    atac.obs_names
)

# We have to copy b/c can't call `filter_var` on a view :(
filtered_mudata = mudata[cells_that_passed_both_tests].copy()

# Grab just the highly variable genes and peaks
mu.pp.filter_var(
    filtered_mudata,
    'highly_variable',
    lambda x: x
)

def filter_func(x):
    # Had to make a fancy function b/c it was hard
    # to get the vectorization to work concisely in
    # a lambda function
    out = np.zeros_like(x, dtype=bool)
    out |= np.random.randint(0, 4, size=len(x)) == 0
    out |= x == 'Gene Expression'
    return out

# Grab 25% of the peaks for now
# We are constrained by the RAM of our machine!
mu.pp.filter_var(
    filtered_mudata,
    'feature_types',
    filter_func
)

cell_by_gene = filtered_mudata['rna'].X
cell_by_atac = filtered_mudata['atac'].X

if args.verbose:
    print("Filtered dataset:")
    print(filtered_mudata)

output = GmGM()({
    ("cell", "gene"): cell_by_gene.toarray(),
    ("cell", "atac"): cell_by_atac.toarray()
}, verbose=args.verbose)

shrink_per_row(
    output,
    {"cell": 5, "gene": 5, "atac": 5},
    safe=False
)

#===#
# Relationship between Graph and UMAP-Space
#===#

if args.verbose:
    print("Calculating UMAP...")

# First run PCA to ease computational load
sc.pp.pca(filtered_mudata['rna'])
sc.pp.pca(filtered_mudata['atac'])

# Have to find nearest neighbors on all modalities
sc.pp.neighbors(filtered_mudata['rna'])
sc.pp.neighbors(filtered_mudata['atac'])

# Now can do joint neighbor network construction
mu.pp.neighbors(filtered_mudata)

# Now can do UMAP
mu.tl.umap(filtered_mudata)

# Color the UMAP Plot by kmeans
cluster = clust.KMeans(
    n_clusters=3,
    random_state=0
).fit(filtered_mudata.obsm['X_umap'])
filtered_mudata.obs['kmeans'] = cluster.labels_.astype(str)

# Color palette
def color_vx(x):
    x = int(x)
    if x == 0: return 'red'
    if x == 1: return 'green'
    if x == 2: return 'blue'
    if x == 3: return 'magenta'
    if x == 4: return 'cyan'
    if x == 5: return 'black'
    if x == 6: return 'orange'
    if x == 7: return 'grey'

# Create igraph from GmGM output
cells_mat = np.abs(output['cell'])
cells_mat = cells_mat + cells_mat.T
cell_graph = ig.Graph.Adjacency(
    cells_mat != 0,
    mode="undirected",
    weighted=True
)

# Fill in cluster information
cell_graph.vs['kmeans_cluster'] = filtered_mudata.obs['kmeans']
cell_graph.vs['kmeans_color'] = [
    color_vx(x) for x in filtered_mudata.obs['kmeans']
]

# Plot cell_graph in UMAP-space
ig.plot(
    cell_graph,
    layout=filtered_mudata.obsm['X_umap'],
    vertex_size=3,
    edge_width=0.1,
    vertex_color=cell_graph.vs['kmeans_color'],
    bbox=(0, 0, 500, 500),
    target="Final-Plots/cell_graph_umap.svg"
)

ig.plot(
    cell_graph,
    vertex_size=3,
    edge_width=0.1,
    vertex_color=cell_graph.vs['kmeans_color'],
    bbox=(0, 0, 500, 500),
    target="Final-Plots/cell_graph.svg"
)

# Do louvain clustering on cell_graph
cell_graph.vs['louvain_cluster'] = cell_graph.community_multilevel().membership
cell_graph.vs['louvain_color'] = [
    color_vx(x) for x in cell_graph.vs['louvain_cluster']
]

# Plot the louvain coloring on our prior graph
ig.plot(
    cell_graph,
    vertex_size=5,
    edge_width=0.1,
    vertex_color=cell_graph.vs['louvain_color'],
    bbox=(0, 0, 500, 500),
    target="Final-Plots/cell_graph_louvain.svg"
)

# What do these look like in UMAP-space?
ig.plot(
    cell_graph,
    layout=filtered_mudata.obsm['X_umap'],
    vertex_size=3,
    edge_width=0.1,
    vertex_color=cell_graph.vs['louvain_color'],
    bbox=(0, 0, 500, 500),
    target="Final-Plots/cell_graph_louvain_umap.svg"
)

# Store clustering in the mudata object
filtered_mudata.obs['louvain']\
    = cell_graph.vs['louvain_cluster']
filtered_mudata.obs['louvain']\
    = filtered_mudata.obs['louvain'].astype(str)

# Move clustering down to the rna level for DE analysis
filtered_mudata['rna'].obs['kmeans']\
    = filtered_mudata.obs['kmeans']
filtered_mudata['rna'].obs['louvain']\
    = filtered_mudata.obs['louvain']

fig, ax = plt.subplots()
mu.pl.umap(
    filtered_mudata,
    color=['louvain'],
    ax=ax,
    show=args.show
)
fig.savefig("Final-Plots/umap_louvain.svg")
plt.close(fig)

#===#
# GO Term Analysis
#===#

# Get the gene sets for humans
gene_set_names = gseapy.get_library_name(organism='Human')

# Louvain DE genes
sc.tl.rank_genes_groups(
    filtered_mudata['rna'],
    'louvain',
    method='wilcoxon',
    key_added='louvain_de'
)

# Dotplot
sc.tl.dendrogram(filtered_mudata['rna'], groupby='louvain')
fig, ax = plt.subplots()
sc.pl.rank_genes_groups_dotplot(
    filtered_mudata['rna'],
    n_genes=5,
    key="louvain_de",
    groupby="louvain",
    ax=ax,
    show=args.show
)
fig.savefig("Final-Plots/louvain_de_dotplot.svg")
plt.close(fig)

if args.verbose:
    print("Calculating GO term enrichment for louvain clusters...")

# louvain
glist_louvain = [None, None, None, None, None, None, None, None]
for i in range(8):
    glist_louvain[i] = sc.get.rank_genes_groups_df(
        filtered_mudata['rna'],
        key='louvain_de',
        group=f'{i}',
        pval_cutoff=0.05
    )
    glist_louvain[i]['names'].squeeze().str.strip()

# louvain enrichment
enr_res_louvain = [None, None, None, None, None, None, None, None]
for i in range(8):
    enr_res_louvain[i] = gseapy.enrichr(
        gene_list=glist_louvain[i][['names', 'logfoldchanges']],
        organism='Human',
        gene_sets=[
            'GO_Biological_Process_2018',
            'GO_Cellular_Component_2018',
            'GO_Molecular_Function_2018',
            'KEGG_2019_Human',
            'Reactome_2016',
            'WikiPathways_2019_Human',
            'NCI-Nature_2016',
            'Panther_2016',
        ]
    )

for i in range(8):
    try:
        gseapy.barplot(
            enr_res_louvain[i].res2d,
            title=f'GO Terms Cluster {i}',
            ofname=f"Final-Plots/GO_Terms_Cluster_{i}.svg"
        )
    except:
        if args.verbose:
            print(f'None in cluster {i}')

if args.verbose:
    print("Done!")