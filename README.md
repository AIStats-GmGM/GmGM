WARNING: This branch represents old code that is unpolished.  For the cleaned code that exists in the final paper, please use the other branch.

# GmGM: Gaussian multi-Graphical Model

This repository contains the implementation of and experiments on
our algorithm presented in the paper "GmGM: a fast multi-axis Gaussian graphical
model".  (TODO: Add arXiv link, once its up)

It is meant to find graphs that represent data, in the multi-modal context.
For example, consider the following dataset:

* A metagenomics matrix of 1000 people x 2000 species
* A metabolomics matrix of 1000 people x 200 metabolites

We may be interested in graph representations of the people, species, and metabolites.
Since both matrices share an axis (people), the dataset needs to be considered holistically
for the best estimate of a graph representation.  Our algorithm was developed to
solve this problem.

We extend prior work, which was previously limited to the unimodal (single-tensor) case.
As a bonus, our algorithm runs faster than prior BiGraphical and TensorGraphical work on single tensors.

## Reproducibility

We include the repositories of prior work as git submodules, so that you can
download this repository and have everything needed to reproduce our paper.

We have provided a file, `environment.yml` that contains all the information needed to
acquire the same dependencies that we used, save for the Matlab dependency (which is only
necessary for our experiments that compare to prior work - our algorithm does not depend
on Matlab)

Here is an example of how to set up your environment:

```{bash}
# For our code and all experiments
conda env create --file environment.yml
conda activate GmGM-final

# You will want to compile the fortran subroutines
# for the algorithm
cd path/to/this/repo/...../GmGM/Backends
f2py -c SUM_LOG_SUM.f90 -m sum_log_sum
f2py -c PROJECT_INV_KRON_SUM.f90 -m project_inv_kron_sum

# Necessary for the experiments comparing our
# algorithm to prior work, but otherwise
# not needed
cd path/to/matlab/...../Matlab/extern/engines/python
python -m pip install .
```

Finally, if you wish to specifically compare our algorithm to EiGLasso, it needs to be compiled
to interface with Matlab.  To do this, open Matlab and `cd` into the "EiGLasso/EiGLasso_JMLR" directory
of this project.  Then, run:

```{bash}
mex -output eiglasso_joint eiglasso_joint_mex.cpp -lmwlapack
```

We have tested this code on a Mac with and without an M1 processor, as well as on Linux (Ubuntu).
The exact versions of the figures given in the paper were all generated on the Linux computer.

### Troubleshooting

If you get an error message that looks like:

```
ResolvePackageNotFound:
    package::name=#.#.#
```

then the version of the dependency that we used for our tests is not available on your computer.
If you delete the `=#.#.#` bit from the `environments.yml` file,
this issue should go away (as you will no longer be requesting this
specific version of the package, just any arbitrary version).

If the error does not go away, that means the package is not available at all on your system.
**This is rather unlikely!**  If it does, remove it from the `environments.yml` file and repeat
until it works.  Most dependencies are for individual experiments on real-world data, rather than the algorithm
itself, so doing this process will still allow you to replicate most experiments.

## Synthetic Data

We generated random graphs where each edge was a bernoulli random variable.

We can see that our algorithm performs as well as state-of-the-art on matrix data:
![Performance on matrices](Final-Plots/precision_recall_50_0.2_10.png)

And that having a shared axis really does improve performance, validating the
reasoning behind the creation of this model and algorithm
![Performance on two matrices](Final-Plots/precision_recall_shared_50_0.2_10.png)

Finally, our algorithm performs well on 3-axis data but not as well as state-of-the-art.
As our model is the same as TeraLasso's when restricted to the single-tensor case, we suspect
this is due to our use of thresholding instead of Lasso.
![Performance on a tensor](Final-Plots/precision_recall_tensor_50_0.2_10.png)

Our algorithm is more than 10x as fast as prior work on the matrix case:
(EiGLasso suddenly getting slow is not a fluke - it happened every time we ran this
experiment!)
![Speed on matrices](Final-Plots/2_axis_runtime_curves.png)

But only slightly so on 3-axis data:
![Speed on tensors](Final-Plots/3_axis_runtime_curves.png)

This is due to the computation of the Gram matrices taking up the most time - their
complexity rises exponentially with the amount of axes in a single tensor.

We can't compare to prior work on the multi-tensor case, as no prior work exists.

To reproduce these experiments, run:

```{bash}
# Warning - this will take a long time!
Final-Experiments/run-synthetic-experiments.sh
```

For PR curves, algorithms were run 50 times and averaged out.  For runtime, they were run 5 times.

## Datasets

We tested on 5 datasets.  Here's how to find them:

### COIL (Duck Video)

[Available here](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php).
We downloaded the "processed" dataset.  To reproduce this experiment, run:

```{bash}
python Final-Experiments/coil-video-reconstruction.py
```

In `coil-data-video-reconstruction.py` we checked how good GmGM was at reconstructing the axes.
We found that for rows, columns, and frames, it got accuracies of 80%, 91%, and 99%, respectively.
Accuracy was defined by the formula: $\frac{1}{2n} \sum^{n}_{i} (\delta_{left}^i + \delta_{right}^i)$, where
$\delta_{left}^i$ is 1 if the row/col/frame before $i$ is correct, and $\delta_{right}^i$ is whether the
row/col/frame after $i$ is correct (both are zero otherwise).

![Visual reconstruction](Final-Plots/coil-20-duck-still.png)

As you can see, even though the accuracy is high, the duck still looks ugly.  We can look
at the precision matrices to try to diagnose this:

![Duck precisions](Final-Plots/coil-20-precisions.png)

As the duck has a lot of rows that often look the same (black background), it's not
surprising that it does so much worse on the rows.

### E-MTAB-2805 (Mouse Embryo Stem Cells)

[Available here](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-2805).
To reproduce this experiment, run:

```{bash}
python Final-Experiments/mouse-embryo-stem-cell-precision.py
```

This dataset consisted of 3 classes, depending on stage of the cell cycle.  We
were hoping that each stage clustered together on our graph, in which case you should
see a 3x3 block-diagonal structure on the following graph:
![Precision matrix](Final-Plots/mouse-embryo-cells-precision.png)

Unfortunately, we do not.  We do see a 3x3 structure overall, implying GmGM is learning
something - but not well.

### EchoNet-Dynamic (Echocardiograms/Heartbeats)

[Available here](https://echonet.github.io/dynamic/).
You have to request access to download this dataset.
To reproduce this experiment, run:

```{bash}
python Final-Experiments/echonet-dynamic-heartbeat-detector.py
```

This dataset is a set of videos of hearts (Echocardiograms).  We would hope that
our algorithm can detect the regular structure of a heartbeat - and indeed it can!

![Precision matrix of a heartbeat (Frames)](Final-Plots/EchoNet-precision-0XF70A3F712E03D87.png)

But, of course, we've cherrypicked that precision matrix - others do not look as good:

![Precision matrix of a heartbeat (Frames)](Final-Plots/EchoNet-precision-0XF60BBEC9C303C98.png)

Even when the precision matrix is not obvious, however, we can still recover the heartbeat!
We detect it by looking at the diagonals of the precision matrix and finding peaks:

![Peaks of precision matrix of a heartbeat](Final-Plots/EchoNet-heartbeat-0XF60BBEC9C303C98.png)

We can verify this method by marking when the mitral valve opens, and seeing whether, given the first such opening,
we can use these peaks to work out when the future openings are.  We had to hand-label videos to do this test,
so we only tested five videos.  The exact predictions are available in `Final-Data/EchoNet-mitral-raw.txt`

### 10x Genomics Multiome (Flash Frozen Lymph Nodes, B Cell Lymphoma)

[Available here](https://www.10xgenomics.com/resources/datasets/fresh-frozen-lymph-node-with-b-cell-lymphoma-14-k-sorted-nuclei-1-standard-2-0-0).
We downloaded the "Filtered feature barcode matrix (HDF5)" file.  To reproduce this experiment, run:

```{bash}
python Final-Experiments/10x-genomics-experiment
```

#### Comparison with UMAP

The first test we ran was to see if the clusters we find on UMAP are also reasonable on our graph:

![UMAP clusters](Final-Plots/cell_graph_umap.png)

![Clusters on our graph](Final-Plots/cell_graph.png)

The first plot above is UMAP, clustered by kmeans.  The second plot is our graph, with the vertices
still colored by UMAP's kmeans clusters.  We can see that our graph splits the data into roughly the
same three main chunks as UMAP.

Since we checked if UMAP results were interpretable in our graph, we also checked if our graph produced
interpretable results in UMAP-space.

![Louvain clusters](Final-Plots/cell_graph_louvain.png)

![Louvain clusters on UMAP](Final-Plots/cell_graph_louvain_umap.png)

The first plot above is our graph, clustered by Louvain.  The second plot is UMAP, with vertices
colored by the cluster they are in from the first plot.  We can also see that these clusters form
sensible regions on UMAP.

#### GO Term Enrichment Analysis

![Numbered clusters](Final-Plots/umap_louvain.png)

We can look at each cluster (Louvain clustering on our graph as discussed earlier) and see if any
specific biological processes are overrepresented.  Clusters 3 & 7 stand out in this context.

![Cluster 0](Final-Plots/GO_Terms_Cluster_0.png)
![Cluster 1](Final-Plots/GO_Terms_Cluster_1.png)
![Cluster 2](Final-Plots/GO_Terms_Cluster_2.png)
![Cluster 3](Final-Plots/GO_Terms_Cluster_3.png)
![Cluster 4](Final-Plots/GO_Terms_Cluster_4.png)
![Cluster 5](Final-Plots/GO_Terms_Cluster_5.png)
![Cluster 6](Final-Plots/GO_Terms_Cluster_6.png)
![Cluster 7](Final-Plots/GO_Terms_Cluster_7.png)

### LifeLines-DEEP (Metagenomics + Multiomics)
[Available here](https://ega-archive.org/studies/EGAS00001001704#).
You have to request access to download this dataset.  We requested access to the 16S, MGS, and metabolomics
datasets, as well as the demographics data - but we did not request the follow-up data.  We ultimately used
their metabolomics data, as well as the MGS data that had already been pre-processed by the authors of the ZiLN
paper and available in their repository.

To reproduce this experiment, run:

```{bash}
Final-Experiments/lifelines.sh
```

This dataset was considered by the authors of the ZiLN paper and algorithm (added as a git submodule here).
To validate their method, they looked at the assortativity (tendency of related species to cluster together) in
their graphs.  We get similar assortativities when looking at just metagenomics and metagenomics+metabolomics:

![Metagenomics assortativities](Final-Plots/assortativities-without-metabolites.png)

![All assortativities](Final-Plots/assortativities-with-metabolites.png)

We noticed that our graphs tended to be more robust than ZiLN's.

ZiLN:

![ZiLN Robustness](Final-Plots/ziln-robustness-without-metabolites.png)

GmGM sans metabolites:

![GmGM without metabolites](Final-Plots/robustness-without-metabolites.png)

GmGM with metabolites:

![GmGM with metabolites](Final-Plots/robustness-with-metabolites.png)
