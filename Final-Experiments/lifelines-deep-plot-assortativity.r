# Load the data
load("./Zi-LN/data/ll_deep.rda")
write.csv(
    taxmat,
    "./Data/LL-Deep Data - Processed/unfiltered-taxmat-ziln.csv"
)

# Rename the taxmat columns to something more informative
colnames(taxmat) <- c(
    "Domain",
    "Phylum",
    "Class",
    "Order",
    "Family",
    "Genus"
)

# Load the libraries
source("Zi-LN/inference.R")
source("Zi-LN/utils/utils.R")
library("huge")
library("igraph")
library("ggplot2")

# Get a boolean 1135x3957 matrix of whether the species
# was found in the person or not
nonzeros <- counts > 0

# Get the number of distinct people that possessed each species
num.nonzeros <- apply(nonzeros, 2, sum)

# Get the total amount of people
total.cells <- dim(counts)[1]

# Only keep the species who appear in more than 20% of the people
keep.indices <- (num.nonzeros / total.cells) > 0.2
counts_el <- as.matrix(counts[, keep.indices])
write.csv(
    counts_el,
    "./Data/LL-Deep Data - Processed/filtered-raw-counts-ziln.csv"
)
write.csv(
    log(counts_el+1),
    "./Data/LL-Deep Data - Processed/filtered-log-counts-ziln.csv"
)

taxmat_el <- taxmat[keep.indices,]

write.csv(
    taxmat_el,
    "./Data/LL-Deep Data - Processed/filtered-taxmat-ziln.csv"
)

# Get the zs
options(warn = -1) # turn warnings off because otherwise it's gonna scream...
zs <- infer_Z(counts_el)
write.csv(zs, "./Data/LL-Deep Data - Processed/filtered-zs-ziln.csv")

source("./Backends/utilities.r")

# Get the matrix for ZiLN methodology
ziln.lambdas <- 10^seq(-0.1, -1.1, by=-0.05)
ziln.path <- huge(zs, lambda=ziln.lambdas)$path

num.elements(ziln.path[[8]])

# Save ziln.path[[8]] to a csv for analysis in python
write.csv(
    as.matrix(ziln.path[[8]]),
    "./Data/LL-Deep Data - Processed/ziln-microbe-graph.csv"
)

kept_species <- colnames(counts[keep.indices])

# Save kept_species as a csv
write.csv(
    kept_species,
    "./Data/LL-Deep Data - Processed/kept-species-ziln.csv"
)

GmGM.mat <- as.matrix(
    read.csv(
        "./Data/LL-Deep Data - Processed/GmGM-Microbes-Subset.csv",
        header=FALSE,
        col.names=paste0("C", 1:565),
        sep=","
    )
)

# These bounds were chosen by eye.
upper.bound <- 0.25#0.6
lower.bound <- 0.1#0.07
sum(abs(GmGM.mat) > upper.bound) / 2 - 282.5
sum(abs(GmGM.mat) > lower.bound) / 2 - 282.5
GmGM.lambdas <- exp(0:19 * (log(upper.bound) - log(lower.bound)) / 19 + log(lower.bound))

GmGM.path <- lapply(
    GmGM.lambdas,
    function(thresh) threshold.matrix(GmGM.mat, thresh)
)

plot.compared.assortativities(
    list(GmGM.path, ziln.path),
    taxmat_el,
    list(shQuote("GmGM"), shQuote("ziln")),
    "Assortativities without Metabolites"
)
ggsave("Final-Plots/assortativities-without-metabolites.svg")

# Repeat the same, with metabolites

GmGM.mat.full <- as.matrix(read.csv(
    "./Data/LL-Deep Data - Processed/GmGM-Microbes-Subset-Metabolites.csv",
    header=FALSE,
    col.names=paste0("C", 1:565),
    sep=","
))

# These bounds were chosen by eye.
upper.bound <- 0.2#0.2
lower.bound <- 0.1#0.1
sum(abs(GmGM.mat.full) > upper.bound) / 2 - 282.5
sum(abs(GmGM.mat.full) > lower.bound) / 2 - 282.5
GmGM.lambdas.full <- exp(
    0:19
    * (log(upper.bound) - log(lower.bound))
    / 19
    + log(lower.bound)
)

GmGM.path.full <- lapply(
    GmGM.lambdas.full,
    function(thresh) threshold.matrix(GmGM.mat.full, thresh)
)

plot.compared.assortativities(
    list(GmGM.path.full, ziln.path),
    taxmat_el,
    list(shQuote("GmGM"), shQuote("ziln")),
    "Assortativities with Metabolites"
)
ggsave("Final-Plots/assortativities-with-metabolites.svg")

# Measure robustness
source('robin/R/ROBIN.R')
     
GmGM.graph <- graph.adjacency(
    GmGM.path.full[[10]],
    mode = "Undirected"
)

print("If these are not roughly the same size, change the indices or thresholds!")
sum(GmGM.path.full[[10]] != 0) / 2
sum(GmGM.path[[13]] != 0) / 2
sum(ziln.path[[8]] != 0) / 2
V(GmGM.graph)$idx <- 1:565
GmGM.graph.zoom <- delete.vertices(
    GmGM.graph,
    which(degree(GmGM.graph) == 0)
)

GmGM.graph.random <- random(graph=GmGM.graph.zoom)
proc <- robinRobust(
    graph=GmGM.graph.zoom,
    graphRandom=GmGM.graph.random,
    measure="nmi", 
    method="louvain",
    type="independent"
)
plotRobin(
    graph=GmGM.graph.zoom,
    model1=proc$Mean,
    model2=proc$MeanRandom,
    legend=c("real data", "null model")
)
ggsave("Final-Plots/robustness-with-metabolites.svg")

GmGM.graph <- graph.adjacency(
    GmGM.path[[10]],
    mode = "Undirected"
)
V(GmGM.graph)$idx <- 1:565
GmGM.graph.zoom <- delete.vertices(
    GmGM.graph,
    which(degree(GmGM.graph) == 0)
)

GmGM.graph.random <- random(graph=GmGM.graph.zoom)
proc <- robinRobust(
    graph=GmGM.graph.zoom,
    graphRandom=GmGM.graph.random,
    measure="nmi", 
    method="louvain",
    type="independent"
)
plotRobin(
    graph=GmGM.graph.zoom,
    model1=proc$Mean,
    model2=proc$MeanRandom,
    legend=c("real data", "null model")
)
ggsave("Final-Plots/robustness-without-metabolites.svg")


ziln.graph <- graph.adjacency(
    ziln.path[[10]],
    mode = "Undirected"
)
V(ziln.graph)$idx <- 1:565
ziln.graph.zoom <- delete.vertices(
    ziln.graph,
    which(degree(ziln.graph) == 0)
)
ziln.graph.random <- random(graph=ziln.graph.zoom)
proc <- robinRobust(
    graph=ziln.graph.zoom,
    graphRandom=ziln.graph.random,
    measure="nmi", 
    method="louvain",
    type="independent"
)
plotRobin(
    graph=ziln.graph.zoom,
    model1=proc$Mean,
    model2=proc$MeanRandom,
    legend=c("real data", "null model")
)
ggsave("Final-Plots/ziln-robustness-without-metabolites.svg")