# ZiLN authors already preprocessed MGS data
load("./Zi-LN/data/ll_deep.rda")

metagenomics.counts <- counts

# You will need to manually change the filepath here
metabolomics.counts <- read.table(
    file.path(
        ".",
        "Data",
        "LL-Deep Data",
        "Metabolomics",
        "EGAF00004996145",
        "EGA_meta_1054base_311fup_metabolomics.txt"
    ),
    sep = "\t",
    header = TRUE,
    row.names = 1,
)
map.file <- read.table(
    file.path(
        ".",
        "Data",
        "LL-Deep Data",
        "Metagenomic-Shotgun",
        "EGAD00001001991",
        "delimited_maps",
        "Run_Sample_meta_info.map"
    ),
    sep = "\t",
    header = FALSE
)
names(map.file)[names(map.file) == "V1"] <- "Metabolomics"
names(map.file)[names(map.file) == "V5"] <- "Shotgun"
names(map.file)[names(map.file) == "V8"] <- "Demographics"

#=======#
# Clean the datasets
#=======#

# Add "id_" to the beginning of each sample name
# in the Metabolomics fields of map.file
map.file$Metabolomics <- paste0("id_", map.file$Metabolomics)

# Remove "LL" from the end of each row name
# in the metabolomics dataset, if it is present
rownames(metabolomics.counts) <- gsub(
    "LL$",
    "",
    rownames(metabolomics.counts)
)

# And do the same for the Metabolomics column of map.file
map.file$Metabolomics <- gsub(
    "LL$",
    "",
    map.file$Metabolomics
)

# Remove all rows whose name begins with id_APK from
# the metabolomics dataset
metabolomics.counts <- metabolomics.counts[
    !grepl("^id_APK", rownames(metabolomics.counts)),
    ,
    drop = FALSE
]

# In the demographics column of map.file,
# remove the beginning of every row
# Specifically, remove everything after the first ;
# but keep everything before it
map.file$Demographics <- gsub(
    ";.*",
    "",
    map.file$Demographics
)

# In the demographics column of map.file,
# remove everything before and including the first =
# but keep everything after it
map.file$Demographics <- gsub(
    ".*=",
    "",
    map.file$Demographics
)

# And now rename the Demographics column to Gender
map.file$Gender <- map.file$Demographics
map.file$Demographics <- NULL

# Finally, make Gender a factor
map.file$Gender <- factor(map.file$Gender)

# Get rid of all unnamed columns
map.file <- map.file[, !grepl("^V", names(map.file))]

# Remove the people missing from the metabolomics dataset
# in map.file
for (i in map.file$Metabolomics) {
    if (!i %in% rownames(metabolomics.counts)) {
        map.file <- map.file[-which(map.file$Metabolomics == i), ]
    }
}

# And likewise in the metagenomics file
metagenomics.counts <- metagenomics.counts[map.file$Shotgun,]

# And finally in the metabolomics file
# This shouldn't remove anything, but will force the ordering
# of this file to match the ordering of the other two
metabolomics.counts <- metabolomics.counts[map.file$Metabolomics,]

# So we can now save the data as csv files
write.csv(
    map.file,
    file.path(
        ".",
        "Data",
        "LL-Deep Data - Processed",
        "Map.csv"
    )
)
write.csv(
    metagenomics.counts,
    file.path(
        ".",
        "Data",
        "LL-Deep Data - Processed",
        "MetagenomicsShotgun.csv"
    )
)
write.csv(
    metabolomics.counts,
    file.path(
        ".",
        "Data",
        "LL-Deep Data - Processed",
        "Metabolomics.csv"
    )
)