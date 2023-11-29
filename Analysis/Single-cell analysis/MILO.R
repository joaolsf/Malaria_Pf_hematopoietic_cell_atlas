
# Reference: https://rawcdn.githack.com/MarioniLab/miloR/7c7f906b94a73e62e36e095ddb3e3567b414144e/vignettes/milo_gastrulation.html#2_Visualize_the_data
# https://github.com/MarioniLab/miloR


# 1- Converting anndata to SingleCellExperiment object (SCE)
library(anndata)
library(dplyr)
library(patchwork)
ad <- anndata::read_h5ad('adata_subset4.h5ad')

library(rhdf5)
library(zellkonverter)
ad <- readH5AD('IMC_Pf_spleen_redsea_corrected_complete.h5ad') #it worked and converted to SingleCell Experiment
#split the adata object by disease group to run the analysis within each group and compare later?

# 2- Save SCE object
saveRDS(ad, "adata_subset4.rds")

ad <- readRDS("IMC_Pf_spleen_redsea_corrected_complete_sce.rds")
ad

library(SingleCellExperiment)
ad@assays@data@listData[["counts"]] <- NULL 
names(assays(ad)) <- "counts"

library(scater)
ad <- ad[, colSums(counts(ad)) > 0]
ad <- logNormCounts(ad)
ad

set.seed(1234)
ad <- runPCA(ad)
saveRDS(ad, "IMC_Pf_spleen_redsea_corrected_complete_sce.rds")

#library(scRNAseq)


# 3- Milo set-up and make milo object
library(miloR)
library(igraph)

milo <- Milo(ad)
milo
saveRDS(milo, "milo.rds")
milo <- readRDS("./milo.rds")

# 4- Add KNN graph if there is no KNN graph in the SCE object 

# Milo looks for neighbourhoods in a KNN graph to perform DA analysis. 
# This need to be stored in the graph slot of the Milo object. Here we have two options:

# A- We can add the KNN graph precomputed with sc.pp.neighbors, using the function buildFromAdjacency. 
# IN SCANPY (PYTHON) RUN THE FOLLOWING CODE:
# Save the binary connectivity matrix
# knn_adjacency = adata.obsp["connectivities"]
# Load in R and run: milo_graph <- buildFromAdjacency(knn_adjacency, k=20, is.binary=TRUE)
# graph(milo) <- miloR::graph(milo_graph)

# B- we can recompute the KNN graph using the dedicated function in miloR. USE THIS!
milo <- buildGraph(milo, k=100, d = 37, transposed = FALSE) #k = 100 because is the same k used in sc.pp.neighbors

# 5- Defining representative neighbourhoods on the KNN graph

# We define the neighbourhood of a cell, the index, as the group of cells connected by an edge in the KNN graph to the index cell. For efficiency, we don’t test for DA in the neighbourhood of every cell, but we sample as indices a subset of representative cells, using a KNN sampling algorithm used by Gut et al. 2015.
# As well as d and k, for sampling we need to define a few additional parameters:
# prop: the proportion of cells to randomly sample to start with. We suggest using prop=0.1 for datasets of less than 30k cells. 
# For bigger datasets using prop=0.05 should be sufficient (and makes computation faster).
# refined: indicates whether you want to use the sampling refinement algorith, or just pick cells at random. 
# The default and recommended way to go is to use refinement. 
# The only situation in which you might consider using random instead, is if you have batch corrected your data with a graph based correction algorithm, such as BBKNN, but the results of DA testing will be suboptimal.
milo <- makeNhoods(milo, prop = 0.01, k = 100, d=37, refined = FALSE, reduced_dims = "PCA")

# Once we have defined neighbourhoods, we plot the distribution of neighbourhood sizes (i.e. how many cells form each neighbourhood) to evaluate whether the value of k used for graph building was appropriate. 
# We can check this out using the plotNhoodSizeHist function.
# As a rule of thumb we want to have an average neighbourhood size over 5 x N_samples. 
# If the mean is lower, or if the distribution is
plotNhoodSizeHist(milo)

# 6- Counting cells in neighbourhoods

# Milo leverages the variation in cell numbers between replicates for the same experimental condition to test for differential abundance. 
# Therefore we have to count how many cells from each sample are in each neighbourhood. 
# We need to use the cell metadata and specify which column contains the sample information.
colData(milo) # to check the metadata

# This adds to the Milo object a n×m matrix, where n is the number of neighbourhoods and m is the number of experimental samples. 
# Values indicate the number of cells from each sample counted in a neighbourhood. 
# This count matrix will be used for DA testing.

# Count cells per ROI
milo <- countCells(milo, meta.data = as.data.frame(colData(milo)), sample="ROI")
# Count cells per Patient
# milo <- countCells(milo, meta.data = as.data.frame(colData(milo)), sample="Case")
# Count cells per Disease group
# milo <- countCells(milo, meta.data = as.data.frame(colData(milo)), sample="Type")

head(nhoodCounts(milo))

# 7- Defining experimental design

# Now we are all set to test for differential abundance in neighbourhoods. 
# We implement this hypothesis testing in a generalized linear model (GLM) framework, specifically using the Negative Binomial GLM implementation in edgeR.

# We first need to think about our experimental design. 
# The design matrix should match each sample to the experimental condition of interest for DA testing. 
# In this case, we want to detect DA between disease progression, stored in the stage column of the dataset colData. 
# We also include the patient ID (Case) column in the design matrix. This represents a known technical covariate that we want to account for in DA testing.

covid_design <- data.frame(colData(milo))[,c("ROI","Patient", "Group")]

## Convert batch info from integer to factor
covid_design$Patient <- as.factor(covid_design$Patient) 
covid_design <- distinct(covid_design)
rownames(covid_design) <- covid_design$ROI
covid_design

# 8- Computing neighbourhood connectivity

# Milo uses an adaptation of the Spatial FDR correction introduced by cydar, where we correct p-values accounting for the amount of overlap between neighbourhoods. 
# Specifically, each hypothesis test P-value is weighted by the reciprocal of the kth nearest neighbour distance. 
# To use this statistic we first need to store the distances between nearest neighbors in the Milo object. 
# This is done by the calcNhoodDistance function (N.B. this step is the most time consuming of the analysis workflow and might take a couple of minutes for large datasets).
milo <- calcNhoodDistance(milo, d=37, reduced.dim = "PCA") #it takes a long time to run!
saveRDS(milo, "milo.rds")

# 9- DA Testing
# Now we can do the DA test, explicitly defining our experimental design. 
# In this case, we want to test for differences between experimental stages, while accounting for the variability between technical batches.
# This calculates a Fold-change and corrected P-value for each neighbourhood, which indicates wheather there is significant differential abundance between groups. 
# The main statistics we consider here are:
# logFC: indicates the log-Fold change in cell numbers between ED and LD;
# PValue: reports P-values before FDR correction
# SpatialFDR: reports P-values corrected for multiple testing accounting for overlap between neighbourhoods

## Reorder rownames to match columns of nhoodCounts(milo)
table(covid_design$Group)
covid_design <- covid_design[colnames(nhoodCounts(milo)), , drop=FALSE]
write.csv(covid_design, "covid_design.csv")
covid_design <- read.csv("covid_design.csv")
rownames(covid_design) <- covid_design$X
covid_design <- covid_design[,-1]

contrast.1 <- c("GroupCM - GroupNon_CM") # the syntax is <VariableName><ConditionLevel> - <VariableName><ControlLevel>

da_results1 <- testNhoods(milo, design = ~ 0 + Group, design.df = covid_design, model.contrasts = contrast.1)
head(da_results1)
da_results1 %>%
  arrange(SpatialFDR) %>%
  head() 

saveRDS(da_results1, "da_results1.rds")
da_results1 <- readRDS("./DA_results1/da_results1.rds")

da_results2 <- testNhoods(milo, design = ~ 0 + Type, design.df = covid_design, model.contrasts = contrast.1, fdr.weighting="graph-overlap")
head(da_results2)
da_results2 %>%
  arrange(SpatialFDR) %>%
  head() 
saveRDS(da_results2, "da_results2.rds")

da_results3 <- testNhoods(milo, design = ~ Case + Type, design.df = covid_design)
head(da_results3)
da_results3 %>%
  arrange(SpatialFDR) %>%
  head() 


# 9.1- Inspecting DA testing results

# We can start inspecting the results of our DA analysis from a couple of standard diagnostic plots. 
# We first inspect the distribution of uncorrected P values, to verify that the test was balanced.
ggplot(da_results1, aes(PValue)) + geom_histogram(bins=50)

# Then we visualize the test results with a volcano plot (remember that each point here represents a neighbourhood, not a cell).
ggplot(da_results1, aes(logFC, -log10(SpatialFDR))) + 
  geom_point() +
  geom_hline(yintercept = 1) ## Mark significance threshold (10% FDR)

# To visualize DA results relating them to the embedding of single cells, we can build an abstracted graph of neighbourhoods that we can superimpose on the single-cell embedding.
# Here each node represents a neighbourhood, while edges indicate how many cells two neighbourhoods have in common. 
# Here the layout of nodes is determined by the position of the index cell in the UMAP embedding of all single-cells. 
# The neighbourhoods displaying singificant DA are colored by their log-Fold Change.

## Plot single-cell UMAP
umap_pl <- plotReducedDim(milo, dimred = "X_umap", colour_by="Group", text_by = "pheno_cluster", 
                          text_size = 3, point_size=0.5) +
  guides(fill="none")

## Plot neighbourhood graph
milo <- buildNhoodGraph(milo)

nh_graph_pl <- plotNhoodGraphDA(milo, da_results1, layout="X_umap",alpha=0.1) 

umap_pl + nh_graph_pl +
  plot_layout(guides="collect")

# We might also be interested in visualizing wheather DA is particularly evident in certain cell types. 
# To do this, we assign a cell type label to each neighbourhood by finding the most abundant cell type within cells in each neighbourhood. 
# We can label neighbourhoods in the results data.frame using the function annotateNhoods. 
# This also saves the fraction of cells harbouring the label.
da_results1 <- annotateNhoods(milo, da_results1, coldata_col = "pheno_cluster")
head(da_results1)

#reorder clusters
da_results1$pheno_cluster_ordered <- factor(da_results1$pheno_cluster_edited,levels=c("CD3+ cell", "Endothelial cell", "Epithelial cell", "Alveolar Macrophage",
                                                                                      "Apoptotic Neutrophil", "Apoptotic Smooth Muscle Cell", "Apoptotic Fibroblast", "Apoptotic Epithelial cell",
                                                                                      "Smooth Muscle Cell", "CD66bLow Neutrophil", "CD66bHigh Neutrophil", "ColHigh Fibroblast", "ColLow Fibroblast", "CD8 T cell",
                                                                                      "CD4 T cell", "Interstitial Macrophage", "Classical Monocyte", "Activated Endothelial cell",
                                                                                      "SARSCoV2+ Epithelial cell", "SARSCoV2+ Alveolar Macrophage"))
library(forcats)
da_results1$pheno_cluster_ordered <- fct_rev(da_results1$pheno_cluster_ordered)


plotDAbeeswarm(da_results1, group.by = "pheno_cluster", alpha = 0.1)
#plot only neighbourhoods with spatialFDR <0.1
plotDAbeeswarm(da_results1, group.by = "pheno_cluster", alpha = 0.1, subset.nhoods = da_results1$SpatialFDR < 0.05)

# While neighbourhoods tend to be homogeneous, we can define a threshold for celltype_fraction to exclude neighbourhoods that are a mix of cell types.
ggplot(da_results1, aes(pheno_cluster_edited_fraction)) + geom_histogram(bins=50)

da_results1$pheno_cluster_edited <- ifelse(da_results1$pheno_cluster_edited_fraction < 0.7, "Mixed", da_results1$pheno_cluster_edited)

# Now we can visualize the distribution of DA Fold Changes in different cell types after filtering
plotDAbeeswarm(da_results1, group.by = "pheno_cluster_edited")

# 10- Finding markers of DA populations
# Once you have found your neighbourhoods showindg significant DA between conditions, 
# you might want to find gene signatures specific to the cells in those neighbourhoods.
# The function findNhoodGroupMarkers runs a one-VS-all differential gene expression test to identify marker genes for a group of neighbourhoods of interest. 
# Before running this function you will need to define your neighbourhood groups depending on your biological question, that need to be stored as a NhoodGroup column in the da_results data.frame.

da_results1$NhoodGroup <- as.numeric(da_results1$SpatialFDR < 0.1 & da_results1$logFC < 0)
da_nhood_markers1 <- findNhoodGroupMarkers(milo, da_results1, subset.row = rownames(milo)[1:38])
head(da_nhood_markers1)

plotNhoodGroups(milo, da_results1, layout="X_umap") 


ggplot(da_nhood_markers1, aes(logFC_0,-log10(adj.P.Val_0 ))) + 
  geom_point() +
  geom_hline(yintercept = 2)

rownames(da_nhood_markers1) <- da_nhood_markers1$GeneID
markers <- rownames(da_nhood_markers1)[da_nhood_markers1$adj.P.Val_0 < 0.01 & da_nhood_markers1$logFC_0 > 0]
plotNhoodExpressionGroups(milo, da_results1, features=markers,
                          subset.nhoods = da_results1$NhoodGroup %in% c('0','1'), scale=FALSE,
                          grid.space = "fixed")






