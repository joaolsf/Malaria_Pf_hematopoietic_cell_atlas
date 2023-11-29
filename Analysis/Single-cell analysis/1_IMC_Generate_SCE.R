
# 1- Converting anndata to SingleCellExperiment object (SCE)
library(rhdf5)
library(zellkonverter)

ad <- readH5AD('./2_h5ad_files/IMC_Pf_spleen_redsea_corrected_complete.h5ad') 
ad_CM <- readH5AD('./2_h5ad_files/adata_CM.h5ad') 
ad_nonCM <- readH5AD('./2_h5ad_files/adata_NonCM.h5ad')

#Trying to convert SCE to SpatialExperiment object (SPE)
library(spatialLIBD)
spe <- sce_to_spe(ad, imageData = NULL) # it did not work because it needs the imgData dataframe
#Trying to convert the SCE to SPE as suggested in https://github.com/theislab/zellkonverter/issues/61
library(SpatialExperiment) #explore more this function
coords <- as.matrix(reducedDim(ad, "spatial"))
colnames(coords) = c("x","y")
spe2 <- SpatialExperiment(
  assay = assay(ad,"X"), 
  colData = ad@colData, 
  spatialCoords = coords,
  
)

#Save object
saveRDS(ad, "./IMC_Pf_spleen_redsea_corrected_complete_sce.rds")
saveRDS(ad_CM, "./adata_CM_sce.rds")
saveRDS(ad_nonCM, "./adata_NonCM_sce.rds")

ad_CM <- readRDS("./adata_CM_sce.rds")
ad_nonCM <- readRDS("./adata_NonCM_sce.rds")


######################################################################################################################################

#3- Analysis using imcRtools functions
#The Bodernmiller's pipeline works with both, SPE and SCE, just need to figure out the correct slots of SCE that should be used in the functions.
#Have a look at the help of each function to check which slot of the SCE should be used.

#3.1- Testing some plotting from Bodernmiller's tutorial using the SingleCellExperiment
library(imcRtools)
library(cytomapper)
library(openxlsx)
library(stringr)
library(dittoSeq)
library(RColorBrewer)
library(Rphenograph)
library(igraph)
library(dittoSeq)
library(viridis)
library(bluster)
library(BiocParallel)
library(ggplot2)
library(scran)
library(CATALYST)
library(kohonen)
library(ConsensusClusterPlus)
library(patchwork)
library(pheatmap)
library(gridExtra)
library(SingleCellExperiment)
library(SpatialExperiment)
library(tidyverse)
library(ggridges)
library(scater)
library(cowplot)
library(viridis)

dittoRidgePlot(ad, var = "CD3", group.by = "Group", assay = "X") +
  ggtitle("CD3")


## UMAP colored by cell type and expression - dittoDimPlot
p1 <- dittoDimPlot(ad, var = "pheno_cluster", 
                   reduction.use = "X_umap", size = 0.2,
                   do.label = TRUE, labels.size = 4, labels.highlight = TRUE) +
  scale_color_manual(values = ad@metadata[["pheno_cluster_colors"]]) +
  theme(legend.title = element_blank()) +
  ggtitle("Cell types on UMAP, integrated cells")

p2 <- dittoDimPlot(ad, var = "IgD", assay = "X",
                   reduction.use = "X_umap", size = 0.2, 
                   colors = viridis(100), do.label = TRUE) +
  scale_color_viridis()

p1 + p2

#3.2- Testing spatial analysis using the SingleCellExperiment object
#3.2.1- Build the Neighborhood graph

ad <- buildSpatialGraph(ad, img_id = "ROI", type = "expansion", threshold = 20, coords = c("X_loc", "Y_loc"))
colPair(ad, "expansion_interaction_graph")

ad_CM <- buildSpatialGraph(ad_CM, img_id = "ROI", type = "expansion", threshold = 20, coords = c("X_loc", "Y_loc"))
colPair(ad_CM, "expansion_interaction_graph")

ad_nonCM <- buildSpatialGraph(ad_nonCM, img_id = "ROI", type = "expansion", threshold = 20, coords = c("X_loc", "Y_loc"))
colPair(ad_nonCM, "expansion_interaction_graph")

# expansion interaction graph 
plotSpatial(ad[,ad$ROI == "PM102-ROI1"], 
            node_color_by = "pheno_cluster", 
            img_id = "ROI", 
            coords = c("X_loc", "Y_loc"),
            draw_edges = TRUE, 
            colPairName = "expansion_interaction_graph", 
            node_size_fix = 1,
            nodes_first = FALSE, 
            directed = FALSE,
            edge_color_fix = "grey") + 
  scale_color_manual(values = ad@metadata[["pheno_cluster_colors"]]) +
  ggtitle("expansion interaction graph")

#4- Spatial Community Analysis

#The detection of spatial communities was proposed by (Jackson et al. 2020). 
#Here, cells are clustered solely based on their interactions as defined by the spatial object graph. 
#In the following example, we perform spatial community detection separately for tumor and stromal cells.

#The general procedure is as follows:
#1. create a colData(spe) entry that specifies if a cell is part of the tumor or stroma compartment. 
#2. use the detectCommunity function of the imcRtools package to cluster cells within the tumor or stroma compartment solely based on their spatial interaction graph as constructed by the steinbock package.
#Both tumor and stromal spatial communities are stored in the colData of the SpatialExperiment object under the spatial_community identifier.
#We set the seed argument within the SerialParam function for reproducibility purposes. 
#This is important as the global seed is not recognized by functions provided by the BiocParallel package.

ad$Follicle_Redpulp <- ifelse(ad$cell_type == "B cells", "CD4 T cells", "Macrophages")

library(BiocParallel)
ad <- detectCommunity(ad, 
                       colPairName = "expansion_interaction_graph", 
                       size_threshold = 10,
                       group_by = "Follicle_Redpulp",
                       BPPARAM = SerialParam(RNGseed = 220819))
#We can now separately visualize the tumor and stromal communities.
#Spatial Tumor communities
plotSpatial(ad[,ad$cell_type == "CD4 T cells"], 
            node_color_by = "spatial_community", 
            img_id = "ROI", 
            coords = c("X_loc", "Y_loc"),
            node_size_fix = 0.2) +
  theme(legend.position = "none") +
  ggtitle("Spatial CD4 T cell communities") +
  scale_color_manual(values = rev(colors()))

#Spatial Stromal communities
plotSpatial(ad[,ad$cell_type != "B cells"], 
            node_color_by = "spatial_community", 
            img_id = "ROI", 
            coords = c("X_loc", "Y_loc"),
            node_size_fix = 0.2) +
  theme(legend.position = "none") +
  ggtitle("Spatial macrophage communities") +
  scale_color_manual(values = rev(colors()))

#In the next step, the fraction of cell types within each spatial community is displayed.
library(pheatmap)
library(viridis)

for_plot <- prop.table(table(ad[,ad$cell_type != "B cells"]$spatial_community, ad[,ad$cell_type != "B cells"]$cell_type), margin = 1)
pheatmap(for_plot, color = viridis(100), show_rownames = FALSE)

#5- Cellular neighborhood analysis

#The following section highlights the use of the imcRtools package to detect cellular neighborhoods. 
#This approach has been proposed by (Goltsev et al. 2018) and (Schürch et al. 2020) to group cells based on information contained in their direct neighborhood.
#(Goltsev et al. 2018) perfomed Delaunay triangulation-based graph construction, neighborhood aggregation and then clustered cells.
#(Schürch et al. 2020) on the other hand constructed a 10-nearest neighbor graph before aggregating information across neighboring cells.
#In the following code chunk we will use the 20-nearest neighbor graph as constructed above to define the direct cellular neighborhood. 
#The aggregateNeighbors function allows neighborhood aggregation in 2 different ways:

#A- For each cell the function computes the fraction of cells of a certain type (e.g., cell type) among its neighbors.
#B- For each cell it aggregates (e.g., mean) the expression counts across all neighboring cells.

#Based on these measures, cells can now be clustered into cellular neighborhoods. 
#We will first compute the fraction of the different cell types among the 20-nearest neighbors and use kmeans clustering to group cells into 6 cellular neighborhoods.

# By celltypes
ad <- aggregateNeighbors(ad, colPairName = "expansion_interaction_graph", 
                          aggregate_by = "metadata", count_by = "pheno_cluster")

set.seed(220705)

cn_1 <- kmeans(ad$aggregatedNeighbors, centers = 6)
cn_2 <- kmeans(ad$aggregatedNeighbors, centers = 4)
ad$cn_celltypes <- as.factor(cn_1$cluster)
ad$cn_celltypes2 <- as.factor(cn_2$cluster)

plotSpatial(ad, 
            node_color_by = "cn_celltypes2", 
            img_id = "ROI", 
            coords = c("X_loc", "Y_loc"),
            node_size_fix = 0.1) +
  scale_color_brewer(palette = "Set3")

#The next code chunk visualizes the cell type compositions of the detected cellular neighborhoods (CN). k=6
library(tidyverse)
for_plot <- colData(ad) %>% as_tibble() %>%
  group_by(cn_celltypes, pheno_cluster) %>%
  summarize(count = n()) %>%
  mutate(freq = count / sum(count)) %>%
  pivot_wider(id_cols = cn_celltypes, names_from = pheno_cluster, 
              values_from = freq, values_fill = 0) %>%
  ungroup() %>%
  select(-cn_celltypes)

pheatmap(for_plot, color = colorRampPalette(c("dark blue", "white", "dark red"))(100), 
         scale = "column")

#The next code chunk visualizes the cell type compositions of the detected cellular neighborhoods (CN). k=4
library(tidyverse)
for_plot <- colData(ad) %>% as_tibble() %>%
  group_by(cn_celltypes2, pheno_cluster) %>%
  summarize(count = n()) %>%
  mutate(freq = count / sum(count)) %>%
  pivot_wider(id_cols = cn_celltypes2, names_from = pheno_cluster, 
              values_from = freq, values_fill = 0) %>%
  ungroup() %>%
  select(-cn_celltypes2)

pheatmap(for_plot, color = colorRampPalette(c("dark blue", "white", "dark red"))(100), 
         scale = "column")

#6- lisaClust
#An alternative to the aggregateNeighbors function is provided by the lisaClust Bioconductor package (Patrick et al. 2021). 
#In contrast to imcRtools, the lisaClust package computes local indicators of spatial associations (LISA) functions and clusters cells based on those. 
#More precise, the package summarizes L-functions from a Poisson point process model to derive numeric vectors 
#for each cell which can then again be clustered using kmeans.

#The lisa function requires a SegmentedCells object which can be generated using the spicyR package.
library(lisaClust)
library(spicyR)

cells <- data.frame(row.names = colnames(ad))
cells$ObjectNumber <- ad$Master_Index
cells$ImageNumber <- ad$ROI
cells$AreaShape_Center_X <- ad@colData@listData[["X_loc"]]
cells$AreaShape_Center_Y <- ad@colData@listData[["Y_loc"]]
cells$cellType <- ad$pheno_cluster

lisa_sc <- SegmentedCells(cells, cellProfiler = TRUE)
lisa_sc

#After creating the SegmentedCells object, the lisa function computes LISA curves across a given set of distances. 
#In the following example, we calculate the LISA curves within a 10µm, 20µm and 50µm neighborhood around each cell. 
#Increasing these radii will lead to broader and smoother spatial clusters.
#However, a number of parameter settings should be tested to estimate the robustness of the results.
lisaCurves <- lisa(lisa_sc, Rs = c(10, 20, 50))

# Set NA to 0
lisaCurves[is.na(lisaCurves)] <- 0

lisa_clusters <- kmeans(lisaCurves, centers = 6)$cluster

ad$lisa_clusters <- as.factor(lisa_clusters)

plotSpatial(ad, 
            node_color_by = "lisa_clusters", 
            img_id = "ROI", coords = c("X_loc", "Y_loc"),
            node_size_fix = 0.2) +
  scale_color_brewer(palette = "Set1")

#We can now observe the cell type composition per spatial cluster.
library(tidyverse)
for_plot <- colData(ad) %>% as_tibble() %>%
  group_by(lisa_clusters, pheno_cluster) %>%
  summarize(count = n()) %>%
  mutate(freq = count / sum(count)) %>%
  pivot_wider(id_cols = lisa_clusters, names_from = pheno_cluster, 
              values_from = freq, values_fill = 0) %>%
  ungroup() %>%
  select(-lisa_clusters)

pheatmap(for_plot, color = colorRampPalette(c("dark blue", "white", "dark red"))(100), 
         scale = "column")

#7- Spatial Context Analysis

#Downstream of CN assignments, we will analyze the spatial context (SC) of each cell using three functions from imcRtools.
#While CNs can represent sites of unique local processes, the term SC was coined by Bhate and colleagues (Bhate et al. 2022) 
#and describes tissue regions in which distinct CNs may be interacting. 
#Hence, SCs may be interesting regions of specialized biological events.

#Here, we will first detect SCs using the detectSpatialContext function. 
#This function relies on CN fractions for each cell in a spatial interaction graph (originally a KNN graph),
#which we will calculate using buildSpatialGraph and aggregateNeighbors. 
#We will focus on the CNs derived from cell type fractions but other CN assignments are possible.

#Of note, the window size (k for KNN) for buildSpatialGraph should reflect a length scale 
#on which biological signals can be exchanged and depends, among others, on cell density and tissue area. 
#In view of their divergent functionality, we recommend to use a larger window size for SC (interaction between local processes) than for CN (local processes) detection. 
#Since we used a 20-nearest neighbor graph for CN assignment, we will use a 40-nearest neighbor graph for SC detection. 
#As before, different parameters should be tested.
#Subsequently, the CN fractions are sorted from high-to-low and the SC of each cell is assigned as the minimal combination of SCs that additively surpass a user-defined threshold. 
#The default threshold of 0.9 aims to represent the dominant CNs, hence the most prevalent signals, in a given window.
#For more details and biological validation, please refer to (Bhate et al. 2022).

library(circlize)
library(RColorBrewer)

# Generate k-nearest neighbor graph for SC detection (k=40) 
ad <- buildSpatialGraph(ad, img_id = "ROI", 
                         type = "expansion", 
                         name = "expansion_spatialcontext_graph", 
                         threshold = 40, coords = c("X_loc", "Y_loc"))


# Aggregate based on clustered_neighbors
ad <- aggregateNeighbors(ad, 
                          colPairName = "expansion_spatialcontext_graph",
                          aggregate_by = "metadata",
                          count_by = "cn_celltypes",
                          name = "aggregatedNeighborhood")

# Detect spatial contexts
ad <- detectSpatialContext(ad, 
                            entry = "aggregatedNeighborhood",
                            threshold = 0.90,
                            name = "spatial_context")

# Define SC color scheme
col_SC <- setNames(colorRampPalette(brewer.pal(11, "Spectral"))(length(unique(ad$spatial_context))), 
                   sort(unique(ad$spatial_context)))
saveRDS(ad, "adata_subset3.rds")

# Visualize spatial contexts on images
plotSpatial(ad, 
            node_color_by = "spatial_context", 
            img_id = "ROI", 
            node_size_fix = 0.05, coords = c("X_loc", "Y_loc"),
            colPairName = "expansion_spatialcontext_graph") +
  scale_color_manual(values = col_SC)

#For ease of interpretation, we will directly compare the CN and SC assignments for a specific ROI.
library(patchwork)

# Compare CN and SC for one patient 
#As expected, we can observe that interfaces between different CNs make up distinct SCs. 
#For instance, interface between CN 3 (TLS region consisting of B and BnT cells) and CN 4 (Plasma- and T-cell dominated) turns to SC 3_4. 
#On the other hand, the core of CN 3 becomes SC 3, since for the neighborhood for these cells is just the cellular neighborhood itself.
p1 <- plotSpatial(ad[,ad$ROI == "PM102-ROI1"], 
                  node_color_by = "cn_celltypes", 
                  img_id = "ROI", 
                  node_size_fix = 0.3, coords = c("X_loc", "Y_loc"),
                  colPairName = "expansion_interaction_graph") +
  scale_color_brewer(palette = "Accent")

p2 <- plotSpatial(ad[,ad$ROI == "PM102-ROI1"], 
                  node_color_by = "spatial_context", 
                  img_id = "ROI", 
                  node_size_fix = 0.3, coords = c("X_loc", "Y_loc"),
                  colPairName = "expansion_spatialcontext_graph") +
  scale_color_manual(values = col_SC)

p1 + p2

#Next, we filter the SCs based on user-defined thresholds for number of group entries (here at least 3 ROIs) 
#and/or total number of cells (here minimum of 100 cells) per SC with filterSpatialContext.

## Filter spatial contexts
# By number of group entries
ad <- filterSpatialContext(ad, 
                            entry = "spatial_context",
                            group_by = "ROI", 
                            group_threshold = 6)

plotSpatial(ad, 
            node_color_by = "spatial_context_filtered", 
            img_id = "ROI", 
            node_size_fix = 0.1, coords = c("X_loc", "Y_loc"),
            colPairName = "expansion_spatialcontext_graph") +
  scale_color_manual(values = col_SC, limits = force)

# By number of group entries and total number of cells
ad <- filterSpatialContext(ad, 
                            entry = "spatial_context_filtered",
                            group_by = "ROI", 
                            group_threshold = 6,
                            cells_threshold = 500)

plotSpatial(ad, 
            node_color_by = "spatial_context_filtered", 
            img_id = "ROI", 
            node_size_fix = 0.05, coords = c("X_loc", "Y_loc"),
            colPairName = "expansion_spatialcontext_graph") +
  scale_color_manual(values = col_SC, limits = force)

#Lastly, we can use the plotSpatialContext function to generate SC graphs, 
#analogous to CN combination maps in (Bhate et al. 2022). 
#Returned objects are ggplots, which can be easily modified further. 
#We will create a SC graph for the filtered SCs here.

## Plot spatial context graph 

# Colored by name and size by n_cells
plotSpatialContext(ad[,ad$Group == "CM2"], 
                   entry = "spatial_context_filtered",
                   group_by = "ROI",
                   node_color_by = "name",
                   node_size_by = "n_cells",
                   node_label_color_by = "name")


# Colored by n_cells and size by n_group                   
plotSpatialContext(ad[,ad$Group == "Non-CM"], 
                   entry = "spatial_context_filtered",
                   group_by = 'ROI',
                   node_color_by = "n_cells",
                   node_size_by = "n_group",
                   node_label_color_by = "n_cells") +
  scale_color_viridis()


#8-  Patch Detection Analysis

#The previous section focused on detecting cellular neighborhoods in a rather unsupervised fashion. 
#However, the imcRtools package also provides methods for detecting spatial compartments in a supervised fashion. 
#The patchDetection function allows the detection of connected sets of similar cells as proposed by (Hoch et al. 2022). 
#In the following example, we will use the patchDetection function to detect function to detect tumor patches in three steps:

#A- Find connected sets of tumor cells (using the steinbock graph).
#B- Components which contain less than 10 cells are excluded.
#C- Expand the components by 1µm to construct a concave hull around the patch and include cells within the patch.

ad <- patchDetection(ad, 
                      patch_cells = ad$hierarchy == "Lymphoid",
                      img_id = "ROI",
                      expand_by = 1,
                      min_patch_size = 10, coords = c("X_loc", "Y_loc"),
                      colPairName = "expansion_interaction_graph")

saveRDS(ad, "adata_subset3.rds")

plotSpatial(ad, 
            node_color_by = "patch_id", 
            img_id = "ROI", 
            node_size_fix = 0.1, coords = c("X_loc", "Y_loc")) +
  theme(legend.position = "none")

#We can now calculate the fraction of T cells within each lymphoid patch to roughly estimate T cell infiltration.
library(tidyverse)
colData(ad) %>% as_tibble() %>%
  group_by(patch_id, Group) %>%
  summarize(Tcell_count = sum(cell_type == "CD8 T cells" | cell_type == "CD4 T cells"),
            patch_size = n(),
            Tcell_freq = Tcell_count / patch_size) %>%
  ggplot() +
  geom_point(aes(log10(patch_size), Tcell_freq, color = Group)) +
  theme_classic()

#We can now calculate the fraction of B cells within each lymphoid patch to roughly estimate B cell infiltration.
colData(ad) %>% as_tibble() %>%
  group_by(patch_id, Group) %>%
  summarize(Bcell_count = sum(cell_type == "B cells"),
            patch_size = n(),
            Bcell_freq = Bcell_count / patch_size) %>%
  ggplot() +
  geom_point(aes(log10(patch_size), Bcell_freq, color = Group)) +
  theme_classic()

#We can now calculate the fraction of Macrophages within each patch to roughly estimate infiltration.
colData(ad) %>% as_tibble() %>%
  group_by(patch_id, Group) %>%
  summarize(DC_count = sum(cell_type == "Dendritic cells"),
            patch_size = n(),
            DC_freq = DC_count / patch_size) %>%
  ggplot() +
  geom_point(aes(log10(patch_size), DC_freq, color = Group)) +
  theme_classic()

#We can now measure the size of each patch using the patchSize function and visualize tumor patch distribution per patient.
patch_size <- patchSize(ad, "patch_id", coords = c("X_loc", "Y_loc"))

patch_size <- merge(patch_size, 
                    colData(ad)[match(patch_size$patch_id, ad$patch_id),], 
                    by = "patch_id")

ggplot(as.data.frame(patch_size)) + 
  geom_boxplot(aes(Patient, log10(size))) +
  geom_point(aes(Patient, log10(size)))

ggplot(as.data.frame(patch_size)) + 
  geom_boxplot(aes(Group, log10(size))) +
  geom_point(aes(Group, log10(size)))

#The minDistToCells function can be used to calculate the minimum distance between each cell and a cell set of interest. 
#Here, we highlight its use to calculate the minimum distance of all cells to the detected tumor patches. 
#Negative values indicate the minimum distance of each tumor patch cell to a non-tumor patch cell.
spe <- minDistToCells(spe, 
                      x_cells = !is.na(spe$patch_id), 
                      img_id = "sample_id")

plotSpatial(spe, 
            node_color_by = "distToCells", 
            img_id = "sample_id", 
            node_size_fix = 0.5) +
  scale_color_gradient2(low = "dark blue", mid = "white", high = "dark red")

#Finally, we can observe the minimum distances to tumor patches in a cell type specific manner.
library(ggridges)
ggplot(as.data.frame(colData(spe))) + 
  geom_density_ridges(aes(distToCells, celltype, fill = celltype)) +
  geom_vline(xintercept = 0, color = "dark red", size = 2) +
  scale_fill_manual(values = metadata(spe)$color_vectors$celltype)

#9- Interaction analysis

#The next section focuses on statistically testing the pairwise interaction between all cell types of the dataset. 
#For this, the imcRtools package provides the testInteractions function which implements the interaction testing strategy proposed by (Schapiro et al. 2017).
#Per grouping level (e.g., image), the testInteractions function computes the averaged cell type/cell type interaction count 
#and computes this count against an empirical null distribution which is generated by permuting all cell labels (while maintaining the tissue structure).
#In the following example, we use the steinbock generated spatial interaction graph and estimate the interaction or avoidance between cell types in the dataset.

library(scales)
out <- testInteractions(ad, 
                        group_by = "ROI",
                        label = "pheno_cluster", 
                        colPairName = "expansion_interaction_graph",
                        BPPARAM = SerialParam(RNGseed = 221029))

head(out)

saveRDS(ad, "adata_subset3.rds")

outCM2 <- testInteractions(ad[,ad$Group == "CM2"], 
                        group_by = "ROI",
                        label = "pheno_cluster", 
                        colPairName = "expansion_interaction_graph",
                        BPPARAM = SerialParam(RNGseed = 221029))

outNonCM <- testInteractions(ad[,ad$Group == "Non-CM"], 
                           group_by = "ROI",
                           label = "pheno_cluster", 
                           colPairName = "expansion_interaction_graph",
                           BPPARAM = SerialParam(RNGseed = 221029))


#The returned DataFrame contains the test results per grouping level (in this case the ROI, group_by),
#“from” cell type (from_label) and “to” cell type (to_label). 
#The sigval entry indicates if a pair of cell types is significantly interacting (sigval = 1),
#if a pair of cell types is significantly avoiding (sigval = -1) or if no significant interaction or avoidance was detected.
#These results can be visualized by computing the sum of the sigval entries across all images:
outNonCM %>% as_tibble() %>%
  group_by(from_label, to_label) %>%
  summarize(sum_sigval = mean(sigval, na.rm = TRUE)) %>%
  ggplot() +
  geom_tile(aes(from_label, to_label, fill = sum_sigval)) +
  scale_fill_gradient2(low = muted("blue"), mid = "white", high = muted("red")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#The imcRtools package further implements an interaction testing strategy proposed by (Schulz et al. 2018) 
#where the hypothesis is tested if at least n cells of a certain type are located around a target cell type (from_cell). 
#This type of testing can be performed by selecting method = "patch" and specifying the number of patch cells via the patch_size parameter.

out <- testInteractions(spe, 
                        group_by = "sample_id",
                        label = "celltype", 
                        colPairName = "neighborhood",
                        method = "patch", 
                        patch_size = 3,
                        BPPARAM = SerialParam(RNGseed = 221029))

out %>% as_tibble() %>%
  group_by(from_label, to_label) %>%
  summarize(sum_sigval = sum(sigval, na.rm = TRUE)) %>%
  ggplot() +
  geom_tile(aes(from_label, to_label, fill = sum_sigval)) +
  scale_fill_gradient2(low = muted("blue"), mid = "white", high = muted("red")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#These results are comparable to the interaction testing presented above. 
#The main difference comes from the lack of symmetry. We can now for example see that 3 or more myeloid cells sit around CD4 T cells 
#while this interaction is not as strong when considering CD4 T cells sitting around myeloid cells.

#Finally, we save the updated SpatialExperiment object.
saveRDS(spe, "spe.rds")






