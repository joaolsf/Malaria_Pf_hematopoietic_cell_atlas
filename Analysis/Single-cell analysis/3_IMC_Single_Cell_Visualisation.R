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
library(tidyverse)
library(ggridges)

#1- Load data
#First, we will read in the previously generated SpatialExperiment object.

ad <- readRDS("./IMC_Pf_spleen_redsea_corrected_complete_sce.rds")

#2- Cell-type level

#In the first section of this chapter, the grouping-level for the visualization approaches 
#will be the cell type classification from the previous script. 
#Other grouping levels (e.g. cluster assignments from Section 9.2) are possible and 
#the user should adjust depending on the chosen analysis workflow.

#2.1- Dimensionality reduction visualization

#As seen before, we can visualize single-cells in low-dimensional space. 
#Often, non-linear methods for dimensionality reduction such as tSNE and UMAP are sued. 
#They aim to preserve the distances between each cell and its neighbors in the high-dimensional space.
#Here, we will use dittoDimPlot from the DittoSeq package and plotReducedDim from the scater package 
#to visualize the fastMNN-corrected UMAP colored by cell type and expression, respectively.
#Both functions are highly flexible and return ggplot objects which can be further modified.

library(dittoSeq)
library(scater)
library(patchwork)
library(cowplot)
library(viridis)

# Define cell_type_markers and cell_state_markers 
state_markers <- c("CD45RA", "CD45RO", "CD27","CD38", "CD138", "VISTA", "PDL1", "PD1", "LAG3",
                  "TIM-3", "CD74", "HLADR", "Ki67", "IgD", "CX3CR1")

type_markers <- c("CD14", "CD16", "CD68","CD163", "CD206", "Iba1", "CD11b", "CD11c", "CD3", "CD4", "CD8", "CD19", "CD20")                  


# Add to spe
rowData(ad)$marker_class <- ifelse(rownames(ad) %in% type_markers, "type",
                                    ifelse(rownames(ad) %in% state_markers, "state", 
                                           "other"))


## UMAP colored by cell type and expression - dittoDimPlot
p1 <- dittoDimPlot(spe, var = "pheno_cluster",
                   reduction.use = "X_umap", size = 0.2,
                   do.label = TRUE, labels.size = 2, labels.highlight = TRUE) +
  scale_color_manual(values = spe@metadata[["pheno_cluster_colors"]]) +
  theme(legend.title = element_blank()) +
  ggtitle("Cell identities on UMAP")

p1

# UMAP colored by expression for all markers - plotReducedDim
plot_list  <- lapply(rownames(spe)[rowData(spe)$marker_class == "type"], function(x){
  p <- plotReducedDim(spe, dimred = "X_umap",
                      colour_by = x,
                      by_exprs_values = "X",
                      point_size = 0.2)
  return(p)
})

plot_grid(plotlist = plot_list)

#2.2- Heatmap visualisation

#Next, it is often useful to visualize single-cell expression per cell type in form of a heatmap. 
#For this, we will use the dittoHeatmap function from the DittoSeq package.
#We sub-sample the dataset to 4000 cells for ease of visualization and 
#overlay the cancer type and patient ID from which the cells were extracted.

set.seed(220818)
cur_cells <- sample(seq_len(ncol(spe)), 2000)

#Heatmap visualization - DittoHeatmap
dittoHeatmap(spe[,cur_cells], genes = rownames(spe)[rowData(spe)$marker_class == "type"],
             assay = "X", order.by = spe[,cur_cells]@colData@listData[["pheno_cluster"]],
             cluster_cols = FALSE, scale = "none",
             heatmap.colors = viridis(100), annot.by = c("pheno_cluster","Group","Patient"),
             annotation_colors = list(pheno_cluster = spe@metadata[["pheno_cluster_colors"]],
                                      Group = spe@metadata[["Group_colors"]],
                                      Patient = spe@metadata[["Patient_colors"]])
)

#Similarly, we can visualize the mean marker expression per cell type for all cells using aggregateAcrossCells from scuttle 
#and then use dittoHeatmap. We will annotate the heatmap with the number of cells per cell type.

library(scuttle)
## by cell type
celltype_mean <- aggregateAcrossCells(as(spe, "SingleCellExperiment"),  
                                      ids = spe$celltype, 
                                      statistics = "mean",
                                      use.assay.type = "exprs", 
                                      subset.row = rownames(spe)[rowData(spe)$marker_class == "type"]
)

# No scaling
dittoHeatmap(celltype_mean,
             assay = "exprs", cluster_cols = TRUE, 
             scale = "none",
             heatmap.colors = viridis(100),
             annot.by = c("celltype","ncells"),
             annotation_colors = list(celltype = metadata(spe)$color_vectors$celltype,
                                      ncells = plasma(100)))

# Min-max expression scaling
dittoHeatmap(celltype_mean,
             assay = "exprs", cluster_cols = TRUE, 
             scaled.to.max = TRUE,
             heatmap.colors.max.scaled = inferno(100),
             annot.by = c("celltype","ncells"),
             annotation_colors = list(celltype = metadata(spe)$color_vectors$celltype,
                                      ncells = plasma(100)))

#2.3-  Violin plot visualization

#The plotExpression function from the scater package allows to plot the distribution of expression values 
#across cell types for a chosen set of proteins. The output is a flexible ggplot object.

#Violin Plot - plotExpression by cell type
plotExpression(spe[,cur_cells], 
               features = rownames(spe)[rowData(spe)$marker_class == "type"],
               x = "cell_type", exprs_values = "X", 
               colour_by = "cell_type") +
  theme(axis.text.x =  element_text(angle = 90, hjust = 1))+
  scale_color_manual(values = spe@metadata[["cell_type_colors"]])

#Violin Plot - plotExpression by cell identity
plotExpression(spe[,cur_cells], 
               features = rownames(spe)[rowData(spe)$marker_class == "type"],
               x = "pheno_cluster", exprs_values = "X", 
               colour_by = "pheno_cluster") +
  theme(axis.text.x =  element_text(angle = 90, hjust = 1))+
  scale_color_manual(values = spe@metadata[["pheno_cluster_colors"]])

#2.4- Scatter plot visualization

#Moreover, a protein expression based scatter plot can be generated with dittoScatterPlot (returns a ggplot object). 
#We overlay the plot with the cell type information.

#Scatter plot
dittoScatterPlot(spe, 
                 x.var = "CD3", y.var="CD20", 
                 assay.x = "exprs", assay.y = "exprs", 
                 color.var = "celltype") +
  scale_color_manual(values = metadata(spe)$color_vectors$celltype) +
  ggtitle("Scatterplot for CD3/CD20 labelled by celltype")


#2.5- Barplot visualization

# by ROI
dittoBarPlot(spe, var = "pheno_cluster", group.by = "ROI") +
  scale_fill_manual(values = spe@metadata[["pheno_cluster_colors"]])

# by Patient
dittoBarPlot(spe, var = "pheno_cluster", group.by = "Patient") +
  scale_fill_manual(values = spe@metadata[["pheno_cluster_colors"]])

# by Patient - count
dittoBarPlot(spe, scale = "count", var = "pheno_cluster", group.by = "Patient") +
  scale_fill_manual(values = spe@metadata[["pheno_cluster_colors"]])

# by disease group - percentage
dittoBarPlot(spe, var = "pheno_cluster", group.by = "Group") +
  scale_fill_manual(values = spe@metadata[["pheno_cluster_colors"]])

#2.6- CATALYST-based visualization
#In the following, we highlight some useful visualization functions from the CATALYST package.
#To this end, we will first convert the SpatialExperiment object into a CATALYST-compatible format.

library(CATALYST)
# save spe in CATALYST-compatible object with renamed colData entries and 
# new metadata information
ad_cat <- ad 

ad_cat$sample_id <- factor(ad$ROI)
ad_cat$Patient <- factor(ad$Patient)
ad_cat$condition <- factor(ad$Group)
ad_cat$cluster_id <- factor(ad$pheno_cluster)

#add celltype information to metadata
metadata(ad_cat)$cluster_codes <- data.frame(pheno_cluster = factor(ad_cat$pheno_cluster))

#2.6.1- Pseudobulk-level MDS plot
#Pseudobulk-level multi-dimensional scaling (MDS) plots can be rendered with the exported pbMDS function.
#Here, we will use pbMDS to highlight expression similarities between cell types and subsequently for each celltype-sample-combination

# MDS pseudobulk by cell type
pbMDS(ad_cat, by = "cluster_id", assay = "X",
      features = rownames(ad_cat)[rowData(ad_cat)$marker_class == "type"], 
      label_by = "cluster_id", k = "pheno_cluster") +
  scale_color_manual(values = spe_cat@metadata[["pheno_cluster_colors"]]) 

# MDS pseudobulk by cell type and sample_id
pbMDS(spe_cat, by = "both", assay = "X",
      features = rownames(spe_cat)[rowData(spe_cat)$marker_class == "type"], 
      k = "pheno_cluster", shape_by = "Group", 
      size_by = TRUE) +
  scale_color_manual(values = spe_cat@metadata[["pheno_cluster_colors"]])

#2.6.2- Reduced dimension plot on CLR of proportions
#The clrDR function produces dimensionality reduction plots on centered log-ratios (CLR) of sample/cell type proportions across cell type/samples.
#As with pbMDS, the output plots aim to illustrate the degree of similarity between cell types based on sample proportions.

# CLR on cluster proportions across samples
clrDR(spe_cat, dr = "PCA", 
      by = "cluster_id", k = "pheno_cluster", 
      label_by = "cluster_id", arrow_col = "Group", 
      point_pal = spe_cat@metadata[["pheno_cluster_colors"]]) +
  scale_color_manual(values = spe_cat@metadata[["Group_colors"]])

clrDR(spe_cat, dr = "PCA", 
      by = "cluster_id", k = "pheno_cluster", 
      label_by = "cluster_id", arrow_col = "Patient", 
      point_pal = spe_cat@metadata[["pheno_cluster_colors"]]) +
  scale_color_manual(values = spe_cat@metadata[["Patient_colors"]])

# 2.6.3- Pseudobulk expression boxplot
# The plotPbExprs generates combined box- and jitter-plots of aggregated marker expression per cell type. 
# Here, we further split the data by disease group.

# State markers
plotPbExprs(ad_cat, assay = "X", k = "pheno_cluster", features = rownames(ad_cat)[rowData(ad_cat)$marker_class == "state"],
            facet_by = "cluster_id", color_by = "condition", ncol = 4) +
  scale_color_manual(values = ad_cat@metadata[["Group_colors"]])

# Type markers
plotPbExprs(ad_cat, assay = "X", k = "pheno_cluster", features = rownames(ad_cat)[rowData(ad_cat)$marker_class == "type"],
            facet_by = "cluster_id", color_by = "condition", ncol = 4) +
  scale_color_manual(values = ad_cat@metadata[["Group_colors"]]) 

#3- Sample-level

#3.1- Dimensionality reduction visualization
#Visualization of low-dimensional embeddings, here comparing non-corrected and fastMNN-corrected UMAPs, 
#and coloring it by sample-levels is often used for “batch effect” assessment as mentioned in Section 7.4.
#We will again use dittoDimPlot.

## UMAP colored by cell type and expression - dittoDimPlot
p1 <- dittoDimPlot(spe, var = "sample_id",
                   reduction.use = "UMAP", size = 0.2, 
                   colors = viridis(100), do.label = FALSE) +
  scale_color_manual(values = metadata(spe)$color_vectors$sample_id) +
  theme(legend.title = element_blank()) +
  ggtitle("Sample ID")

p2 <- dittoDimPlot(spe, var = "sample_id",
                   reduction.use = "UMAP_mnnCorrected", size = 0.2, 
                   colors = viridis(100), do.label = FALSE) +
  scale_color_manual(values = metadata(spe)$color_vectors$sample_id) +
  theme(legend.title = element_blank()) +
  ggtitle("Sample ID")

p3 <- dittoDimPlot(spe, var = "patient_id", 
                   reduction.use = "UMAP", size = 0.2,
                   do.label = FALSE) +
  scale_color_manual(values = metadata(spe)$color_vectors$patient_id) +
  theme(legend.title = element_blank()) +
  ggtitle("Patient ID")

p4 <- dittoDimPlot(spe, var = "patient_id", 
                   reduction.use = "UMAP_mnnCorrected", size = 0.2,
                   do.label = FALSE) +
  scale_color_manual(values = metadata(spe)$color_vectors$patient_id) +
  theme(legend.title = element_blank()) +
  ggtitle("Patient ID")

(p1 + p2) / (p3 + p4)

#3.2- Heatmap visualization
#It can be beneficial to use a heatmap to visualize single-cell expression per sample and patient. 
#Such a plot, which we will create using dittoHeatmap, can highlight biological differences across samples/patients.
#Heatmap visualization - DittoHeatmap
dittoHeatmap(spe[,cur_cells], genes = rownames(spe)[rowData(spe)$marker_class == "type"],
             assay = "exprs", order.by = c("patient_id","sample_id"),
             cluster_cols = FALSE, scale = "none",
             heatmap.colors = viridis(100), 
             annot.by = c("celltype","indication","patient_id","sample_id"),
             annotation_colors = list(celltype = metadata(spe)$color_vectors$celltype,
                                      indication = metadata(spe)$color_vectors$indication,
                                      patient_id = metadata(spe)$color_vectors$patient_id,
                                      sample_id = metadata(spe)$color_vectors$sample_id))

#aggregated mean marker expression per sample/patient allow identification of samples/patients with outlying expression patterns.
#Here, we will focus on the patient level and use aggregateAcrossCells and dittoHeatmap. 
#The heatmap will be annotated with the number of cells per patient and cancer type and displayed using two scaling options.

#by patient_id
patient_mean <- aggregateAcrossCells(as(spe, "SingleCellExperiment"),  
                                     ids = spe$patient_id, 
                                     statistics = "mean",
                                     use.assay.type = "exprs", 
                                     subset.row = rownames(spe)[rowData(spe)$marker_class == "type"]
)

# No scaling
dittoHeatmap(patient_mean,
             assay = "exprs", cluster_cols = TRUE, 
             scale = "none",
             heatmap.colors = viridis(100),
             annot.by = c("patient_id","indication","ncells"),
             annotation_colors = list(patient_id = metadata(spe)$color_vectors$patient_id,
                                      indication = metadata(spe)$color_vectors$indication,
                                      ncells = plasma(100)))

# Min-max expression scaling
dittoHeatmap(patient_mean,
             assay = "exprs", cluster_cols = TRUE, 
             scaled.to.max =  TRUE,
             heatmap.colors.max.scaled = inferno(100),
             annot.by = c("patient_id","indication","ncells"),
             annotation_colors = list(patient_id = metadata(spe)$color_vectors$patient_id,
                                      indication = metadata(spe)$color_vectors$indication,
                                      ncells = plasma(100)))

#3.3- Barplot visualization

#Complementary to displaying cell type frequencies per sample/patient, 
#we can use dittoBarPlot to display sample/patient frequencies per cell type.
dittoBarPlot(spe, var = "patient_id", group.by = "pg_clusters") +
  scale_fill_manual(values = metadata(spe)$color_vectors$patient_id)

dittoBarPlot(spe, var = "sample_id", group.by = "pg_clusters") +
  scale_fill_manual(values = metadata(spe)$color_vectors$sample_id)


#3.4- CATALYST-based visualization

#3.4.1 Pseudobulk-level MDS plot
#Expression-based pseudobulks for each sample can be compared with the pbMDS function.


# MDS pseudobulk by sample_id 
pbMDS(spe_cat, by = "sample_id", assay = 'X',
      color_by = "condition", 
      features = rownames(spe_cat)[rowData(spe_cat)$marker_class == "type"]) +
  scale_color_manual(values = spe_cat@metadata[["Group_colors"]])


#3.4.2- Reduced dimension plot on CLR of proportions

#The clrDR function can also be used to analyze similarity of samples based on cell type proportions.
# CLR on sample proportions across clusters
clrDR(spe_cat, dr = "PCA", 
      by = "sample_id", point_col = "sample_id",
      k = "celltype", point_pal = metadata(spe_cat)$color_vectors$sample_id) +
  scale_color_manual(values = metadata(spe_cat)$color_vectors$celltype)

#4- Publication-ready ComplexHeatmap

#For this example, we will concatenate heatmaps and annotations horizontally into one rich heatmap list. The grouping-level for the visualization will again be the cell type information from Section 9.3
#Initially, we will create two separate Heatmap objects for cell type and state markers.
#Then, metadata information, including the cancer type proportion and number of cells/patients per cell type, will be extracted into HeatmapAnnotation objects.
#Notably, we will add spatial features per cell type, here the number of neighbors extracted from colPair(spe) and cell area, in another HeatmapAnnotation object.
#Ultimately, all objects are combined in a HeatmapList and visualized.
### 1. Heatmap bodies ###

library(ComplexHeatmap)
library(circlize)
library(tidyverse)
set.seed(22)

# Heatmap body color 
col_exprs <- colorRamp2(c(0,1,2,3,4), 
                        c("#440154FF","#3B518BFF","#20938CFF",
                          "#6ACD5AFF","#FDE725FF"))

# Create Heatmap objects
# By celltype markers
celltype_mean <- aggregateAcrossCells(as(spe, "SingleCellExperiment"),  
                                      ids = spe$celltype, 
                                      statistics = "mean",
                                      use.assay.type = "exprs", 
                                      subset.row = rownames(spe)[rowData(spe)$marker_class == "type"])

h_type <- Heatmap(t(assay(celltype_mean, "exprs")),
                  column_title = "type_markers",
                  col = col_exprs,
                  name= "mean exprs",
                  show_row_names = TRUE, 
                  show_column_names = TRUE)

# By cellstate markers
cellstate_mean <- aggregateAcrossCells(as(spe, "SingleCellExperiment"),  
                                       ids = spe$celltype, 
                                       statistics = "mean",
                                       use.assay.type = "exprs", 
                                       subset.row = rownames(spe)[rowData(spe)$marker_class == "state"])

h_state <- Heatmap(t(assay(cellstate_mean, "exprs")),
                   column_title = "state_markers",
                   col = col_exprs,
                   name= "mean exprs",
                   show_row_names = TRUE,
                   show_column_names = TRUE)


### 2. Heatmap annotation ###

### 2.1  Metadata features

anno <- colData(celltype_mean) %>% as.data.frame %>% select(celltype, ncells)

# Proportion of indication per celltype
indication <- colData(spe) %>% 
  as.data.frame() %>% 
  select(celltype, indication) %>% 
  group_by(celltype) %>% 
  table() %>% 
  as.data.frame()

indication <- indication %>% 
  group_by(celltype) %>% 
  mutate(fra = Freq/sum(Freq)) 

indication <- indication %>% 
  select(-Freq) %>% 
  pivot_wider(id_cols = celltype, 
              names_from = indication, 
              values_from = fra) %>% 
  column_to_rownames("celltype")

# Number of contributing patients per celltype
cluster_PID <- colData(spe) %>% 
  as.data.frame() %>% 
  select(celltype, patient_id) %>% 
  group_by(celltype) %>% table() %>% 
  as.data.frame()

n_PID <- cluster_PID %>% 
  filter(Freq>0) %>% 
  group_by(celltype) %>% 
  count(name = "n_PID") %>% 
  column_to_rownames("celltype")

# Create HeatmapAnnotation objects
ha_anno <- HeatmapAnnotation(celltype = anno$celltype,
                             border = TRUE, 
                             gap = unit(1,"mm"),
                             col = list(celltype = metadata(spe)$color_vectors$celltype),
                             which = "row")

ha_meta <- HeatmapAnnotation(n_cells = anno_barplot(anno$ncells, width = unit(10, "mm")),
                             n_PID = anno_barplot(n_PID, width = unit(10, "mm")),
                             indication = anno_barplot(indication,width = unit(10, "mm"),
                                                       gp = gpar(fill = metadata(spe)$color_vectors$indication)),
                             border = TRUE, 
                             annotation_name_rot = 90,
                             gap = unit(1,"mm"),
                             which = "row")

### 2.2 Spatial features

# Add number of neighbors to spe object (saved in colPair)
n_neighbors <- colPair(spe) %>% 
  as.data.frame() %>% 
  group_by(from) %>% 
  count() %>% 
  arrange(desc(n))

spe$n_neighbors <- n_neighbors$n[match(seq_along(colnames(spe)), n_neighbors$from)]
spe$n_neighbors <- spe$n_neighbors %>% replace_na(0)

# Select spatial features and average over celltypes
spatial <- colData(spe) %>% 
  as.data.frame() %>% 
  select(area, celltype, n_neighbors)

spatial <- spatial %>% 
  select(-celltype) %>% 
  aggregate(by = list(celltype = spatial$celltype), FUN = mean) %>% 
  column_to_rownames("celltype")

# Create HeatmapAnnotation object
ha_spatial <- HeatmapAnnotation(
  area = spatial$area,
  n_neighbors = spatial$n_neighbors,
  border = TRUE,
  gap = unit(1,"mm"),
  which = "row")

### 3. Plot rich heatmap ###

# Create HeatmapList object
h_list <- h_type +
  h_state +
  ha_anno +
  ha_spatial +
  ha_meta

# Add customized legend for anno_barplot()
lgd <- Legend(title = "indication", at = colnames(indication), 
              legend_gp = gpar(fill = metadata(spe)$color_vectors$indication))

# Plot
draw(h_list,annotation_legend_list = list(lgd))

#Finally, we save the updated SpatialExperiment object.
saveRDS(spe, "spe.rds")
saveRDS(spe_cat, "spe_cat.rds")


































