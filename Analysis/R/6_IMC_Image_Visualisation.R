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
library(EBImage)

#1- Load data
#First, we will read in the previously generated SpatialExperiment object.

#The following section describes how to visualize the abundance of biomolecules (e.g. protein or RNA) 
#as well as cell-specific metadata on images. 
#Section 11.1 focuses on visualizing pixel-level information including the generation of pseudo-color composite images. 
#Section 11.2 highlights the visualization of cell metadata (e.g. cell phenotype) as well as summarized pixel intensities on cell segmentation masks.

#The cytomapper R/Bioconductor package was developed to support the handling and visualization of multiple multi-channel images and segmentation masks (Eling et al. 2020). 
#The main data object for image handling is the CytoImageList container which we used in Section 5 to store multi-channel images and segmentation masks.
#We will first read in the previously processed data and randomly select 3 images for visualization purposes.

library(SpatialExperiment)
library(cytomapper)
spe <- readRDS("spe.rds")
spe2 <- readRDS("adata_subset3.rds")

images <- readRDS("images.rds")
masks <- readRDS("masks.rds")

# Sample images
set.seed(220517)
cur_id <- sample(unique(spe$sample_id), 3)
cur_images <- images[names(images) %in% cur_id]
cur_masks <- masks[names(masks) %in% cur_id]

#1.1- Pixel visualization

#The following section gives examples for visualizing individual channels or multiple channels as pseudo-color composite images. 
#For this the cytomapper package exports the plotPixels function which expects a CytoImageList object storing one or multiple multi-channel images. 
#In the simplest use case, a single channel can be visualized as follows:

#The bcg parameter (default c(0, 1, 1)) stands for “background”, “contrast”, “gamma” and controls these attributes of the image. 
#This parameter takes a named list where each entry specifies these attributes per channel. 
#The first value of the numeric vector will be added to the pixel intensities (background); 
#pixel intensities will be multiplied by the second entry of the vector (contrast); 
#pixel intensities will be exponentiated by the third entry of the vector (gamma). 
#In most cases, it is sufficient to adjust the second (contrast) entry of the vector.
plotPixels(cur_images, 
           colour_by = "CD20",
           bcg = list(CD20 = c(0, 7, 1)))

#The following example highlights the visualization of 6 markers (maximum allowed number of markers) at once per image. 
#The markers indicate the spatial distribution of tumor cells (E-caherin), T cells (CD3), B cells (CD20), CD8+ T cells (CD8a),
#plasma cells (CD38) and proliferating cells (Ki67).
plotPixels(cur_images, 
           colour_by = c("CD235ab", "CD68", "CD45", "CD8a", "CD3", "CD20"),
           bcg = list(CD68 = c(0, 5, 1),
                      CD3 = c(0, 5, 1),
                      CD20 = c(0, 5, 1),
                      CD8a = c(0, 5, 1),
                      CD138 = c(0, 8, 1),
                      IgD = c(0, 5, 1)))

#1.1.1- Adjusting colors
#The default colors for visualization are chosen by the additive RGB (red, green, blue) color model. 
#For six markers the default colors are: red, green, blue, cyan (green + blue), magenta (red + blue), yellow (green + red). 
#These colors are the easiest to distinguish by eye. 
#However, you can select other colors for each channel by setting the colour parameter:

#The colour parameter takes a named list in which each entry specifies the colors from which a color gradient is constructed via colorRampPalette. 
#These are usually vectors of length 2 in which the first entry is "black" and the second entry specifies the color of choice. 
#Although not recommended, you can also specify more than two colors to generate a more complex color gradient.

plotPixels(cur_images, 
           colour_by = c("CD68", "CD3", "CD20"),
           bcg = list(CD68 = c(0, 5, 1),
                      CD3 = c(0, 5, 1),
                      CD20 = c(0, 5, 1)),
           colour = list(CD68 = c("black", "burlywood1"),
                         CD3 = c("black", "cyan2"),
                         CD20 = c("black", "firebrick1")))


#1.1.2- Image normalization
#As an alternative to setting the bcg parameter, images can first be normalized. 
#Normalization here means to scale the pixel intensities per channel between 0 and 1 (or a range specified by the ft parameter in the normalize function). 
#By default, the normalize function scales pixel intensities across all images contained in the CytoImageList object (separateImages = FALSE). 
#Each individual channel is scaled independently (separateChannels = TRUE).

#After 0-1 normalization, maximum pixel intensities can be clipped to enhance the contrast of the image (setting the inputRange parameter). 
#In the following example, the clipping to 0 and 0.2 is the same as multiplying the pixel intensities by a factor of 5.

# 0 - 1 channel scaling across all images
norm_images <- normalize(cur_images)

# Clip channel at 0.2
norm_images <- normalize(norm_images, inputRange = c(0, 0.2))

plotPixels(cur_images, 
           colour_by = c("CD235ab", "CD68", "CD45", "CD8a", "CD3", "CD20"))

#The default setting of scaling pixel intensities across all images ensures comparable intensity levels across images. 
#Pixel intensities can also be scaled per image therefore correcting for staining/expression differences between images:
# 0 - 1 channel scaling per image
norm_images <- normalize(cur_images, separateImages = TRUE)

# Clip channel at 0.2
norm_images <- normalize(norm_images, inputRange = c(0, 0.2))

plotPixels(norm_images, 
           colour_by = c("CD235ab", "CD68", "CD45", "CD8a", "CD3", "CD20"))

#Finally, the normalize function also accepts a named list input for the inputRange argument. 
#In this list, the clipping range per channel can be set individually:
# 0 - 1 channel scaling per image
norm_images <- normalize(cur_images, 
                         separateImages = TRUE,
                         inputRange = list(CD235ab = c(0, 50), 
                                           CD68 = c(0, 30),
                                           CD45 = c(0, 40),
                                           CD8a = c(0, 50),
                                           CD3 = c(0, 10),
                                           CD20 = c(0, 70)))

plotPixels(norm_images, 
           colour_by = c("Ecad", "CD3", "CD20", "CD8a", "CD38", "Ki67"))


#2- Cell visualization

#In the following section, we will show examples on how to visualize single cells either as segmentation masks 
#or outlined on composite images. 
#This type of visualization allows to observe the spatial distribution of cell phenotypes, 
#the visual assessment of morphological features and quality control in terms of cell segmentation and phenotyping.

#2.1- Visualzing metadata
#The cytomapper package provides the plotCells function that accepts a CytoImageList object containing segmentation masks. 
#These are defined as single channel images where sets of pixels with the same integer ID identify individual cells. 
#This integer ID can be found as an entry in the colData(spe) slot and as pixel information in the segmentation masks. 
#The entry in colData(spe) needs to be specified via the cell_id argument to the plotCells function. 
#In that way, data contained in the SpatialExperiment object can be mapped to segmentation masks. 
#For the current dataset, the cell IDs are stored in colData(spe)$ObjectNumber.

#As cell IDs are only unique within a single image, plotCells also requires the img_id argument. 
#This argument specifies the colData(spe) as well as the mcols(masks) entry that stores the unique image name from which each cell was extracted. 
#In the current dataset the unique image names are stored in colData(spe)$sample_id and mcols(masks)$sample_id.

#Providing these two entries that allow mapping between the SpatialExperiment object and segmentation masks, 
#we can now color individual cells based on their cell type:
celltype <- setNames(c("#3F1B03", "#F4AD31", "#894F36", "#1C750C", "#EF8ECC", 
                       "#6471E2", "#4DB23B", "#F4800C", "#BF0A3D"),
                     c("B cells", "CD4 T cells", "CD8 T cells", "Dendritic cells", "Endothelial cells", 
                       "Macrophages", "NK cells", "RBCs", "Smooth Muscle Cells"))

metadata(spe)$color_vectors$celltype <- celltype

cur_masks2 <- getImages(masks, 1)

plotCells(cur_masks2,
          object = spe, 
          cell_id = "ObjectNumber", img_id = "sample_id",
          colour_by = "cell_type",
          image_title = list(text = mcols(cur_masks2)$sample_id,
                             position = "topright",
                             colour = "grey",
                             margin = c(5,5),
                             font = 2,
                             cex = 2))

plotCells(cur_masks,
          object = spe, 
          cell_id = "ObjectNumber", img_id = "sample_id",
          colour_by = "cell_type",
          colour = list(cell_type = metadata(spe)$color_vectors$celltype))

#For consistent visualization, the plotCells function takes a named list as color argument. 
#The entry name must match the colour_by argument.
plotCells(cur_masks2,
          object = spe, 
          cell_id = "ObjectNumber", img_id = "sample_id",
          colour_by = "pheno_cluster",
          colour = list(pheno_cluster = metadata(spe)$color_vectors$pheno_cluster))

#If only individual cell types should be visualized, the SpatialExperiment object can be subsetted (e.g., to only contain CD8+ T cells). 
#In the following example CD8+ T cells are colored in red and all other cells that are not contained in the dataset are colored in white (as set by the missing_color argument).
CD8 <- spe[,spe$celltype == 'CD8 T cells']

plotCells(cur_masks,
          object = CD8, 
          cell_id = "ObjectNumber", img_id = "sample_id",
          colour_by = "celltype",
          colour = list(celltype = c(CD8 = "red")),
          missing_colour = "white")

#In terms of visualizing metadata, any entry in the colData(spe) slot can be visualized 
#the plotCells function automatically detects if the entry is continuous or discrete. 
#In this fashion, we can now visualize the area of each cell:
plotCells(cur_masks,
          object = spe, 
          cell_id = "ObjectNumber", img_id = "sample_id",
          colour_by = "area")

#2.2- Visualizating expression
#Similar to visualizing single-cell metadata on segmentation masks, 
#we can use the plotCells function to visualize the aggregated pixel intensities per cell. 
#In the current dataset pixel intensities were aggregated by computing the mean pixel intensity per cell and per channel. 
#The plotCells function accepts the exprs_values argument (default counts) that allows selecting the assay which stores the expression values that should be visualized.

#In the following example, we visualize the asinh-transformed mean pixel intensities of the epithelial marker E-cadherin on segmentation masks.
plotCells(cur_masks,
          object = spe, 
          cell_id = "ObjectNumber", img_id = "sample_id",
          colour_by = "CD68",
          exprs_values = "counts")

#We will now visualize the maximum number of allowed markers as composites on the segmentation masks. 
#As above the markers indicate the spatial distribution of tumor cells (E-caherin), T cells (CD3), B cells (CD20), CD8+ T cells (CD8a), plasma cells (CD38) and proliferating cells (Ki67).
plotCells(cur_masks,
          object = spe, 
          cell_id = "ObjectNumber", img_id = "sample_id",
          colour_by = c("CD3", "CD20", "CD8a"),
          exprs_values = "counts")

#While visualizing 6 markers on the pixel-level may still allow the distinction of different tissue structures, 
#observing individual expression levels is difficult when visualizing many markers simultaneously due to often overlapping expression.

#Similarly to adjusting marker colors when visualizing pixel intensities, 
#we can change the color gradients per marker by setting the color argument:
plotCells(cur_masks,
          object = spe, 
          cell_id = "ObjectNumber", img_id = "sample_id",
          colour_by = c("Ecad", "CD3", "CD20"),
          exprs_values = "exprs",
          colour = list(Ecad = c("black", "burlywood1"),
                        CD3 = c("black", "cyan2"),
                        CD20 = c("black", "firebrick1")))

#2.3- Outlining cells on images

#The following section highlights the combined visualization of pixel- and cell-level information at once. 
#For this, besides the SpatialExperiment object, the plotPixels function accepts two CytoImageList objects. 
#One for the multi-channel images and one for the segmentation masks. 
#By specifying the outline_by parameter, the outlines of cells can now be colored based on their metadata.

#The following example first generates a 3-channel composite images displaying the expression of E-cadherin, 
#CD3 and CD20 before coloring the cells’ outlines by their cell phenotype.

plotPixels(image = cur_images,
           mask = cur_masks,
           object = spe, 
           cell_id = "ObjectNumber", img_id = "sample_id",
           colour_by = c("CD68", "CD3", "CD20"),
           outline_by = "cell_type",
           bcg = list(CD68 = c(0, 5, 1),
                      CD3 = c(0, 5, 1),
                      CD20 = c(0, 5, 1)),
           colour = list(celltype = metadata(spe)$color_vectors$cell_type),
           thick = TRUE)

#Distinguishing individual cell phenotypes is nearly impossible in the images above.
#However, the SpatialExperiment object can be subsetted to only contain cells of a single or few phenotypes. 
#This allows the selective visualization of cell outlines on composite images.
#Here, we select all CD8+ T cells from the dataset and outline them on a 2-channel composite image displaying the expression of CD3 and CD8a.

CD8 <- spe[,spe$celltype == "CD8 T cell"]

plotPixels(image = cur_images,
           mask = cur_masks,
           object = CD8, 
           cell_id = "ObjectNumber", img_id = "sample_id",
           colour_by = c("CD3", "CD8a"),
           outline_by = "celltype",
           bcg = list(CD3 = c(0, 5, 1),
                      CD8a = c(0, 5, 1)),
           colour = list(celltype = c("CD8" = "white")),
           thick = TRUE)

#3- Adjusting plot annotations

#The cytomapper package provides a number of function arguments to adjust the visual appearance of figures 
#that are shared between the plotPixels and plotCells function.
#For a full overview of the arguments please refer to ?plotting-param.
#We use the following example to highlight how to adjust the scale bar, the image title, 
#the legend appearance and the margin between images.
plotPixels(cur_images, 
           colour_by = c("Ecad", "CD3", "CD20", "CD8a", "CD38", "Ki67"),
           bcg = list(Ecad = c(0, 5, 1),
                      CD3 = c(0, 5, 1),
                      CD20 = c(0, 5, 1),
                      CD8a = c(0, 5, 1),
                      CD38 = c(0, 8, 1),
                      Ki67 = c(0, 5, 1)),
           scale_bar = list(length = 100,
                            label = expression("100 " ~ mu * "m"),
                            cex = 0.7, 
                            lwidth = 10,
                            colour = "grey",
                            position = "bottomleft",
                            margin = c(5,5),
                            frame = 3),
           image_title = list(text = mcols(cur_images)$indication,
                              position = "topright",
                              colour = "grey",
                              margin = c(5,5),
                              font = 2,
                              cex = 2),
           legend = list(colour_by.title.cex = 0.7,
                         margin = 10),
           margin = 40)

#4- Displaying individual images

#By default, all images are displayed on the same graphics device. 
#This can be useful when saving all images at once (see next section) to zoom into the individual images instead of opening each image individually. 
#However, when displaying images in a markdown document these are more accessible when visualized individually. 
#For this, the plotPixels and plotCells function accepts the display parameter that when set to "single" displays each resulting image in its own graphics device:

plotCells(cur_masks,
          object = spe, 
          cell_id = "ObjectNumber", img_id = "sample_id",
          colour_by = "pg_clusters",
          colour = list(celltype = c(dittoColors(1)[1:length(unique(spe$pg_clusters))])),
          display = "single",
          legend = NULL)

#5- Saving and returning images
#The final section addresses how to save composite images and how to return them for integration with other plots.
#The plotPixels and plotCells functions accept the save_plot argument which takes a named list of the following entries: 
#filename indicates the location and file type of the image saved to disk; 
#scale adjusts the resolution of the saved image (this only needs to be adjusted for small images).

plotCells(cur_masks,
          object = spe, 
          cell_id = "ObjectNumber", img_id = "sample_id",
          colour_by = "cell_type",
          colour = list(cell_type = metadata(spe)$color_vectors$cell_type))

#The composite images (together with their annotation) can also be returned. 
#In the following code chunk we save two example plots to variables (out1 and out2).

out1 <- plotCells(cur_masks,
                  object = spe, 
                  cell_id = "ObjectNumber", img_id = "sample_id",
                  colour_by = "celltype",
                  colour = list(celltype = metadata(spe)$color_vectors$celltype),
                  return_plot = TRUE)

out2 <- plotCells(cur_masks,
                  object = spe, 
                  cell_id = "ObjectNumber", img_id = "sample_id",
                  colour_by = c("Ecad", "CD3", "CD20"),
                  exprs_values = "exprs",
                  return_plot = TRUE)

#The composite images are stored in out1$plot and out2$plot 
#and can be converted into a graph object recognized by the cowplot package.

#The final function call of the following chunk plots both object next to each other.
library(cowplot)
library(gridGraphics)
p1 <- ggdraw(out1$plot, clip = "on")
p2 <- ggdraw(out2$plot, clip = "on")

plot_grid(p1, p2)





















