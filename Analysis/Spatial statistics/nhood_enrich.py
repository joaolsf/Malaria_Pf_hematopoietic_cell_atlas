#!/usr/bin/env python
# coding: utf-8

# # NHood Enrich 22 / 02 / 22

# In[32]:


def nhood_enrichment_hyperion(adata, cluster_identifier, ROI_column_name, ROIs_to_exclude=[],n_neighbours=4,run_initial=True,average_over_rois=True):

    import pandas as pd
    import squidpy as sq
    import scipy as sp
    
    """This function perform neighbourhood enrichment for each individual ROI, then combine them and then add them back into the original adata. This function will first perform the analysis on all the samples together, and so will overwrite any existing analyses from spatial_neighbors and nhood_enrichment. T
    Args:
        adata
            AnnData object with single cell information for all ROIs
        
        cluster_identifier
            adata.obs that specifies the clustering information/populations
        
        ROI_column_name
            adata.obs that specifies the ROI/image identifier
            
        ROIs_to_exclude
            List of ROI names to exclude
            
        n_neighs
            Number of nearest neighbours, default is 10
            
        average_over_rois
            True by default - will take of mean of counts of interactions over the ROIs. If False, will just sum.
            
        run_initial
            By default will run an initital analysis on the whole adata to setup the data structure to store the new results

    Returns:
        Adds the 'corrected' neighborhood information onto the adata
        
        If misc_table=True, then will also return a separate DataFrame with all the misc columns
    """
    
    if run_initial:
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbours, coord_type="generic")
        sq.gr.nhood_enrichment(adata, cluster_key=cluster_identifier)   
    
    
    #Load all the ROIs in the adata
    ROI_list = adata.obs[ROI_column_name].unique().tolist()
    
    #If specified, remove ROIs
       
    if ROIs_to_exclude!=[]:
        
        if isinstance(ROIs_to_exclude, str):
            ROIs_to_exclude = [ROIs_to_exclude]

        for x in ROIs_to_exclude:
            ROI_list.remove(x) 

  
    # Create empty lists which will be populated at end and made into a data frame
    pop_1 = []
    pop_2 = []
    roi = []
    value_result = []

    for i in ROI_list:
        
        print('Calculating for '+i)
        
        working_adata = adata[adata.obs[ROI_column_name]==i].copy()    

        sq.gr.spatial_neighbors(working_adata, n_neighs=n_neighbours, coord_type="generic")
        sq.gr.nhood_enrichment(working_adata, cluster_key=cluster_identifier)

        pops = working_adata.obs[cluster_identifier].cat.categories
        ne_counts = working_adata.uns[(cluster_identifier+'_nhood_enrichment')]['count']
        num_pops = range(len(ne_counts))

        #Loop across every row in the array
        for k,x in zip(ne_counts,num_pops):

                #Loop for every item in the list
                for b in num_pops:

                    value_result.append(k[b])
                    pop_1.append(pops[x])
                    pop_2.append(pops[b])
                    roi.append(i)

    #Make a dataframe of the lists        
    df = pd.DataFrame(list(zip(roi,pop_1,pop_2,value_result)),
                   columns =['ROI','Population 1', 'Population 2', 'Count'])

    if average_over_rois==True:
         #Average the counts over the different ROIs
        summary_df = df.groupby(['Population 1','Population 2']).mean()
    else:
         #Sum the counts over the different ROIs
        summary_df = df.groupby(['Population 1','Population 2']).sum()   
    
    #Calculate the z-scores of the counts
    summary_df['z-score']=sp.stats.zscore(summary_df['Count'])

    summary_df.reset_index(inplace=True)

    #Final dataframes that you can use to look at data
    final_array_zscore = summary_df.pivot(index='Population 1',columns='Population 2',values='z-score')
    final_array_count = summary_df.pivot(index='Population 1',columns='Population 2',values='Count')

    #Put back into original adata
    adata.uns[(cluster_identifier+'_nhood_enrichment')]['zscore']=final_array_zscore.to_numpy()
    adata.uns[(cluster_identifier+'_nhood_enrichment')]['count']=final_array_count.to_numpy()
    

    
# Example use on my data
#nhood_enrichment_hyperion(adata_myeloid,'final_cluster','ROI',average_over_rois=False,ROIs_to_exclude=['3B3'])
         
# Plot the graph
#sq.pl.nhood_enrichment(adata_myeloid, 
                        #cluster_key='final_cluster',
                        #mode='zscore', 
                        #figsize=(8,8), 
                        #vmin=-3, 
                        #vmax=3, 
                        #save='n_enrich_myeloid.svg', 
                        #dpi=80,
                        #method='ward')    

