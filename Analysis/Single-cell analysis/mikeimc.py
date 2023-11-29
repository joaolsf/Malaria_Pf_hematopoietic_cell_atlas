import pandas as pd
import scanpy as sc
    
def celltable_to_adata(column_properties,cell_table,dictionary,misc_table=False,dict_index='ROI',quiet=False,marker_normalisation=None,xy_as_obs=True):
    
    """This function will load any cell_table irrespective of which pipeline used, as long as the 
    Args:
        column_properties (.csv):
            Annotated list of all the columns in the file. Valid labels and how the are handled:
                marker - Will be added as a maker
                roi - Unique image identifier, e.g, name of the region of interest or image
                x - the location of the cell in the x axis
                y - the location of the cell i the y axis
                observation - the column will be added as observation, eg a categorial variable such as patient or treatment
        cell_table:
            The master cell_table .csv file as produced by whatever pipeline you are using
        dictionary:
            This will be used to add extra columns for each ROI based upon other characteristics of your ROI. This will allow you to easily group analyses later, for example by specifying that specific ROIs came from the same patient, or are grouped in some other way.
        misc_table:
            If this is set as True, then a separate DataFrame with all the misc columns will also be returned
        dict_index:
            The name of the column used as the index in the dictionary file, defaults to 'ROI'
    Returns:
        AnnData:
            Completely annotated dataset
        
        If misc_table=True, then will also return a separate DataFrame with all the misc columns
    """
    import pandas as pd
    import scanpy as sc
    
    #This stops a warning getting returned that we don't need to worry about
    pd.set_option('mode.chained_assignment',None)
    
    #Load in the data to Pandas dataframe
    master = pd.read_csv(cell_table, low_memory=False)
    columns_table = pd.read_csv(column_properties, low_memory=False)

    # Reset inputs
    markers = []
    ROIs = []
    misc = []
    observation = []
    X_loc = ""
    Y_loc = ""

    # This will run through the columns table and sort out which columns from the raw table go where in the resulting adata table
    for index, row in columns_table.iterrows():
        if row['Category']=='marker':
            markers.append(row['Column'])
            if quiet==False:
                print(row['Column'] + ' --- added as MARKER')
        elif row['Category']=='roi':
            if ROIs == []:
                ROIs = row['Column']
                if quiet==False:
                    print(row['Column'] + ' --- added as ROI information')
            else:
                stop('ERROR: Multiple ROI columns')
        elif row['Category']=='x':
            if quiet==False:    
                print(row['Column'] + ' ---  added as X location')
            X_loc = row['Column']
        elif row['Category']=='y':
            if quiet==False:
                print(row['Column'] + ' ---  added as Y location')
            Y_loc = row['Column']
        elif row['Category']=='observation':
            if quiet==False:
                print(row['Column'] + ' --- added as OBSERVATION') 
            observation.append(row['Column'])  
        elif row['Category']=='misc':
            if quiet==False:
                print(row['Column'] + ' --- added to MISC dataframe')
            misc.append(row['Column'])
        else:
            if quiet==False:
                print(row['Column'] + " ---  DISCARDED")

    #Error catching to make sure markers or ROIs were identified
    if markers==[]:
        stop("ERROR: No markers were identified")
    if ROIs==[]:
        stop("ERROR: No ROI identifiers were found")    
    
    # Create the anndata firstly with the marker information
    if marker_normalisation==None:
        adata = sc.AnnData(master[markers])
    elif marker_normalisation=='99th':
        raw_markers = master[markers]
        markers_normalised = raw_markers.div(raw_markers.quantile(q=.99)).clip(upper=1)
        adata = sc.AnnData(markers_normalised)
        print('\nData normalised to 99th percentile')

    # Add in a master index to uniquely identify each cell over the entire dataset    
    adata.obs['Master_Index']=master.index.copy()

    # Add ROI information
    adata.obs['ROI']= master[ROIs].values.tolist()    
 
    # Add in other observations if found
    if not observation==[]:
        for a in observation:
            adata.obs[a]=master[a].tolist()

    # Add in spatial data if it's provided
    if not X_loc=="" and not Y_loc=="":
        adata.obsm['spatial'] = master[[X_loc, Y_loc]].to_numpy()
        
        #Adds in the x and y a observations. These wont be used by scanpy, but can be useful for easily exporting later.
        if xy_as_obs==True:
            print('\nX and Y locations also stored as observations (and in .obsm[spatial])')
            adata.obs['X_loc']=master[X_loc].values.tolist()
            adata.obs['Y_loc']=master[Y_loc].values.tolist()       
        
    else:
        print("No or incomplete spatial information found")

    #If a dictionary was specified, then it will be used here to populate extra columns as observations in the anndata
    if not dictionary==None:
            #Read dictionary from file
            master_dict = pd.read_csv(dictionary, low_memory=False)

            # Setup dictionary
            m = master_dict.set_index(dict_index).to_dict()

            # Add the new columns based off the dictionary file
            for i in master_dict.columns:
                if not i==dict_index:
                    master[i]=master[ROIs].map(m[i])
                    adata.obs[i]=master[i].values.tolist()
    else:
        print("No dictionary found")

    # Add any misc data to spare dataframe
    if not misc==[]:
        misc_data = master[misc]
        misc_data['Master_Index']=misc_data.index.copy()
    
    if misc_table==True:
        return adata, misc_data
    else:
        return adata
    
def remove_list(full_list,to_remove):
    filtered =  [m for m in full_list if m not in to_remove]
    return filtered


def return_adata_xy(adata):
    import numpy as np
    X, Y = np.split(adata.obsm['spatial'],[-1],axis=1)
    return X, Y


def pop_distribution(adata_analysis,groupby_name,observs):
    import seaborn as sb
    import scanpy as sc
    import pandas as pd
    #adata_analysis = adata_myeloid_2
    #groupby_name = 'pheno_louvain'
    #observs = ['ROI','HEClass','Type']

    # Matrix plot
    sc.tl.dendrogram(adata_analysis, groupby = groupby_name)
    sc.pl.matrixplot(adata_analysis, adata_analysis.var_names, groupby=groupby_name, dendrogram=True, save=True)

    #Create a color palette
    color_pal = sb.color_palette("Paired")

    for i in observs:
        tmp = pd.crosstab(adata_analysis.obs[groupby_name],adata_analysis.obs[i], normalize='index')
        tmp.plot.bar(stacked=True,color=color_pal, figsize=(16, 6)).legend(bbox_to_anchor=(1.1, 1))

    # Plotting with Seaborn
    graph = sb.lmplot(data = adata_analysis.obs,x = 'X_loc',y='Y_loc',hue =groupby_name,palette = 'bright',height = 8,col = 'ROI',col_wrap = 3,fit_reg = False)
    graph.savefig("MappedPlots.png")