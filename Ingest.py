import numpy as np
import pandas as pd
import math
import re
import time
import gc
import os
import lightgbm as lgb

def Ingest(files_wanted, features_wanted):

    tictic = time.perf_counter()

    dirname = os.path.dirname(__file__)
    
    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)
    
    #Take out the trash (Clear mem before ingest).
    gc.collect()



    ####Read in GDSC data####
    print("Loading in GDSC Data and processing")
    tic = time.perf_counter()

    gdsc1 = pd.read_csv(os.path.join(dirname, r'Data/GDSC1_fitted_dose_response_15Oct19.csv'), 
                        usecols=['CELL_LINE_NAME', 'LN_IC50', 'DRUG_NAME', 'PUTATIVE_TARGET']) 

    gdsc2 = pd.read_csv(os.path.join(dirname, r'Data/GDSC2_fitted_dose_response_15Oct19.csv'), 
                        usecols=['CELL_LINE_NAME', 'LN_IC50', 'DRUG_NAME', 'PUTATIVE_TARGET']) 
    

    #Combine the two GDSC datasets using .copy() to prevent mem references to the old datasets
    gdsc =  gdsc1.append(pd.DataFrame(data = gdsc2), ignore_index=True).copy()
 
    #Change cell line column name
    gdsc.rename(columns={'CELL_LINE_NAME':"cell_line"}, inplace=True)

    #Remove '-' from CELL_LINE_NAMES
    f = lambda x: re.sub('[-]', '' , x)
    gdsc.iloc[:,0] = gdsc.iloc[:,0].apply(f)
    print('\n', gdsc.head())
    
    gdsc.drop_duplicates(subset=['cell_line', 'DRUG_NAME'], inplace = True, keep = 'last')
    
    #Recover Mem
    del gdsc1, gdsc2
    toc = time.perf_counter()
    print("\nCompleted in ", str(toc-tic), 'seconds')

    ### Load in Sample Data ####
    print("\nLoading in sample data")
    #Load in sample data
    sampleinfo = pd.read_csv((os.path.join(dirname, r'Data/Achilles_sample_info_19Q4.csv')))
    sampleinfo.rename(columns={'stripped_cell_line_name':"cell_line"}, inplace=True)
    print("Complete")
    

    #### Load in mRNA Expression Data###
    print("\nload in expression data")
    #Read in expression data
    expression = pd.read_csv((os.path.join(dirname, r'Data/CCLE_expression.csv')))
    expression.rename(columns={'Unnamed: 0':"DepMap_ID"}, inplace=True)
    names = pd.Series(expression.columns, dtype = 'string')

    def string_strip(n):
        ls = list()  
        for i in n: 
            temp = re.sub(pattern = r'\((.*?)\)', repl = '', string = i.replace(" ", ""))
            ls.append(temp)
        return(ls)

    expression.columns = string_strip(names)
    print("Complete")

    

    #Only keep the top1000 genes
    print("\nPruning data to keep top 968 most variable genes")
    l1000 = pd.read_csv(os.path.join(dirname,r"Data/landmarkGenes.txt"), sep = "\t")
    l1000_names = pd.Series(l1000.loc(axis=1)["Symbol"], dtype = 'string')
    depmap = pd.Series(data = "DepMap_ID", dtype = 'string')
    l1000_names = l1000_names.append(depmap, ignore_index=True).copy()
    

    if features_wanted == 'LINCS':
        expression = expression[expression.columns.intersection(l1000_names)]

    ### Model Feature Selection ###
    endmodel = lgb.Booster(model_file=os.path.join(dirname, r'endmodel'))
    beginmodel =  lgb.Booster(model_file=os.path.join(dirname, r'beginmodel'))

    efeatures = pd.DataFrame(index = endmodel.feature_name(), data = endmodel.feature_importance(), columns = ['Feature_Weight'])
    efeatures.sort_values(by = 'Feature_Weight', ascending = False, inplace=True)
    bfeatures = pd.DataFrame(index = beginmodel.feature_name(), data = beginmodel.feature_importance(), columns = ['Feature_Weight'])
    bfeatures.sort_values(by = 'Feature_Weight', ascending = False, inplace=True)

    efeatures = efeatures.append(bfeatures)
    efeatures.sort_values(by = 'Feature_Weight', ascending = False, inplace=True)
    efeatures.reset_index(inplace=True)
    efeatures.columns = ['Feature', 'Feature_Weight']
    efeatures.drop_duplicates(inplace=True, subset='Feature')
    

    if features_wanted == 'Model':
        top1000 = efeatures.iloc(axis=0)[0:1000]
        top1000 = top1000.iloc(axis = 1)[0]
        top1000 = top1000.iloc(axis=0)[2:]
        dep = pd.Series(data = ['DepMap_ID'])
        top = top1000.append(dep)    
        top = list(top)
        expression = expression[expression.columns.intersection(top)]


    if type(features_wanted) == int :
        top1000 = efeatures.iloc(axis=0)[0:features_wanted]
        top1000 = top1000.iloc(axis = 1)[0]
        top1000 = top1000.iloc(axis=0)[2:]
        top = top1000.append(l1000_names, ignore_index =  True) 
        top.drop_duplicates(inplace = True)
        expression = expression[expression.columns.intersection(top)]
    


    ### ENDS HERE ###

    # overlap = top1000[top1000['Feature'].isin(Inter)]
    # depmap = pd.DataFrame(expression['DepMap_ID'])
    # depmap    

    # expression['DepMap_ID'] = depmap
    
    
    #Remove all columns that only have one unique entry.
    # print("\nRemoving columns with one unique entry")
    # for col in expression.columns:
    #     if expression[col].nunique()==1:
    #         print("removed: ", col, "due to the column containing one unique entry")
    #         del expression[col]

    # print("Complete")


    # print("\nAssessing pearson correlation")
    # #Correlation matrix for the expression data
    # tic = time.perf_counter()
    # corr = expression.corr()
    # toc = time.perf_counter()
    # print("That took a while: ", str(toc-tic))

    # correlated_features = set()

    # print("\nLooping through columns to find correlation greater than 0.9")
    # #"Looping through columns to find correlation greater than 0.9"
    # #Better Correlation Loop
    # for i in range(0,len(corr.columns)):
    #     for j in range(i+1,len(corr.columns)):
    #         if abs(corr.iloc[i, j]) > 0.9:
    #             colname = (corr.pop(corr.iloc[i].name)).name
    #             correlated_features.add(colname)
    #             break #No need to keep checking rows if one row is correlated!
    # print("Complete")

    # print("\nRemoved", str(len(correlated_features)), "correlated features")
    # expression.drop(labels=correlated_features, axis=1, inplace=True)  
    # print("Complete")

    #remove features with very little variation
    from sklearn.feature_selection import VarianceThreshold
    constant_filter = VarianceThreshold(threshold=0.01)
    constant_filter.fit(expression.iloc[:,1:])

    print("\nScale the expression data")
    #Preprocess the genetic information to scale it
    from sklearn import preprocessing
    sc = preprocessing.StandardScaler()
    sc.fit(expression.iloc[:,1:])
    expression.iloc[:,1:] = sc.transform(expression.iloc[:,1:])
    print("Complete")

    print("\nAdd cell line names to mRNA data")
    #InnerJoin to add cell line names to mRNA data
    expression = pd.merge(sampleinfo.iloc[:,0:2], expression, on='DepMap_ID', how='inner')
    expression.drop(labels='DepMap_ID', axis=1, inplace=True, errors='raise')
    print("Complete")




    # #### load in DNA Methylation data#####
    # print("\nLoading in CCLE DNA Methylation data")
    # methylation = pd.read_csv(os.path.join(dirname, r'Data/CCLE_methylation_processed.csv'))
    # methylation = methylation.drop(columns = 'avg_coverage')
    # methylation = methylation.transpose(copy = True)
    # methylation.reset_index(inplace = True)
    # methylation.iloc[0,0] = 'cell_line'
    # methylation.columns = methylation.loc(axis = 0)[0]
    # methylation = methylation.drop(index = 0)
    # methylation = methylation.infer_objects()

    # #Rename the index rows as just DMS53, SW116, etc. Thereby dropping the origin descriptor
    # clines = pd.Series(methylation.iloc(axis=1)[0], dtype='string')
    # lets = clines.str.split(pat = '_', expand = True)
    # lets = lets.iloc(axis = 1)[0]
    # methylation.loc(axis = 1)['cell_line'] = lets
    
    # #Drop duplicate cell_lines
    # methylation.drop_duplicates(subset = 'cell_line', inplace = True)
    
    # #How to deal with NAs??? Na means that the gene doesn't exist at that posistion? So should be exclude the column
    # #because it isn't applicable to all the cell lines? or impute a value even thought it would be wrong because the
    # #gene doesn't even exist in that posistion??
    # #Set all Na to 0? The issue is that 0 is as important as 1 because it implies the gene is not methylated at all??
    # #Current decision is to drop all columns with NAs for ease of use.
    # methylation.dropna(axis = 1, how = 'any', inplace = True)
    # methylation.reset_index(inplace = True)
    # methylation.drop(axis = 1, labels = 'index', inplace = True)
    
    # cols=pd.Series(methylation.columns)
    # for dup in methylation.columns[methylation.columns.duplicated(keep=False)]: 
    #     cols[methylation.columns.get_loc(dup)] = ([dup + '_' + str(d_idx) 
    #                                     if d_idx != 0 
    #                                     else dup 
    #                                     for d_idx in range(methylation.columns.get_loc(dup).sum())]
    #                                     )
    # methylation.columns=cols

    # print("Complete")



   # ###GENE COPY NUMBER VARIATION#####
    # genecn = pd.read_csv(os.path.join(dirname, r'Data\CCLE_gene_cn.csv'))
    # genecn.rename(columns={'Unnamed: 0':"DepMap_ID"}, inplace=True)
    # genecn.dropna(axis = 0, how = 'any', inplace = True)
    # gene_cn_names = pd.Series(genecn.columns, dtype = 'string')
    # genecn.columns = string_strip(gene_cn_names)
    # genecn = pd.merge(sampleinfo.iloc[:,0:2], genecn, on='DepMap_ID', how='inner')
    # genecn.drop(labels='DepMap_ID', axis=1, inplace=True, errors='raise')
    
    # constant_filter.fit(genecn.iloc[:,1:])
    # genecn.iloc[:, 1:] = sc.fit_transform(genecn.iloc[:, 1:])
    
    # #Create correlation matrix
    # corr = genecn.corr()
    # correlated_features = set()
    
    # print("\nLooping through columns to find correlation greater than 0.9")
    # #Looping through columns to find correlation greater than 0.9"
    # #Better Correlation Loop
    # for i in range(0,len(corr.columns)):
    #     for j in range(i+1,len(corr.columns)):
    #         if abs(corr.iloc[i, j]) > 0.9:
    #             colname = (corr.pop(corr.iloc[i].name)).name
    #             correlated_features.add(colname)
    #             break #No need to keep checking rows if one row is correlated!
    # print("Complete")

    # print("\nRemoved", str(len(correlated_features)), "correlated features")
    # genecn.drop(labels=correlated_features, axis=1, inplace=True)  
    # print("Complete")

    # genecn.to_csv(r"C:\Users\joshc\OneDrive\University\Honours\Code\genecn")




    # ###Load in the Metabolomics data####
    # print("\nLoading in Metabolomics data")
    # def rm_specials(n):
    #     ls = list()  
    #     for i in n: 
    #         temp = ''.join(e for e in i if e.isalnum())
    #         ls.append(temp)
    #     return(ls)

    # metabolomics = pd.read_csv(os.path.join(dirname, r'Data/CCLE_metabolomics_20190502.csv'))
    # metabolomics = pd.merge(sampleinfo.iloc[:,0:2], metabolomics, on='DepMap_ID', how='inner')
    # metabolomics.dropna(axis = 0, how = 'any', inplace = True) 
    # metabolomics.columns = rm_specials(pd.Series(metabolomics.columns, dtype = 'string'))
    # metabolomics = metabolomics.rename(columns = {'cellline':'cell_line'})
    # metabolomics.drop(errors = 'raise', axis = 1, labels = ['DepMapID', 'CCLEID'], inplace=True)
    # print("Complete")

    # print("\nPerforming basic feature selection")
    # #Basic feature selection
    # constant_filter.fit(metabolomics.iloc[:,3:])
    # metabolomics.iloc[:, 3:] = sc.fit_transform(metabolomics.iloc[:, 3:])
    
    # ### Feature Selection for metabolomcis ###
    # #Create correlation matrix
    # corr = metabolomics.corr()
    # correlated_features = set()
    
    # print("\nLooping through columns to find correlation greater than 0.9")
    # #Looping through columns to find correlation greater than 0.9"
    # #Better Correlation Loop
    # for i in range(0,len(corr.columns)):
    #     for j in range(i+1,len(corr.columns)):
    #         if abs(corr.iloc[i, j]) > 0.9:
    #             colname = (corr.pop(corr.iloc[i].name)).name
    #             correlated_features.add(colname)
    #             break #No need to keep checking rows if one row is correlated!
    # print("Complete")

    # print("\nRemoved", str(len(correlated_features)), "correlated features")
    # metabolomics.drop(labels=correlated_features, axis=1, inplace=True)  
    # print("Complete")

    # #Remove all columns that only have one unique entry.
    # print("\nRemoving columns with one unique entry")
    # for col in metabolomics.columns:
    #     if metabolomics[col].nunique()==1:
    #         print("removed: ", col, "due to the column containing one unique entry")
    #         del metabolomics[col]

    
    ### Create X & Y keys ###
    X_to_in = 'E' #files_wanted  #'E' #input("(E)xpression, (M)etabolomics, (Me)thylation, (A)ll")
    
    if X_to_in == 'E':
        #Create mRNA/Drug training set
        print("\nAdd gdsc to X")
        X = pd.merge(gdsc, expression, on='cell_line', how='inner')
        
    # if X_to_in == 'M':
    #     #Add metabolomics 
    #     print("\nAdd metabolomics to X")
    #     X = pd.merge(gdsc, metabolomics, on='cell_line', how='inner')

    # if X_to_in == 'Me':
    #     #Add methylation
    #     print('\nAdd methylation to X')
    #     X = pd.merge(gdsc, methylation, on = 'cell_line', how = 'inner')

    # if X_to_in == "A":
    #     X = pd.merge(gdsc, expression, on='cell_line', how='inner')
    #     X = pd.merge(X, metabolomics, on='cell_line', how='inner')
    #     X = pd.merge(X, methylation, on = 'cell_line', how = 'inner')

    print("Complete")
    
    X.dropna(axis = 0, how = 'any', inplace = True)
    X.reset_index(inplace=True)
    X.drop(axis=1, labels='index', inplace=True)

    #Pop out the AUC for the y's
    print("\nPop out Y")
    y = X.pop("LN_IC50")
    print("Complete")

    # Drop Cell line before test train split
    print("\nDrop cell line")
    X.pop("cell_line")
    print("Complete")
    #Recover Memory

    print("\nConvert Putative Target and Drug name to categorical")
    X.loc(axis=1)['PUTATIVE_TARGET'] = X.loc(axis=1)['PUTATIVE_TARGET'].astype('category')
    X.loc(axis=1)['DRUG_NAME'] = X.loc(axis=1)['DRUG_NAME'].astype('category')


    ### CREATE VALIDATION DATASET ###
#     PRISM = pd.read_csv(os.path.join(dirname, r'Data/secondary-screen-dose-response-curve-parameters.csv'), 
#                         usecols=['depmap_id', 'ic50', 'name', 'target'])

#     PRISM.loc(axis = 1)['ic50'].replace([np.inf, -np.inf], np.nan, inplace=True)
#     PRISM.dropna(axis=0, how='any', subset = ['ic50'], inplace=True)
#     PRISM.loc(axis = 1)['ic50'] = np.clip(PRISM.loc(axis = 1)['ic50'], 1.0E-20, 1.0E10)

#     PRISM.loc(axis=1)['ic50'] = np.log(PRISM.loc(axis=1)['ic50'])

#     #PRISM.loc(axis = 1)['ic50'].describe()
#     #y.describe()

#     #PRISM.loc(axis = 1)['ic50'] = np.clip(PRISM.loc(axis = 1)['ic50'], 1.0E-20, 1.0E10)
#     #PRISM.loc(axis = 1)['ic50'].replace([np.inf, -np.inf], np.nan, inplace=True)
#    # PRISM.dropna(axis=0, how='any', subset = ['ic50'])
#     #PRISM.loc(axis=1)['ic50'] = np.log(PRISM.loc(axis=1)['ic50'])

#     #PRISM.loc(axis = 1)['ic50'].describe() 


#     PRISM.rename(columns={'ic50' : 'LNIC_50', 'depmap_id':'DepMap_ID'}, inplace=True)
#     PRISM = pd.merge(sampleinfo.iloc[:,0:2], PRISM, on='DepMap_ID', how='inner')
#     PRISM.drop(labels='DepMap_ID', axis=1, inplace=True, errors='raise')
#     PRISM.rename(columns={'name':'DRUG_NAME', 'target':'PUTATIVE_TARGET'}, inplace=True)
#     PRISM = pd.merge(PRISM, expression, on = 'cell_line', how = 'inner')  
#     PRISM.loc(axis=1)['PUTATIVE_TARGET'] = PRISM.loc(axis=1)['PUTATIVE_TARGET'].astype('category')
#     PRISM.loc(axis=1)['DRUG_NAME'] = PRISM.loc(axis=1)['DRUG_NAME'].astype('category')

#     #Find the intersecting drugs between GDSC and PRISM
#     drugs = pd.Series(np.intersect1d(gdsc.loc(axis=1)['DRUG_NAME'], PRISM.loc(axis=1)['DRUG_NAME']))
#     sset = PRISM[PRISM['DRUG_NAME'].isin(drugs)]
#     sset.dropna(how = 'any', subset = [ 'PUTATIVE_TARGET'], inplace=True)    
#     sset.reset_index(inplace=True)
#     sset.drop(columns = 'index', inplace=True)

#     y_val = sset.pop('LNIC_50')
#     sset.drop(columns = 'cell_line', inplace=True)
#     X_val = sset


    print("\nRemove traces I was ever here........")
    del sampleinfo, gdsc
    gc.collect()

    val = 'n' #input("Do you want to save the files (X & y)? (y/n)")

    if val == 'y' or val == "Y":
        y.to_csv(os.path.join(dirname, 'y'))
        X.to_csv(os.path.join(dirname, 'X'))
        print("Files Saved")

    toctoc = time.perf_counter()

    print(X.info())

    print("\nIngest Completed in:", (toctoc - tictic), 'seconds')


    return X, y#, X_val, y_val
