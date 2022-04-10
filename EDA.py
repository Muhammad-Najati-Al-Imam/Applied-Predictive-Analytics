from glob import glob
from pathlib import Path
import math
import pandas as pd 
import numpy as np 
import pickle 
import time
from collections import Counter
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt 
import seaborn as sns



# A function that checks for NA values
def check_na(df):   
  na1 = df.isna().sum().sum()




# A function that checks and drops duplicate values
def check_duplicates(df):
  print("number of duplicate rows:", df.duplicated().sum())
  dropped_df = df.drop_duplicates(inplace = True)
  print("Duplicates Removed")




# A function that checks for duplicate values from EDA import uncommon_features A function that takes 2 sets of features as lists and check for their existence in the other, then returns the features that are not common (either from first or second)
def uncommon_features(first_list, second_list):
  z1 =  set(first_list) - set(second_list)
  z2 =  set(second_list) - set(first_list) 
  z3 = z1.union(z2)
  print (z3)





# A function that takes 2 dictionaries and outputs differences. This is used to check which columns have different datatypes
def compare_dtypes(first_dict, second_dict):
  different_items = {k: first_dict[k] for k in first_dict if k in second_dict and first_dict[k] != second_dict[k]}
  if not different_items:
    print("no conflicting datatypes")
  else:
    print(different_items)
    
    
#  A function that converts datatypes whenever possible to reduce memory consumption
def reduce_mem_usage(name, df):
    """     
        https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
        iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    percentage = 100 * (start_mem - end_mem) / start_mem
    
    print(f'{percentage:.1f}% memory reduction for {name} (from {start_mem:.2f} MB to {end_mem:.2f} MB)') 

    return df





# This function checks for correlations and then displays the name of feature and number of correlations (only if there is more than 1, since each feature is correlated with itself) The threshold for displaying is 0.9. The function then heatmaps the correlations between found features
def corr_checker(df):
  feature_list = []
  corr_matrix = abs(df.corr())
  for x in corr_matrix:
    if (len(corr_matrix.loc[corr_matrix[x] > 0.9]) > 1):
      #print(x, len(corr_matrix.loc[corr_matrix[x] > 0.9])) #prints the highly correlated features
      feature_list.append(x)
  corr_updated = corr_matrix.loc[feature_list, feature_list]
  sfig, ax = plt.subplots(figsize=(10,5))
  ax.set_title(df.name, fontsize=24)
  return sns.heatmap(corr_updated, annot=True, cmap = 'Blues', ax=ax)


# unused - displays correlations with all columns. Use the one below
#def corr_with_target(df):
#    corr_matrix = abs(df.corr())
#    x = corr_matrix[['converted']]
#    sfig, ax = plt.subplots(figsize=(5,12))
#    sns.heatmap(x, annot=True, cmap = 'Blues', ax=ax)
    

# This functions plots a correlation heatmap with the target variable
def corr_with_target(df):
    feature_list = []
    corr_matrix = abs(df.corr())
    feature_list.sort()
    for x in corr_matrix:
        if (len(corr_matrix.loc[corr_matrix[x] > 0.1]) > 1):
        #print(x, len(corr_matrix.loc[corr_matrix[x] > 0.9])) #prints the highly correlated features
            feature_list.append(x)
    corr_updated = corr_matrix.loc[feature_list, feature_list]
    x = corr_updated[corr_updated["converted"] > 0.1]
    x = x[["converted"]]
    sfig, ax = plt.subplots(figsize=(5,8))
    ax.set_title(df.name, fontsize=18)
    return sns.heatmap(x, annot=True, cmap = 'Blues', ax=ax)


# A function that plots a barchart of two binary variables
def visualize_binaries(df, n1, n2):
    CrosstabResult=pd.crosstab(index=df[n1],columns=df[n2])
    print(CrosstabResult)
    sns.set(rc={'figure.figsize':(8,6)})
    return CrosstabResult.plot.bar(title = df.name)


# A function that checks for binary variables when passed a dataframe
def binary_checker(df):
  colnames = []
  counter = 0
  for i, col in enumerate(df):
    if (df.iloc[:,i].isin([0,1]).all()):
      colnames.append(col)
      counter += 1
  print("There are", counter,"binary features")
  print(colnames)
    
    
    
# Z-Score Outlier Counter and Remover
# A function that displays number of outliers out of total number of rows for dataframe. Threshold is 3 standard deviations
def Zscore_outlier_counter(df, threshold):
  z_scores = stats.zscore(df)
  abs_z_scores = np.abs(z_scores)
  filtered_entries = (abs_z_scores < threshold).all(axis=1)
  counter = df[filtered_entries].count()
  percentage = round(((df.shape[0] - counter[0])*100)/df.shape[0])
  return ("There are {} rows with outliers in dataset {} out of: {} rows. percentage of outliers is: {}%. Zscore threshold is {}".format((df.shape[0] - counter[0]), df.name, df.shape[0], percentage, threshold))


# IQR Outlier Counter and Remover
# A function that displays number of outliers out of total number of rows for dataframe. Threshold is 3 standard deviations
def IQR_outlier_counter(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    df_final=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]
    counter = df_final.count()[0]
    percentage = round(((df.shape[0] - counter)*100)/df.shape[0])
    return ("There are {} rows with outliers in dataset {} out of: {} rows. percentage of outliers is: {}%".format((df.shape[0] - counter), df.name, df.shape[0], percentage))

#function for drawing boxplots
def box_plot(title ,df):
  sns.set(rc={'figure.figsize':(8,5)})
  cont_vars = []
  for column in df:
    cont_vars.append(column)
  #print(cont_vars) # print features
  ax = sns.boxplot(data=df[cont_vars], palette=sns.color_palette("husl"), showfliers=False, width=0.3)  
  plt.setp(ax.get_xticklabels(), rotation=90)
  ax.set_title(title, fontsize=18)



# A function that shows the percentage of the 1s to 0s of the "converted" feature.
def target_percent(df):
  x = round(df.converted.isin([1]).sum() * 100 / df.converted.isin([0]).sum(), 2)
  return x