"""
Created on Fri Jun  4 15:44:31 2021

@author: david.steiner
"""

import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


class DataSplitter:
    def __init__(self, df):
        self.df = df
        self.hospitals = self.df['Dataset_Id'].unique()

    def _preprocess(self, df):
        # Separate outcome and scale
        oc = df['converted'].astype('int8')
        df = df.drop('converted', axis=1)

        scaled = MinMaxScaler().fit_transform(df)
        df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
        df['converted'] = oc.values

        return df

    def per_hospital_processing(self):
        # Store here, return this, dance around
        datasets_by_facility = {}

        # Remove values with missingness below threshold
        # Preprocessing should take care of column rearrangement
        df_count = self.df.count() / self.df.shape[0]
        self.df = self.df[df_count.index]

        # Do a per-facility split and impute / scale accordingly
        for hosp in self.hospitals:
            df_hosp_data = self.df.query('Dataset_Id == @hosp')
            df_hosp_data = self._preprocess(df_hosp_data)

            datasets_by_facility[hosp] = df_hosp_data

        return datasets_by_facility


class StratifiedDatasetCreator:
    def __init__(self, full_data):
        self.df = full_data

        splitter = DataSplitter(self.df)
        self.datasets_by_facility = splitter.per_hospital_processing()

        # Store SKF split data here
        self.training_datasets = {}
        self.testing_datasets = {}

    def retching_maw(self, undersampling=False):
        for facility, df_facility in self.datasets_by_facility.items():
            X = df_facility.drop('converted', axis=1)
            y = df_facility[['converted']]

            # Store dataframes as iterables
            self.training_datasets[facility] = []
            self.testing_datasets[facility] = []

            # Splitter
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3
                )

            if undersampling == True:
                #print('Apply undersampling')
                ros = RandomUnderSampler()
                X_train, y_train = ros.fit_resample(X_train, y_train)

            training_data = pd.concat((X_train, y_train), axis=1)
            testing_data = pd.concat((X_test, y_test), axis=1)

            self.training_datasets[facility].append(training_data)
            self.testing_datasets[facility].append(testing_data)


# =============================================================================
# Create Dataset Control
# =============================================================================

def create_dataset(full_data, undersampling=True):
    data_preprocesser = StratifiedDatasetCreator(full_data=full_data) 
    data_preprocesser.retching_maw(undersampling) 
    training_datasets = data_preprocesser.training_datasets
    testing_datasets = data_preprocesser.testing_datasets
    return training_datasets, testing_datasets


