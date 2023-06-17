import sklearn as sk
import pandas as pd
import numpy as np
import csv
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import pickle
import sys

from preprocess import *




def preprocessing(df, null_thresh=1):
    # calculate the percentage of null values in each column
    null_pct = df.isnull().sum() / len(df)

    # get the list of column names where null percentage is less than or equal to 0.7
    valid_cols = null_pct[null_pct <= null_thresh].index.tolist()

    # create a new DataFrame with only valid columns
    df = df[valid_cols]
    df['patient'] = df['patient'].apply(lambda x: int(x.split("_")[1].split(".")[0]))
    df = df.sort_values(['patient', 'ICULOS'])
    df = pd.get_dummies(df, columns=['Gender'], dtype=float)
    groups = df.groupby('patient')

    def filter_group(group):
        # Find the index of the first row with a "1" in the binary column
        group_1 = group.loc[group['SepsisLabel'] == 1]

        if group_1.empty:
            # Find the row with the least number of nulls
            least_nulls_row = group.loc[group.iloc[:, 0] == group.isnull().sum(axis=1).idxmin()]

            # Calculate the row-wise average of all the other rows
            average_row = group.mean(skipna=True)

            # Fill null values in the least-nulls row with the average row values
            filled_row = least_nulls_row.fillna(average_row)
            return filled_row

        idx = group['SepsisLabel'].idxmax()
        # Loop through columns and fill NaN values
        for column in group.columns:
            if pd.isna(group.loc[idx, column]):
                first_idx = int(group.iloc[0].iloc[0])
                if idx == first_idx:
                    group.loc[idx, column] = group[column].mean()
                else:
                    for i in range(idx, first_idx, -1):
                        value = group.loc[i - 1, column]
                        if not np.isnan(value):
                            group.loc[idx, column] = value
                            break

        selected_row = group.loc[idx]

        # Return only the selected row, and drop the rest of the rows
        return selected_row.to_frame().T

    # Apply the custom function to each group, and concatenate the results
    filtered_df = pd.concat([filter_group(group) for _, group in groups])
    filtered_df = filtered_df.fillna(df.mean())
    return filtered_df[['HR', 'O2Sat', 'SBP', 'MAP', 'Age', 'HospAdmTime','ICULOS','Gender_0','Gender_1',
                        'SepsisLabel','patient']]

def main(path):
    data = create_df(path,'sec_test_sample',test=True)
    #_, df = get_original_data(load=False)
    #df = pd.read_csv("patient_df_test.csv")
    #data = preprocessing(df)

    #data = data.reset_index(drop=True)
    X_test = data.drop(['SepsisLabel', 'pid'], axis=1)
    with open('GBcf_new.pkl', 'rb') as f:
        GBcf = pickle.load(f)
    data['GBcf'] = GBcf.predict(X_test)
    prediction = data[['pid', 'GBcf']].rename(columns = {'pid': 'id', 'GBcf': 'prediction'})
    prediction.to_csv("prediction.csv", index=False)


if __name__=='__main__':
    #main(sys.argv[1])
    main('sec_test_sample')
