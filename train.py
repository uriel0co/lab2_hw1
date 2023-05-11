# Before using models, check:
#                   -there is no missing data
#                   -the data is normalized
#                   -categorical values are encoded
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



########
def load_data(path):
    print(f'_{path.split("/")[1]}')
    files = os.listdir(path)
    to_csv = []
    for file in range(0, len(files)):
        path_name = os.path.abspath(os.path.join(path, files[file]))
        to_csv.append(path_name)
    flag = True
    for i, one_file in enumerate(to_csv):
        print(i)
        with open(one_file, "r", encoding="utf-8") as f:
            if flag == True:
                patient_df = pd.read_csv(one_file, sep='|')
                patient_df['patient'] = files[i].split('patient')[1]
                flag = False
            else:
                new_patient_df = pd.read_csv(one_file, sep='|')
                new_patient_df['patient'] = files[i].split('patient')[1]
                patient_df = pd.concat([patient_df, new_patient_df], ignore_index=True)
    str_save = f'patient_df_{path.split("/")[1]}_1.csv'
    return patient_df.to_csv(str_save)

def grid_search_RF(X_train, y_train, RForest):
    # Training the models
    param_grid_RF = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    # Create a GridSearchCV object and fit it on the training data
    grid_search_RF = GridSearchCV(RForest, param_grid=param_grid_RF, cv=5, n_jobs=-1, scoring='f1')
    grid_search_RF.fit(X_train, y_train)

    # Print the best hyperparameters and the corresponding score
    print("Best hyperparameters:", grid_search_RF.best_params_)
    print("Best F1 score:", grid_search_RF.best_score_)



def grid_search_GB(X_train, y_train, GBcf):
    param_grid_GB = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [0.5, 0.8, 1.0],
        'subsample': [0.5, 0.8, 1.0]
    }

    # Create a GridSearchCV object and fit it on the training data
    grid_search_GB = GridSearchCV(GBcf, param_grid=param_grid_GB, cv=5, n_jobs=-1, scoring='f1')
    grid_search_GB.fit(X_train, y_train)

    # Print the best hyperparameters and the corresponding score
    print("Best hyperparameters:", grid_search_GB.best_params_)
    print("Best F1 score:", grid_search_GB.best_score_)


def rand_search_GB(X_train, y_train, GBcf):
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [0.5, 0.8, 1.0],
        'subsample': [0.5, 0.8, 1.0]
    }

    # Create the RandomizedSearchCV object
    n_iter_search = 20  # Number of parameter settings that are sampled
    random_search = RandomizedSearchCV(
        GBcf, param_distributions=param_grid, n_iter=n_iter_search, cv=5)

    # Fit the random search object to the data
    random_search.fit(X_train, y_train)

    # Print the best parameters found during the random search
    print(random_search.best_params_)

def rand_search_RF(X_train, y_train, RForest):
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    # Create the RandomizedSearchCV object
    n_iter_search = 20  # Number of parameter settings that are sampled
    random_search = RandomizedSearchCV(
        RForest, param_distributions=param_grid, n_iter=n_iter_search, cv=5)

    # Fit the random search object to the data
    random_search.fit(X_train, y_train)

    # Print the best parameters found during the random search
    print(random_search.best_params_)


def train_models(train_data):
    train_data = train_data.reset_index(drop=True)

    y_train = train_data['SepsisLabel']
    X_train = train_data.drop(['SepsisLabel', 'patient'], axis=1)

    X_train = X_train.iloc[:, 2:]
    RForest = RandomForestClassifier(random_state=42)
    #grid_search_RF(X_train, y_train, RForest)
    #rand_search_RF(X_train, y_train, RForest)
    RForest.fit(X_train, y_train)


    # KNN = KNeighborsClassifier(n_neighbors=3)
    # KNN.fit(X_train, y_train)

    #logReg = LogisticRegression(random_state=0, solver='lbfgs', max_iter=100, C=200)
    #logReg.fit(X_train, y_train)

    # try gradient boosting classifier?

    GBcf = GradientBoostingClassifier(random_state=42)
    #rand_search_GB(X_train, y_train, GBcf)
    GBcf.fit(X_train, y_train)


    # We need to add grid search!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return RForest, GBcf


def classify_test(test_data, RForest, GBcf):
    # evaluate the model
    test_data = test_data.reset_index(drop=True)
    y_test = test_data['SepsisLabel']
    X_test = test_data.drop(['SepsisLabel', 'patient'], axis=1)
    X_test = X_test.iloc[:, 2:]
    df_predictions = pd.DataFrame({})
    print(X_test.shape)
    # predict Labels for test set
    df_predictions['RForest'] = RForest.predict(X_test)
    f1_rf = sk.metrics.f1_score(y_test, df_predictions['RForest'])
    print('The F1 score of the random forest on the test set is ', f1_rf)
    print('recall ', sk.metrics.recall_score(y_test, RForest.predict(X_test)))
    print('precision ', sk.metrics.precision_score(y_test, RForest.predict(X_test)))

    # predict Labels for test set
    df_predictions['GBcf'] = GBcf.predict(X_test)
    f1_rf = sk.metrics.f1_score(y_test, df_predictions['GBcf'])
    print('The F1 score of the GB on the test set is ', f1_rf)
    print('recall ', sk.metrics.recall_score(y_test, GBcf.predict(X_test)))
    print('precision ', sk.metrics.precision_score(y_test, GBcf.predict(X_test)))

    # df_predictions['KNN'] = KNN.predict(X_test)
    # f1_knn = sk.metrics.f1_score(y_test, df_predictions['KNN'])
    # print('The F1 score of the KNN on the test set is ', f1_knn)

    # df_predictions['logReg'] = logReg.predict(X_test)
    # f1_lr = sk.metrics.f1_score(y_test, df_predictions['logReg'])
    # print('The F1 score of the logistic regression model on the test set is ', f1_lr)
    # print('recall ', sk.metrics.recall_score(y_test, logReg.predict(X_test)))
    # print('precision ', sk.metrics.precision_score(y_test, logReg.predict(X_test)))
    #

    # Choose the label that is the output of at list 2 of the models
    # for N=2k-1
    #df_predictions['finalLabel'] = df_predictions.mode(axis=1)
    # for N=2k
    df_predictions['finalLabel'] = df_predictions.mode(axis=1).iloc[:, 0]
    df_predictions.to_csv('/home/student/094295_hw1/predictions.csv')
    # calculate f1
    f1 = sk.metrics.f1_score(y_test, df_predictions['finalLabel'])
    print("The F1 score of the combined model on the test set is ", f1)


def preprocessing(df):
    df['patient'] = df['patient'].apply(lambda x: int(x.split("_")[1].split(".")[0]))
    df = df.sort_values(['patient', 'ICULOS'])
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
    return filtered_df


def main():
    #read train data
    path_train = 'data/train'
    load_data(path_train)

    #read test data
    path_test = 'data/test'
    load_data(path_test)

    #preProcessing

    df_train = pd.read_csv("/home/student/094295_hw1/patient_df_train_1.csv")
    df_test = pd.read_csv("/home/student/094295_hw1/patient_df_test_1.csv")
    train_data = preprocessing(df_train)
    test_data = preprocessing(df_test)
    train_data.to_csv('/home/student/094295_hw1/train_processed.csv')
    test_data.to_csv('/home/student/094295_hw1/test_processed.csv')
    print("finished pre-processing")

    train_processed = pd.read_csv("/home/student/094295_hw1/train_processed_thresh_0.75.csv")
    test_processed = pd.read_csv("/home/student/094295_hw1/test_processed_thresh_0.75.csv")
    RForest, GBcf = train_models(train_processed)
    classify_test(test_processed, RForest, GBcf)


main()
