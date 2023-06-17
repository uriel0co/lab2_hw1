# Before using models, check:
#                   -there is no missing data
#                   -the data is normalized
#                   -categorical values are encoded
import sklearn as sk
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from preprocess import *
import pickle

########
def preprocessing(first=False):
    if first:
        print("Starting train set creation")
        train_df = create_df('/home/student/lab2_hw1/data/train', 'train_loaded', test=False)
        print("Ending train set creation")
        print("Starting test set creation")
        test_df = create_df('/home/student/lab2_hw1/data/test', 'test_loaded', test=True)
        print("Ending test set creation")
    else:
        train_df = pd.read_csv('train_loaded.csv')
        test_df = pd.read_csv('test_loaded.csv')

    r = 0.33
    train_df = train_resample(train_df, r)
    print("finished train and test set creation")
    return train_df, test_df

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
    #train_data = train_data.reset_index(drop=True)

    y_train = train_data['SepsisLabel']
    X_train = train_data.drop(['SepsisLabel', 'pid'], axis=1)

    # X_train = X_train.iloc[:, 2:]
    RForest = RandomForestClassifier(random_state=42)
    #grid_search_RF(X_train, y_train, RForest)
    #rand_search_RF(X_train, y_train, RForest)
    RForest.fit(X_train, y_train)


    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(X_train, y_train)

    # logReg = LogisticRegression(random_state=42,solver='lbfgs', max_iter=1000)
    # logReg.fit(X_train, y_train)

    # try gradient boosting classifier?

    GBcf = GradientBoostingClassifier(random_state=42)
    #rand_search_GB(X_train, y_train, GBcf)
    GBcf.fit(X_train, y_train)


    # We need to add grid search!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return RForest, GBcf, KNN


def classify_test(test_data, RForest, GBcf, KNN):
    # evaluate the model
    #test_data = test_data.reset_index(drop=True)
    y_test = test_data['SepsisLabel']
    X_test = test_data.drop(['SepsisLabel', 'pid'], axis=1)
    # X_test = X_test.iloc[:, 2:]
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

    df_predictions['KNN'] = KNN.predict(X_test)
    f1_knn = sk.metrics.f1_score(y_test, df_predictions['KNN'])
    print('The F1 score of the KNN on the test set is ', f1_knn)

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


def main():
    train_df, test_df = preprocessing(first=False)
    # #read train data
    # path_train = 'data/train'
    # load_data(path_train)

    # #read test data
    # path_test = 'data/test'
    # load_data(path_test)

    # #preProcessing

    # df_train = pd.read_csv("/home/student/094295_hw1/patient_df_train_1.csv")
    # df_test = pd.read_csv("/home/student/094295_hw1/patient_df_test_1.csv")
    # train_data = preprocessing(df_train)
    # test_data = preprocessing(df_test)
    # train_data.to_csv('/home/student/094295_hw1/train_processed.csv')
    # test_data.to_csv('/home/student/094295_hw1/test_processed.csv')
    # print("finished pre-processing")

    # train_processed = pd.read_csv("/home/student/094295_hw1/train_processed_thresh_0.75.csv")
    # test_processed = pd.read_csv("/home/student/094295_hw1/test_processed_thresh_0.75.csv")
    RForest, GBcf, KNN = train_models(train_df)
    filename = 'RForest_new.pkl'

    # Save the model to a file
    with open(filename, 'wb') as file:
        pickle.dump(RForest, file)
    classify_test(test_df, RForest, GBcf, KNN)


main()
