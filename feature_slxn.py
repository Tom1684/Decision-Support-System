import pandas as pd
import numpy as np
import pygmo as pg
import random
import functools
import math as m
import copy
from numpy import genfromtxt

# Plotting libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# sklearn modules
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, roc_curve
from sklearn.metrics import RocCurveDisplay, precision_recall_curve, recall_score
from sklearn.metrics import make_scorer, precision_score, classification_report
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn import datasets

from sklearn.feature_selection import SelectFromModel
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from scipy.stats import spearmanr

import sklearn_relief as relief

# For ignoring warnings about bad classifiers - it's the nature of the algorithm to come across these
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import json

# Specify the path to the JSON file
json_file_path = "user_input.json"

# Open and read the JSON file
with open(json_file_path, "r") as json_file:
    json_data = json.load(json_file)

plt.rcParams['figure.figsize'] = (8,6)
plt.rcParams['figure.dpi'] = 80
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)

###################################### USER INPUT ######################################

X_train = pd.read_csv(f'X_train.csv')
X_test = pd.read_csv(f'X_test.csv')
y_train = genfromtxt(f'y_train.csv', delimiter=',').astype(int)
y_test = genfromtxt(f'y_test.csv', delimiter=',').astype(int)

###################################### USER INPUT ######################################

# Number of features in model
total_features = int(X_train.shape[1])

# Feature names
feat_names = np.asarray(list(X_train.columns))

# Make list of indices of all features
feat_indices = np.asarray(list(range(total_features)))

# Make dictionary of indices of all features, for easy lookup
feat_dictionary = {}

for i, feature in enumerate(feat_names):
    
    feat_dictionary[feature] = i

# Declare set of learning algorithms
model_dict = {
    "Support Vector Classifier": SVC(random_state=42,kernel='linear',max_iter=int(1e4),probability=True),
    "Logistic Regression": LogisticRegression(random_state=42,penalty='l1', solver = 'liblinear'),
    "Decision Trees": DecisionTreeClassifier(random_state=42),
    "XGBoost": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42),
}

# In order for multi-thread processing functionalities to work, a main function must be declared
def main():

    # Declare learning algorithm choice
    model_name = json_data['model']

    random.seed(42)

    ### Functions for Ten Feature Selection Methods ###
    def recursive_feature_elimination(model_name,k):
        '''
        Employs the recursive feature elemination scikit-learn module to get the top k features in the training matrix
        '''

        # Selecting the best important features according to Logistic Regression
        rfe_selector = RFE(estimator=model_dict[model_name], n_features_to_select = k, step = 1)

        rfe_selector.fit(X_train, y_train)

        return feat_indices[rfe_selector.get_support().tolist()]

    def select_from_model(model_name, k):
        '''
        For a given learning algorithm, get features that pass the mean threshold in the training matrix
        '''

        sfm_selector = SelectFromModel(estimator=model_dict[model_name], max_features=k)

        sfm_selector.fit(X_train, y_train)
        
        features = np.asarray(feat_indices[sfm_selector.get_support().tolist()])
        
        # For handling missing values
        if len(features) < k:
            
            # Get values not in the list
            missing = feat_indices[~np.in1d(feat_indices, features)]
            
            # Get the number of missing features you need to add to output k
            num_missing = k - len(features)
            
            return np.append(features, missing[:num_missing])
        
        # Add feature indices to list
        return features

    def selectkbest_ANOVA(k):
        '''
        Use analysis of variance to return k best features in the training matrix
        '''
    
        ANOVA_selector = SelectKBest(k=k)

        ANOVA_selector.fit(X_train, y_train)
        
        return feat_indices[ANOVA_selector.get_support().tolist()]

    def selectkbest_mutual_inf(k):
        '''
        Use the muutal information of features to get the relative importance of k best features in the training matrix
        '''

        
        MI_selector = SelectKBest(mutual_info_classif, k=k)

        MI_selector.fit(X_train, y_train)
        
        return feat_indices[MI_selector.get_support().tolist()]

    def selectkbest_chi2(k):
        '''
        Rank features according to chi-2 values - cannot accept negative feature values
        '''

        chi2_selector = SelectKBest(score_func=chi2, k=k)

        chi2_selector.fit(X_train, y_train)
        
        return np.asarray(feat_indices[chi2_selector.get_support().tolist()])

    def pearson_corr_select(k):
        '''
        Rank features according to pearson correlation co-efficient
        '''
        
        # Get correlations
        correlation_matrix = np.corrcoef(X_train, y_train, rowvar=False)
        
        # Extract the correlation coefficients of each feature with the target (last column)
        correlations_with_target = correlation_matrix[:-1, -1]
        
        # Sort the indices of the features based on the correlation coefficients in descending order
        sorted_indices = np.argsort(correlations_with_target)[::-1]
        
        return sorted_indices[:k]

    def spearman_feature_importance(X, y):
        
        X = X.values
        
        num_features = X.shape[1]
        feature_importance_scores = np.zeros(num_features)
        
        for i in range(num_features):
            feature_scores, _ = spearmanr(X[:, i], y)
            feature_importance_scores[i] = abs(feature_scores)
                    
        return np.argsort(feature_importance_scores)

    def mean_abs_deviation(k):
        '''
        Get k best features using mean absolute deviation
        '''
        # Calculate MAD for each feature
        mean_abs_diff = np.sum(np.abs(X_train - np.mean(X_train,axis=0)),axis=0)/X_train.shape[0]

        # Sort by decreasing values of MAD, get the k best features
        l = mean_abs_diff.to_frame().sort_values(by=[0], ascending=False)
        k_best_strings = list(l.index[:k])

        return np.asarray([feat_dictionary[el] for el in k_best_strings])

    def fisher_score_func(k):
        '''
        Select best k features using fisher score
        '''
        # Use the Fisher algorithm to get a score for each metric feature
        ranks = fisher_score.fisher_score(X_train.values,y_train)
        
        return np.asarray(ranks[:k])

    def Relief(k):
        '''
        Use relief algorithm to get feature rankings for k top features
        '''

        r = relief.Relief(
            n_features=total_features
        ) 

        my_transformed_matrix = r.fit_transform(
            X_train.values,
            y_train
        )

        # Higher weights are better
        weights = np.zeros((total_features ,2))

        weights[:,0], weights[:,1] = np.asarray(r.w_), feat_indices

        weights_sort = weights[weights[:, 0].argsort()[::-1]]

        return np.asarray(weights_sort[:k,1].astype(int))
        

    ## Building a Borda voting system

    # Methods that return ranked lists 

    # For example, the relief algorithm does return a score for each feature, allowing for very easy ranking. 
    # But some selection methods return only the k features they selected in numerical order. 
    # So in order to get a ranking from every one, this process is used

    def get_ranked_features():
        '''
        This function returns a ranked list of features from all the methods listed in it
        '''
        
        features = []
        k = total_features
        
        features.append(spearman_feature_importance(X_train,y_train))
        features.append(pearson_corr_select(k))
        features.append(mean_abs_deviation(k))
        features.append(fisher_score_func(k))
        features.append(Relief(k))
        
        return np.asarray(features)

    # Methods that don't return ranked lists

    def get_features_for_k(k, model_name):
        '''
        Used recursively with incremented k to build a ranking list for feature selection methods that don't
        return featuers in order they are ranked in
        '''
        
        features=[]
        
        try:
            features.append(recursive_feature_elimination(model_name,k))
            features.append(select_from_model(model_name, k))
            features.append(selectkbest_ANOVA(k))
            features.append(selectkbest_mutual_inf(k))
            features.append(selectkbest_chi2(k))
        
        # If there are negative values in the data-set, chi-2 cannot be used
        except ValueError:
            features.append(recursive_feature_elimination(model_name,k))
            features.append(select_from_model(model_name, k))
            features.append(selectkbest_ANOVA(k))
            features.append(selectkbest_mutual_inf(k))
        
        return np.asarray(features)

    def add_unique_elements_column(matrix1, matrix2):
        '''
        Adds unique feature indices found with each iteration of k to the rankings matrix
        '''
        # Initialize an empty array to store the new column
        new_column = np.empty((matrix1.shape[0], 1), dtype=matrix1.dtype)

        # Iterate through the rows of both matrices
        for i in range(matrix1.shape[0]):
            # Find the unique elements in the corresponding row of matrix2
            unique_elements = np.setdiff1d(matrix2[i], matrix1[i])

            # Add the first unique element to the new column
            new_column[i, 0] = unique_elements[0]

        # Concatenate the new column to matrix1
        result_matrix = np.hstack((matrix1, new_column))

        return result_matrix

    def get_feature_matrix():
        '''
        Calls all function necessary to build feature matrix
        '''
        
        ranked = get_ranked_features()
        
        output = get_features_for_k(1, model_name)
        
        for k in range(2,total_features+1):
        
            matrix2 = get_features_for_k(k, model_name)

            output = add_unique_elements_column(output, matrix2)

        return np.vstack((ranked, output))

    def calculate_borda_scores(rankings_matrix, weights=None):
        '''
        I: ranking matrix from get_feature_matrix
        I (optional) weights, array of floats from zero to one, represent the voting weight that particular feature selection method has
        O: borda scores for all features in the data matrix
        '''
        num_features = len(rankings_matrix[0])
        num_methods = len(rankings_matrix)
        borda_scores = np.zeros(num_features)

        if weights is None:
            weights = np.ones(num_methods)

        if len(weights) != num_methods:
            raise ValueError("The number of weights must be equal to the number of methods.")

        for i, method_rankings in enumerate(rankings_matrix):
            try:
                # Calculate the scores for each feature based on its ranking in the method and apply weights
                scores = weights[i] * (num_features - np.array(method_rankings))
                borda_scores += scores
            except ValueError as e:
                print("Error:", e)
                print("Ensure that all feature rankings have the same number of features.")
                return None

        return borda_scores
    
    # Compute rankings matrix for each feature selection method
    rankings_matrix = get_feature_matrix()

    # Calculate the Borda count scores for each feature
    borda_scores = calculate_borda_scores(rankings_matrix)

    def generate_borda_figure(borda_scores):
        '''
        generates bar graph PDF of borda scores, saves in main directory
        '''

        # Create a DataFrame to organize the results
        results_df = pd.DataFrame({
            "Feature Index": range(len(borda_scores)),
            "Borda Score": [int(score) for score in borda_scores],
            "Feature Name": [feat_names[i] for i in range(len(borda_scores))]
        })

        # Sort the DataFrame by 'Borda Score' in descending order
        df_sorted = results_df.sort_values(by='Borda Score', ascending=False)

        # Partition the features into 3 sets based on the sorted Borda scores
        num_features = len(df_sorted)
        num_features_per_set = num_features // 3
        set1 = df_sorted.iloc[:num_features_per_set]
        set2 = df_sorted.iloc[num_features_per_set:2*num_features_per_set]
        set3 = df_sorted.iloc[2*num_features_per_set:]

        # Create the bar plot with different colors for each set
        plt.figure(figsize=(12, 6))  # Adjust the figure size as per your requirement

        plt.bar(set1['Feature Name'], set1['Borda Score'], color='darkgreen', label='$\mathcal{B}$')
        plt.bar(set2['Feature Name'], set2['Borda Score'], color='cornflowerblue', label='$\mathcal{G}$')
        plt.bar(set3['Feature Name'], set3['Borda Score'], color='firebrick', label='$\mathcal{W}$')

        # Set x-axis labels with rotation for better visibility
        plt.xticks(rotation=45, ha='right')

        # Set axis labels and title
        plt.ylabel('Borda Score')

        # Add a legend
        plt.legend()

        # Save the plot
        plt.tight_layout()

        plt.savefig(f"borda_scores.pdf", bbox_inches='tight', format="pdf", dpi=1200)

    generate_borda_figure(borda_scores)

    def partition_borda_scores(borda_scores):
        '''
        Partitions feature set into three equal sets based on Borda scores
        '''
        # Get the feature indices sorted based on Borda scores (in descending order)
        sorted_indices = sorted(range(len(borda_scores)), key=lambda x: borda_scores[x], reverse=True)

        # Calculate the size of each set (best, good, worst) to achieve roughly equal partitions
        set_size = len(sorted_indices) // 3
        remaining = len(sorted_indices) % 3

        # Partition the indices into three sets
        best_features = sorted_indices[:set_size + remaining]
        good_features = sorted_indices[set_size + remaining: 2 * (set_size + remaining)]
        worst_features = sorted_indices[2 * (set_size + remaining):]

        psi = [best_features,good_features,worst_features]

        ordered = [np.unique(feat_set) for feat_set in psi]

        return ordered[0], ordered[1], ordered[2]

    # Partition the feature sets
    best_features, good_features, worst_features = partition_borda_scores(borda_scores)

    # Save as csv
    np.savetxt(f"best.csv", best_features, delimiter=",")
    np.savetxt(f"good.csv", good_features, delimiter=",")
    np.savetxt(f"worst.csv", worst_features, delimiter=",")

if __name__ == '__main__':
    main()

