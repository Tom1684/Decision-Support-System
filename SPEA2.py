import pandas as pd
import numpy as np
import pygmo as pg
import random
import functools
import math as m
import copy
from numpy import genfromtxt

# Geometry modules
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Plotting libraries
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as patches 
import seaborn as sns

# Misc
import time 
from time import process_time
from scipy.spatial import distance_matrix
from scipy.spatial import ConvexHull
import secrets

# sklearn modules
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, roc_curve
from sklearn.metrics import RocCurveDisplay, precision_recall_curve, recall_score
from sklearn.metrics import make_scorer, precision_score, classification_report
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegressionCV

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 

from deap import algorithms, base, creator, tools


# For ignoring warnings about bad classifiers - it's the nature of the algorithm to come across these
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

# Matplotlib configuration
plt.rcParams['figure.figsize'] = (8,6)
plt.rcParams['figure.dpi'] = 80
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import json

# Specify the path to the JSON file
json_file_path = "user_input.json"

# Open and read the JSON file
with open(json_file_path, "r") as json_file:
    json_data = json.load(json_file)

X_df = pd.read_csv(f'X_train.csv')

X_train = pd.read_csv(f'X_train.csv').values
X_test = pd.read_csv(f'X_test.csv').values
y_train = genfromtxt(f'y_train.csv', delimiter=',').astype(int)
y_test = genfromtxt(f'y_test.csv', delimiter=',').astype(int)

# Declare standard scaler
scaler = StandardScaler()
# Transform training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Number of features in model
total_features = int(X_train.shape[1])

# Feature names
feat_names = np.asarray(list(X_df.columns))

# Choice of Classifier
model_string = json_data['model']
objective = json_data['objective']
scoring_metric = json_data['scoring_metric']
num_iters = json_data['iters_spea2']
constraint_features = 0

# Set of learning algorithms
model_dict = {
    "Support Vector Classifier": SVC(random_state=42,kernel='linear',max_iter=int(1e4),probability=True),
    "Logistic Regression": LogisticRegression(random_state=42,penalty='l1', solver = 'liblinear'),
    "Decision Trees": DecisionTreeClassifier(random_state=42),
    "XGBoost": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42),
}

clf_ = model_dict[model_string]


#### Inputs end, functions start

def MatthewsCorrelation(confmat):
    '''
    Calculates the Matthews Correlation coefficient based on confusion matrix elements.
    '''
    
    # Ignore NaN warnings
    np.seterr(divide='ignore', invalid='ignore')
    
    # Decompose confusion mattrix elements
    [[tn, fp],[fn, tp]] = confmat
    
    # Calculate Matthew's Correlation Coefficient
    mc_coef = ((tp*tn) - (fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    
    # Catch NaNs, return them as zero
    if np.isnan(mc_coef):
        return 0
    
    return mc_coef

def get_score(predictor, pipeline, feature_indices):
    
    y_pred = pipeline.predict_proba(X_test[:,feature_indices])[:, 1]
    AUC_score = roc_auc_score(y_test, y_pred)
              
    dictionary = {"Classification Accuracy": accuracy_score(y_test, predictor),
                  "Specificity": recall_score(y_test, predictor, pos_label =0),
                  "Recall": recall_score(y_test, predictor),
                  "Precision": precision_score(y_test, predictor, zero_division=0.0),
                  "F1 Score": f1_score(y_test, predictor, zero_division=0.0),
                  "ROC AUC": AUC_score,
                  "Matthew\'s Correlation": MatthewsCorrelation(confusion_matrix(y_true = y_test, y_pred=predictor))}
    
    return dictionary[scoring_metric]

################################# SPEA-II ########################################

# Define the evaluation function
def evaluate_SPEA2(individual):
    """
    Evaluates an individual by first selecting the features that are turned on in the individual,
    then applying feature selection to the training and test data, and finally training a classifier and calculating the accuracy.
    The function returns the number of selected features and the negative accuracy.
    """

    selected_features = [i for i, val in enumerate(individual) if val]
    if not selected_features:
        return 0, 0,  # Return 0 accuracy and 0 features if none are selected
    
    n_feats = len(np.where(np.asarray(individual) == 1)[0].tolist())

    # Apply feature selection
    selected_X_train = X_train[:, selected_features]
    selected_X_test = X_test[:, selected_features]

    # Train a classifier
    start = process_time()
    model = clf_.fit(selected_X_train, y_train)
    end = process_time()

    y_pred = model.predict(selected_X_test)

    accuracy = get_score(y_pred, clf_, selected_features)
        
    if objective == 'Time (s)':
        obj2 = end-start
    else:
        obj2 = n_feats

    # Feature cardinality constraint
    if n_feats < constraint_features and objective == 'Time (s)':
        obj2 = 1.2*obj2
    
    if n_feats < constraint_features and objective == 'Number of Features':
        obj2 = m.ceil(1.2*obj2)
        
    # Return negative accuracy to maximize it
    return -accuracy, obj2

# DEAP initialization
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=total_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selSPEA2) # this selection operator makes this an SPEA2
toolbox.register("evaluate", evaluate_SPEA2)


# Set up and run the algorithm
def main():
    """
    Sets up and runs the SPEA-2 algorithm.
    It first creates a random population of individuals,
    then uses the `eaMuPlusLambda()` algorithm to evolve the population over 50 generations.
    The algorithm uses the `cxTwoPoint()` and `mutFlipBit()` genetic operators to create new individuals,
    and the `selSPEA2()` selection operator to select individuals for the next generation.

    The main function then displays the Pareto front solutions, which are the individuals that have the best trade-off between the number of selected features and the accuracy.
    """

    random.seed(42)
    
    pop = toolbox.population(n=50)
    hof = tools.ParetoFront()

    algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.3, ngen=num_iters, stats=None, halloffame=hof, verbose=True)

    return hof

# Hall of fame is a functionality in DEAP that returns the Pareto fittest individuals
hof = main()

spea2_nondom_sols = np.zeros((len(hof),3+total_features))

# Write the Pareto front solutions into a matrix
for i, ind in enumerate(hof):
    
    score = -ind.fitness.values[0]
    
    obj2 = ind.fitness.values[1]
    
    # Write in the same universal results matrix format
    spea2_nondom_sols[i,:3] = (score, obj2, 1-score)
    
    spea2_nondom_sols[i,3:] = ind
    
# Algorithm sometimes returns zeros as results, this gets rid of them
for i, point in enumerate(spea2_nondom_sols):

    if point[0] <=0:
        spea2_nondom_sols = np.delete(spea2_nondom_sols, i, 0)
        
np.savetxt(f"SPEA2.csv", spea2_nondom_sols, delimiter=",")

################################## SPEA-II ########################################