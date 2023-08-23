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
from skfeature.function.similarity_based import fisher_score
from sklearn.impute import SimpleImputer
import sklearn_relief as relief
import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Matplotlib configuration
plt.rcParams['figure.figsize'] = (8,6)
plt.rcParams['figure.dpi'] = 80
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Streamlit configuration
st.set_page_config(page_title="Decision Support System")

# Title, instructions
st.title("Decision Support System")

st.write("---")

st.subheader("Description")
st.markdown("This application performs feature selection using two Multi-Objective Optimization Metaheuristic Algorithms on a user-provided Machine Learning data-set. If you don't have your own data-set, you can use the IBM churn data provided in the 'data' folder.")
st.markdown("The algorithms are designed specifically for customer churn prediction models but any binary classification data-set is accepted - multi-class classification sets will not work. The minimum number of features supported is nine.")
st.markdown("Note that all features should be non-negative, otherwise some feature selection methods used in the algorithm will be unavailable. If your data has negative values, the algorithms will still run but use nine feature selection methods instead of ten.")

st.subheader("Instructions")
st.markdown("- Save your testing and training data matrices, which must be **Pandas dataframes**, as X_train.csv and X_test.csv in this directory.")
st.markdown("- Save your response variables, which must be a **flat numpy arrays** delimited with a comma, as y_train.csv and y_test.csv in this directory.")
st.markdown("- Select a classification model scoring metric as well as the model itself below that is most appropriate for your data-set below.")
st.markdown("- There is a choice of two objectives: time taken to fit a classification model with the selected features and the number of features selected.")
st.markdown("- Iterations are set to ten by default - increasing these may provide better performance but at higher computational times.")

st.write("")

def reset_page():
    st.experimental_rerun()

############################## User Inputs ##############################

button_clicked = False

if st.button("Reset Page"):
    reset_page()

st.subheader("User Input")

options_scores = ['Select an option', 'Classification Accuracy', 'Specificity', 'Recall', 'Precision', 'F1 Score', 'ROC AUC', 'Matthew\'s Correlation'] 
selected_score = st.selectbox("Select a scoring metric for the algorithm to use:", options_scores)

options_clf = ['Select an option', 'Support Vector Classifier', 'Logistic Regression', 'Decision Trees' , 'XGBoost']
selected_clf = st.selectbox("Select a classification algorithm to use:", options_clf)

options_obj = ['Select an option', 'Time (s)', 'Number of Features']
selected_obj = st.selectbox("Select a second objective to run the algorithm with:", options_obj)

# Numerical user inputs
fnsga2_input = st.text_input("Number of iterations for the FNSGA-II", value=10)
spea2_input = st.text_input("Number of iterations for the SPEA-II", value=10)

# Convert user input to a number (if possible)
try:
    iters_fnga2, iters_spea2 = int(fnsga2_input), int(spea2_input)
except ValueError:
    iters_fnga2, iters_spea2 = None

# Only accept a valid integer as input
if iters_fnga2 is None or iters_spea2 is None:
    st.write("Invalid numerical input. Please enter a valid integer number.")


# Only a selected option as an input
if selected_score == 'Select an option' or selected_clf == 'Select an option' or selected_obj == 'Select an option':
    st.write("Please select a valid option before running.")

# Show run button only if user provides valid inputs 
else:

    # Button prompt
    st.write("Click the button to solve the multi-objective optimization problem with your selected options.")

    # Add a button
    button_clicked = st.button("Run")

# 
results_collected = False

if button_clicked:

    # Write user inputs as json instructions for the other python files to read
    json_data = {'scoring_metric': selected_score,
                'model': selected_clf,
                'objective': selected_obj,
                'iters_fnsga2': iters_fnga2,
                'iters_spea2': iters_spea2}

    with open(f'user_input.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    st.write('Running ten feature selection methods.')
    import feature_slxn
    feature_slxn.main()
    st.write('Running SPEA-II.')
    import SPEA2
    st.write('Running FNSGA-II.')
    import FNSGA2
    st.write('Completed.')

    # Import the combined non-dominated solutions from both algorithms - they are combined in the FNSGA2 workbook
    results = genfromtxt(f'FNSGA2.csv', delimiter=',')

    # Separate the objective values and the features into two variables
    data = results[:,:2]
    features = results[:,3:].astype(int)

    # For plotting
    x = data[:, 0]
    y = data[:, 1]

    results_collected = True

############################## User Inputs ##############################

############################## Functions for results ##############################

def show_barplot():
    ######## Bar plot of feature frequency ########

    # Count the frequency of each string using Counter
    string_counts = Counter(features_PF)

    # Extract unique strings and their frequencies
    sorted_counts = sorted(string_counts.items(), key=lambda item: item[1], reverse=False)
    unique_strings = [item[0] for item in sorted_counts]
    frequencies = [item[1] for item in sorted_counts]

    # Create a bar graph using Plotly
    bar_trace = go.Bar(y=unique_strings, x=frequencies, orientation='h', marker=dict(color='red'))
    bar_fig = go.Figure(data=[bar_trace])

    # Set axis labels and chart title, as well as tick settings
    bar_fig.update_layout(xaxis_title="Frequency")
    bar_fig.update_layout(bargap=0.1, margin=dict(t=30, b=10, l=200, r=10))
    bar_fig.update_yaxes(dtick=2)

    # Subheader
    st.subheader('Bar Graph of Feature Frequency Across All Pareto-Optimal Solutions')

    # Display the bar graph in Streamlit
    st.plotly_chart(bar_fig)

    # Explain bar graph
    st.write('Mouse over a bar on the graph to see the feature associated with that frequency.')

    ####### Bar plot of feature frequency ########

def show_scatterplot():
    ####### Scatter plot of objective space ########

    # Subtitle
    st.subheader('Objective Space Pareto Front Scatter Plot')

    st.write('Use the interactive tools to zoom into the area of interest.')

    # Create a scatter plot using Plotly
    scatter_fig = px.scatter(x=x, y=y)
    scatter_fig.update_layout(xaxis_title=selected_score, yaxis_title=selected_obj)

    # Grid settings for the 
    scatter_fig.update_xaxes(showgrid=True, gridwidth=0.1, gridcolor='lightgray')
    scatter_fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor='lightgray')

    scatter_fig.update_traces(marker=dict(size=10, color='red', opacity=0.8))
    # Display the scatter plot in Streamlit
    st.plotly_chart(scatter_fig)
    ####### Scatter plot of objective space ########

def select_features():

    ####### Choosing a feature and displaying its features ########

    st.subheader('Features Used For All Solutions')

    all_sols_tuples = [tuple([point[0],point[1]]) for point in data]

    for i, soln in enumerate(all_sols_tuples):

        if selected_obj == 'Time (s)':
            st.markdown(f'**({np.round(soln[0],3)}, {np.round(soln[1],3)})**')

        else:
            st.markdown(f'**{soln}**')

        st.write(f"{len(results_features[i])} Features Used")
        st.write(f'{results_features[i]}')

def show_all():

    show_barplot()
    show_scatterplot()
    select_features()

if results_collected:
    # Getting feature names for each point
    X_df = pd.read_csv(f'X_train.csv')
    feat_names = np.asarray(X_df.columns)

    results_features = []
    features_PF = []

    for point in features:
        
        # Get the names of features used by classifier to get this solution
        indices = np.where(point == 1)[0]
        names = feat_names[indices]

        # Getting the strings of features associated with a solution
        results_features.append(names)

        # For tallying up frequency of features in the entire PF
        for feature in names:
            features_PF.append(feature)

    show_all()


############################## Visualising results ##############################

# Footer
st.write("---")
st.write("Created by Tom Murarik for a University of Edinburgh MSc Dissertation in Collaboration with Vodafone Data Analytics")
