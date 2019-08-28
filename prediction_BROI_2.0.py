"""
prediction_BROI version 2
by Imam Widodo

TODO:
 - Further optimize RandomForest
 - Implement way to delete duplicates before training - creates better improvement?

Creates regression models that return ROI based on: "brand", "program", "brandBucketCategory"
"""

import json
import pandas as pd
from pandas.api.types import is_string_dtype
import sklearn
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.base import clone
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import csv

""" 
WARNING :
 - instead of string "prediction/Predictive_Model", use following for actual deployment:
        os.path.dirname(os.path.realpath(__file__))
 - string is used purely for development compatibility reasons
"""
__location__ = ""
# os.path.dirname(os.path.realpath(__file__))
# "prediction/Predictive_Model"
# os.path.dirname(os.path.realpath(__file__))


"""
GLOBAL DATAFRAMES - keep UNMODIFIED: *only modified at beginning to remove outliers
    - df_all: dataframe of all brands (no numbers)
    - df_data: dataframe of brands with ROI data
"""
df_all = pd.DataFrame()
df_data = pd.DataFrame()


"""
FUNCTION & HELPER FUNCTION
 - import_data() - Imports data from .json files, defines 2 dataframes
 - remove_outliers(df_to_modify) - Cleans up dataframes of:
    - Outliers from df_data
    - Errant string spaces in columns of string type
"""
def import_data(roi_name, no_roi_name):
    global df_data
    global df_all

    filename = os.path.join(__location__, no_roi_name)
    with open(filename, 'r') as json_file:
        roi = json_file.read()
        data = json.loads(roi)
        df_all_raw = pd.io.json.json_normalize(data)
    json_file.close()
    filename = os.path.join(__location__, roi_name)
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        df_data_raw = pd.io.json.json_normalize(data)
    json_file.close()

    df_data = remove_outliers(df_data_raw)
    df_all = remove_outliers(df_all_raw)

def remove_outliers(df_to_modify):
    df_to_modify = df_to_modify.copy()

    if "date" in df_to_modify:
        df_to_modify[['year','quarter']] = df_to_modify['date'].str.split('-',expand=True)

    #Removes spaces from brandBucketCategory [omittable]
    for col in df_to_modify:
        if is_string_dtype(df_to_modify[col]):
            df_to_modify[col] = df_to_modify[col].str.strip()

    #Remove outliers in range [-2, 9] - reconsider?
    if "brandROI" in df_to_modify:
        indices_to_drop = df_to_modify[(df_to_modify['brandROI'] < -2.0) | (df_to_modify['brandROI'] > 9.0)].index
        df_to_modify.drop(indices_to_drop, inplace=True)
    
    return df_to_modify


"""
FUNCTIONs - Clean up data by:
    - keep_cols(cols_to_keep, df_to_modify): Drop columns not specified to keep
    Call after column removal for efficiency: (before just does extra that will be removed)
    - intersection_df_data(): Modify df_data to only contain data in df_all
    - intersection_df_all(): Define df_no_data to only have brands in df_all without data
"""
def keep_cols(cols_to_keep, df_to_mod):
    #Drops list of nonneeded columns
    df_to_modify = df_to_mod.copy()
    columns = list(df_to_modify)
    cols_to_drop = list(set(columns) - set(cols_to_keep))

    #Print warning if column mismatch between result and input cols_to_keep
    if(len(list(set(columns) - set(cols_to_drop))) < len(cols_to_keep)):
        print("WARNING - The following columns are not in the dataframe: "
                + " ".join(list(set(cols_to_keep) - set(columns))) )

    df_to_modify.drop(cols_to_drop, axis=1, inplace=True) 
    #df_to_modify = df_to_modify.drop_duplicates()

    return df_to_modify

def intersection(df_to_keep, df_to_intersect, noROI=False): #TODO: keeps intersection - EXCLUSIVE/LIMITED DISTRO
    df_keep = df_to_keep.copy()
    df_intersect = df_to_intersect.copy()

    columns_1 = list(df_keep)
    columns_2 = list(df_intersect)
    columns_intersect = [col for col in columns_1 if col in columns_2]
    
    intersection = pd.merge(df_keep, df_intersect, how="inner", on=columns_intersect)

    if noROI:
        if "brandROI" in intersection.columns:
            pure = list(set(intersection.columns) - set(columns_intersect))
            intersection.drop(pure, axis=1, inplace=True)
            intersection = intersection.drop_duplicates()

    return intersection

def difference_df_all(df_all_analog): 
    df_difference = pd.DataFrame()
    df_all_analog = df_all_analog.copy()
    columns_df_all = list(df_all_analog)
    
    df_data_analog = df_data[columns_df_all].copy()
    df_data_analog.drop_duplicates(inplace=True)
    columns_df_data = list(df_data_analog)

    #Get symmetric difference in rows
    if (set(columns_df_all) == set(columns_df_data)):
        comb = pd.concat([df_all_analog, df_data_analog]).reset_index(drop=True) #concat and re-index
        df_gpby = comb.groupby(list(columns_df_all))
        unique_idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1]
        df_difference = comb.reindex(unique_idx)
    else:
        raise Exception("The difference between df_all and df_data can't be computed")
    
    return df_difference
    

"""
FUNCTION: Construct indicator dataframe from inputted dataframe
    - For input into regression algorithm
"""
def indicatorer(df_input):
    df_indicator = df_input.copy()
    columns = list(set(list(df_indicator)) - set(["brandROI"])) #Don't remove brandROI

    for col in columns:
        temp = pd.get_dummies(df_indicator[col])
        df_indicator.drop([col], axis=1, inplace=True)
        df_indicator = pd.concat([df_indicator, temp], axis=1)

    return df_indicator


"""
REGRESSION MODELS & Random Forest Optimizers
"""
def ridge_model():
    """Params"""
    alpha = 1 #Original: alpha=1  
    """******"""  
    model = Ridge(alpha = alpha) 
    return model

def linear_model():
    model = LinearRegression()
    return model

def randomForest_model():
    """Params"""
    n_estimators = 100 #100 old 
    max_depth    = 35 #35 old
    random_state = 0
    criterion = "mse"
    bootstrap = True #True old
    min_samples_split = 20 #20 old
    min_samples_leaf = 1 #1 or 2 old
    """******"""  
    model = RandomForestRegressor(criterion = criterion, n_estimators = n_estimators, 
        max_depth = max_depth, random_state = random_state, bootstrap=bootstrap, 
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    return model

def optimizedRF_model(): #Maybe better than original?
    """Params"""
    n_estimators = 100
    max_depth    = 70
    min_samples_split = 10 #10
    min_samples_leaf = 2
    max_features = "sqrt"
    criterion = "mse"
    bootstrap = True
    """******"""  
    model = RandomForestRegressor(criterion = criterion, n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split,
    min_samples_leaf = min_samples_leaf, max_features = max_features, bootstrap = bootstrap)
    return model

def extraForest_model():
    """Params"""
    n_estimators = 100 
    max_depth    = 35
    random_state = 0
    criterion = "mse"
    bootstrap = True #True
    min_samples_split = 20 #20
    min_samples_leaf = 1 #1 or 2
    """******"""  
    model = ExtraTreesRegressor(criterion = criterion, n_estimators = n_estimators, 
        max_depth = max_depth, random_state = random_state, bootstrap=bootstrap, 
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    return model

#optimizer for random forest model
def randomForest_optimizer(df_to_train_on): #Hyperparameter Tuning: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    data = df_to_train_on.copy()

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)] #[100, 150, 200, 500]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] #[25, 35, 45, 55]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    
    # Fit the random search model
    #-----------------------------------------
    min_max_scaler = preprocessing.MinMaxScaler()
    data['Norm_brandROI'] = min_max_scaler.fit_transform(data[['brandROI']]) #requires 2d array for some reason

    """K Fold splitter: Splits data into training and test sets"""
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    #kf = RepeatedKFold(n_splits=5, n_repeats=2)
    split = kf.split(data) #USE ITERATION https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f
    result = next(split, None)
    train = data.iloc[result[0]].copy() 
    test = data.iloc[result[1]].copy()
    Y_train = train.Norm_brandROI
    Y_test  = test.Norm_brandROI
    Y_train = pd.DataFrame(Y_train)
    Y_test  = pd.DataFrame(Y_test)
    X_train = train
    X_test  = test
    X_train.drop(['brandROI'], axis=1, inplace=True) #WARNING - this modifies df train
    X_test.drop(['brandROI'], axis=1, inplace=True)
    X_train.drop(['Norm_brandROI'], axis=1, inplace=True)
    X_test.drop(['Norm_brandROI'], axis=1, inplace=True)
    rf_random.fit(X_train, Y_train.values.ravel()) #Creates multilinear regression model

    #result = {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 90, 'bootstrap': True}
    print(rf_random.best_params_)


"""
FUNCTION: Regression Trainer
    1) Split into train and test sets
    2) Run inputted regression
    3) Output trained_model and scaler used for normalization
    - Takes input of indicator df as data to train on
    - Outputs scaler for reverting purposes
"""
def trainer(model_type, df_to_train_on):
    """Normalizes 'brandROI' to 'Norm_brandROI' for optimization purposes
    (analog of LogbrandROI in old code)""" 
    data = df_to_train_on.copy()
    #data['Norm_brandROI'] = data['brandROI']
    min_max_scaler = preprocessing.MinMaxScaler()
    data['Norm_brandROI'] = min_max_scaler.fit_transform(data[['brandROI']]) #requires 2d array for some reason

    """K Fold splitter: Splits data into training and test sets"""
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    #kf = RepeatedKFold(n_splits=5, n_repeats=2)
    split = kf.split(data) #USE ITERATION https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f
    result = next(split, None)

    brand_model = clone(model_type) #type of model
    test_scores = []
    train_scores = []

    """Iterate and train over all K-Fold Splits"""
    while result is not None:
        train = data.iloc[result[0]].copy() 
        test = data.iloc[result[1]].copy()

        # Target values for X
        # Defined before X since X modifies train
        Y_train = train.Norm_brandROI
        Y_test  = test.Norm_brandROI
        Y_train = pd.DataFrame(Y_train)
        Y_test  = pd.DataFrame(Y_test)

        # Sparse matrix of Training Data
        X_train = train
        X_test  = test
        X_train.drop(['brandROI'], axis=1, inplace=True) #WARNING - this modifies df train
        X_test.drop(['brandROI'], axis=1, inplace=True)
        X_train.drop(['Norm_brandROI'], axis=1, inplace=True)
        X_test.drop(['Norm_brandROI'], axis=1, inplace=True)

        # Model fitter
        brand_model.fit(X_train, Y_train.values.ravel()) #Creates multilinear regression model
    
        """Compare error vs other data sets by calculating Mean Squared Error (MSE)"""
        pred_train  = brand_model.predict(X_train)
        pred_test   = brand_model.predict(X_test)

        #if isinstance(brand_model, RandomForestRegressor):
        pred_train = [[x] for x in pred_train]
        pred_test = [[x] for x in pred_test]

        # Reverse Normalization
        pred_test   = min_max_scaler.inverse_transform(pred_test)
        Y_test      = min_max_scaler.inverse_transform(Y_test)
        pred_train  = min_max_scaler.inverse_transform(pred_train)
        Y_train     = min_max_scaler.inverse_transform(Y_train)
        #compare predicted tests and trains
        # ** Aim to minimize mse **
        mse_test = mean_squared_error(Y_test, pred_test)    
        mse_train = mean_squared_error(Y_train, pred_train)

        test_scores.append(mse_test)
        train_scores.append(mse_train)

        result = next(split, None)

    print("~Testing Mean MSE~: ", np.mean(test_scores))
    print("*Testing MSEs*: ", test_scores)
    print("Training MSEs:  ", train_scores)

    return brand_model, min_max_scaler

"""
FUNCTION: Predictor
 - Creates ROI predictions for output based on model
 - Takes inputs of indicator dataframe to predict and trained model to use
"""
def predictor(indicator_dataframe_predict, model_to_use):
    #if isinstance(model_to_use, RandomForestRegressor):
    predicted = [[x] for x in model_to_use.predict(indicator_dataframe_predict)]
        #predicted = model_to_use.predict(indicator_dataframe_predict)
    #else:
     #   predicted = model_to_use.predict(indicator_dataframe_predict)
    return predicted


"""
FUNCTION: Output constructor
 1. Takes input of: 
    - df_clean_all (what was predicted)
    - df_difference (what was predicted)
    - ROI predictions for brands WITH ROI data
    - Normalization scaler used for predicting the above (for reverse scaling)
    - ROI predictions for brands WITHOUT ROI data
    - Normalization scaler used for predicting the above (for reverse scaling)
 2. Creates unnormalized version of predictions using scalers 
 3. Appends the normalized and unnormalized scalers to output dataframe
 4. Outputs dataframe to .json file in predictions folder
"""
def output_constructor(df_clean_all, df_difference, norm_data_ROI, norm_no_data_ROI, scaler_data, scaler_no_data):
    # Append ROI predictions for Brands with ROI data 
    df_output = df_clean_all.copy()
    df_output["brandROI"] = scaler_data.inverse_transform(norm_data_ROI)
    if isinstance(norm_data_ROI[0], list):
        norm_data_ROI = [x[0] for x in norm_data_ROI]
    df_output["NormbrandROI"] = norm_data_ROI

    # Append ROI predictions for Brands without ROI data 
    df_output_2 = df_difference.copy()
    df_output_2["brandROI"] = scaler_no_data.inverse_transform(norm_no_data_ROI)
    if isinstance(norm_no_data_ROI[0], list):
        norm_no_data_ROI = [x[0] for x in norm_no_data_ROI]
    df_output_2["NormbrandROI"] = norm_no_data_ROI

    df_output = df_output.append(df_output_2, ignore_index=True)
    df_output = df_output.drop_duplicates()   #Workaround for duplicates in certain rows
    df_output.sort_values(by=list(df_output.columns), inplace=True)
    return df_output

def outputter(df_output):
    #File writing code
    out = df_output.to_json(orient='records')
    out_filename = os.path.join(__location__, 
        'predictions/predicted_brandROI_v2.json')
    with open(out_filename, 'w') as f:
        f.write(out)
    f.close()


"""
VISUALIZER: Visualize data
"""
def visualizer(df_input, norm_scaler, axes):
    df_to_visualize = df_input.copy()

    if len(axes) == 1: 
        primary = axes[0]
        df_vis = df_to_visualize[[primary, 'brandROI']].copy()
        df_vis = df_vis.groupby(primary, as_index=False).mean() #Condense brand_ROI to avg
        for pri in df_vis[primary].unique().tolist():
            to_plot = df_vis.loc[df_vis[primary] == pri]
            plotter(pri, to_plot[primary], to_plot["brandROI"])
    elif len(axes) == 2:
        primary = axes[0]
        secondary = axes[1]
        #visualizes for each program where x is brands
        df_vis = df_to_visualize[[primary, secondary, 'brandROI']].copy()
        df_vis = df_vis.groupby([primary, secondary], as_index=False).mean() #Condense brand_ROI to avg
        for pri in df_vis[primary].unique().tolist():
            to_plot = df_vis.loc[df_vis[primary] == pri]
            plotter(pri, to_plot[secondary], to_plot["brandROI"])
    elif len(axes) == 3: #Adds colors and legend for 3rd axis
        primary = axes[0]
        secondary = axes[1]
        tertiary = axes[2]
        #visualizes for each program where x is brands
        df_vis = df_to_visualize[[primary, secondary, tertiary, 'brandROI']].copy()
        df_vis = df_vis.groupby([primary, secondary, tertiary], as_index=False).mean() #Condense brand_ROI to avg
        #display(df_vis)
        for pri in df_vis[primary].unique().tolist():
            to_plot = df_vis.loc[df_vis[primary] == pri]
            plotter(pri, to_plot[secondary], to_plot["brandROI"], to_plot[tertiary])  
    else:
        print("ERROR: Enter [1-3] axes")

    #GET DATAFRAME WHERE: df.loc[df['column_name'] == some_value]
    pass

#TODO: Works wrong for x equals "world" - only shows "color"
def plotter(title, x, y, z=None):
    """plotter"""
    if z is not None: #for 3 axis graph
        fig, ax = plt.subplots()
        for i in np.unique(z):  
            ix = np.where(z == i)
            for ixx in ix:
                ax.scatter(x.iloc[ixx], y.iloc[ixx], label = i) 
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    else:
        fig, ax = plt.subplots()
        #plt.figure()
        ax.scatter(x, y)
    
    #RESIZE plot
        #fig_size = plt.rcParams["figure.figsize"]     # Get current size
        #print ("Current size:", fig_size)     # Prints: [6.0, 4.0]
        #keep 3:2 ratio?
        #fig_size[0] = 18
        #fig_size[1] = 12
        #plt.rcParams["figure.figsize"] = fig_size
        
    ax.set_ylim(-2,9)   
    labels = x
    ax.set_xticklabels(labels, rotation=90)
    ax.set_title(title)
    plt.show()



""" *****************************************************
MAIN RUNNER
****************************************************  """
def main():
    #Order: input_data/roi_name, input_data/no_roi_name
    import_data('input_data/data_ROI_brand_generic.json', 'input_data/no_ROI_brand_generic.json')

    """ JUST CHANGE DATATYPES/FIELDS TO MODEL ON -----------------------------------------------------
    Input columns to run model on (KEEP brandROI in columns_to_keep) - CASE SENSITIVE
    Note: "year" and "quarter" fields have been created from "date" """
    # NOTE: Don't use "quarter" unless df_all also includes quarters
    columns_to_keep = ["brand", "brandBucketCategory", "program", "brandROI", "world"]
    no_data_columns_to_keep = ["brandBucketCategory", "program", "brandROI", "world"]
    #old_cols_to_keep =["brand", "program", "brandROI", "world", "subClass", "brandBucketCategory"] #FOR COMPARISON W/ OLD CODE

    """Choose regression model type: 
        ridge_model, linear_model, randomForest_model, optimizedRF_model, extraForest_model"""
    #randomForest or extraForest are best
    model_type = optimizedRF_model()
    """--------------------------------------------------------------------------------------------"""

    """Trains model for df_data - dataset with ROI data"""
    #df_clean_data  = cleaned up df_data
    df_main_data   = keep_cols(columns_to_keep, df_data)
    df_main_all    = keep_cols(columns_to_keep, df_all)

    #TODO: Empty for some reason
    df_clean_data   = intersection(df_main_data, df_main_all)

    #Creates brand model
    df_data_model_indicator = indicatorer(df_clean_data)
    model_data, scaler_data = trainer(model_type, df_data_model_indicator)

    #Optimize randomforest
    #randomForest_optimizer(df_data_model_indicator)

    """Trains model for df_unknown - dataset without ROI data"""    
    #redo with modded df_data
    #df_data_no_brands: df_data without brands for just training
    df_data_no_brands  = intersection(df_all, df_data) #switched order
    df_data_no_brands  = keep_cols(no_data_columns_to_keep, df_data_no_brands)

    #Creates no brand model
    df_no_data_model_indicator = indicatorer(df_data_no_brands)
    model_no_data, scaler_no_data = trainer(model_type, df_no_data_model_indicator)

    """Predicts data for output on Brands with ROI information (after model trained)"""
    #df_clean_all  = cleaned up df_all for brands also in df_data w/o ROIs
    df_clean_all   = keep_cols(columns_to_keep, df_all)
    #df_clean_all= intersection_df_all(df_clean_all)
    df_clean_all = intersection(df_clean_all, df_data, noROI=True)

    # Predicts data for output on Brands with ROI information
    df_all_indicator = indicatorer(df_clean_all)

    #pred_brand_data_ROI = array of predicted outputs
    pred_brand_data_ROI  = predictor(df_all_indicator, model_data) 

    """Predicts data for output on Brands WITHOUT ROI information (after model trained)"""
    #Switched since difference_df_all, keep_cols order
    df_difference  = difference_df_all(df_all) #LOSES EXCLUSIVE/LIMITED DISTRO

    df_difference_brands = keep_cols(columns_to_keep, df_difference)
    df_difference  = keep_cols(no_data_columns_to_keep, df_difference)

    # Predicts data for output on Brands WITHOUT ROI information
    df_difference_indicator = indicatorer(df_difference)

    """Jank workaround for missing/too many values in non-ROI brand prediction"""
    not_included_over = list(set(list(df_difference_indicator)) - set(list(df_no_data_model_indicator)))
    df_difference_indicator.drop(not_included_over, axis=1, inplace=True)
    
    not_included_under = list(set(list(df_no_data_model_indicator)) - set(list(df_difference_indicator)))
    if not_included_under:
        for i in not_included_under:
            if i != "brandROI":
                df_difference_indicator[i] = 0
    """----------------------------------------------------------------------"""

    #pred_brand_no_data_ROI = array of predicted outputs
    pred_brand_no_data_ROI  = predictor(df_difference_indicator, model_no_data) 

    """Outputs .json dataframe 
        - Requires arrays: pred_brand_data_ROI (model 1), pred_brand_no_data_ROI (model 2)"""
    df_output = output_constructor(
        df_clean_all, df_difference_brands, 
        pred_brand_data_ROI, pred_brand_no_data_ROI, 
        scaler_data, scaler_no_data) #Prepares data
    
    outputter(df_output) #Actually writes file

    """Display data - uncomment to use
        - View input data
        - View predicted ROI"""
    #display(df_data)
    #display(df_output)

    """Visualizes data - uncomment to use
        - Takes 1 or 2 categories to visualize on
        - If 2: First category is primary visualizer"""
    visualizer(df_clean_data, scaler_data, ["program", "brandBucketCategory"]) #brand, brandBucketCategory
    #visualizer(df_output, scaler_data, ["program", "brandBucketCategory", "world"]) #brand, brandBucketCategory    

main()