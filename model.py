import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

def model_prep(train,validate,test):
    
    ''' This function takes in data that has been split into train, validate and test and prepares for modeling 
        returns
    '''
    
    #seperate target
    X_train = train['clean_text']
    y_train = train['language']

    X_validate = validate['clean_text']
    y_validate = validate['language']

    X_test = test['clean_text']
    y_test = test['language']
        
        
    return X_train,y_train,X_validate,y_validate, X_test, y_test



    ################## YC #############

def get_tree(x_train, y_train, x_validate, y_validate, x_test, y_test):
    '''
    Function gets Decision Tree model accuracy on train and validate data set 
    ''' 
    # create decision tree model using defaults and random state to replicate results
    tree1 = DecisionTreeClassifier(max_depth=3, random_state=123)

    # fit model on training data
    tree1 = tree1.fit(x_train, y_train)

    # print result
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(tree1.score(x_train, y_train)))
    print('Accuracy of Decision Tree classifier on validate set: {:.2f}'
      .format(tree1.score(x_validate, y_validate)))
    
def get_forest(x_train, y_train, x_validate, y_validate, x_test, y_test):
    '''
    Function gets Random Forest model accuracy on train and validate data set 
    ''' 
    # create random forest model using random state to replicate results
    rf = RandomForestClassifier(max_depth =5, 
                            min_samples_leaf = 1, 
                            random_state=123)
    # fit model on training data
    rf.fit(x_train, y_train)

    # print result
    print('Accuracy of Random Forest classifier on training set: {:.2f}'
      .format(rf.score(x_train, y_train)))
    print('Accuracy of Random Forest classifier on validate set: {:.2f}'
      .format(rf.score(x_validate, y_validate)))

def get_reg(x_train, y_train, x_validate, y_validate, x_test, y_test):
    '''
    Function gets Logistic Regression model accuracy on train and validate data set 
    '''
    # create logistic regression model
    logit = LogisticRegression(C=1,random_state=123)
    # specify the features used
    features_model1 = ['contract_type_Month-to-month', 'contract_type_One year', 'contract_type_Two year']

    # fit model on training data
    logit.fit(x_train[features_model1], y_train)

    # print result
    print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
      .format(logit.score(x_train[features_model1], y_train)))
    print('Accuracy of Logistic Regression classifier on validate set: {:.2f}'
      .format(logit.score(x_validate[features_model1], y_validate)))

def get_knn(x_train, y_train, x_validate, y_validate, x_test, y_test):
    '''
    Function gets KNN model accuracy on train and validate data set 
    '''
    # create KNN model
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    # fit model on training data
    knn.fit(x_train, y_train)

    # print results
    print('Accuracy of KNN classifier on training set: {:.2f}'
      .format(knn.score(x_train, y_train)))
    print('Accuracy of KNN classifier on validate set: {:.2f}'
      .format(knn.score(x_validate, y_validate)))

########################## Model Test Functions ##############################

def get_rf_test(x_train, y_train, x_validate, y_validate, x_test, y_test):
    '''
    Function gets Random Forest model accuracy on test data set 
    ''' 
    # create random forest model using random state to replicate results
    rf = RandomForestClassifier(max_depth =5, 
                            min_samples_leaf = 1, 
                            random_state=123)
    # fit model on training data
    rf.fit(x_train, y_train)

    # print result
    print('Accuracy of Random Forest classifier on test set: {:.2f}'
      .format(rf.score(x_test, y_test)))


###################################################
################## Evaluate Models ################
###################################################
models = ['Baseline Train', 'SimpleLinear Train', 'GeneralizedLinear Train','Baseline Validate', 'SimpleLinear Validate', 'GeneralizedLinear Validate']
def make_stats_df():
    '''
    Function creates dataframe for results of pearsonsr statistical 
    test for all features.
    '''
    evaluate_df = pd.DataFrame()
    evaluate_df['models'] = models
    return evaluate_df

def final_eval(train, validate, evaluate_df):
    base_train = baseline_mean_errors(train,validate)
    simp_train = lm_errors(train)
    gen_train = glm_errors(train)
    base_val = baseline_mean_errors(train,validate)
    simp_val = lm_errors(validate)
    gen_val = glm_errors(validate)


    scores = [base_train, simp_train, gen_train, base_val, simp_val, gen_val]
    evaluate_df['RMSE']=scores
    
    return evaluate_df

def baseline_mean_errors(train):
    
    train['Baseline']= train.Happiness_Score.mean()
    y = train.Happiness_Score
    yhat = train.Baseline
    
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    R2 = explained_variance_score(y,yhat)
    base_train = RMSE
    
    #print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
    #  "\nRMSE: ", round(RMSE, 2))
    return base_train
    
def lm_errors(train):
    y = train.Happiness_Score
    yhat = train.lm_predictions

    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    R2 = explained_variance_score(y,yhat)
    simp_train = RMSE
    
    #print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
    #  "\nRMSE: ", round(RMSE, 2))
    return simp_train

def glm_errors(train):
    y = train.Happiness_Score
    yhat = train.glm_predictions
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    R2 = explained_variance_score(y,yhat)
    gen_train = RMSE
    
    #print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
    #  "\nRMSE: ", round(RMSE, 2))
    return gen_train

def baseline_mean_errors(train, validate):
    
    validate['Baseline']= train.Happiness_Score.mean()
    y = validate.Happiness_Score
    yhat = validate.Baseline
    
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    R2 = explained_variance_score(y,yhat)
    base_val = RMSE
    
    #print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
    #  "\nRMSE: ", round(RMSE, 2))
    return base_val
    
def lm_errors(validate):
    y = validate.Happiness_Score
    yhat = validate.lm_predictions

    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    R2 = explained_variance_score(y,yhat)
    simp_val = RMSE
    
    #print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
    #  "\nRMSE: ", round(RMSE, 2))
    return simp_val

def glm_errors(validate):
    y = validate.Happiness_Score
    yhat = validate.glm_predictions
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    R2 = explained_variance_score(y,yhat)
    gen_val = RMSE
    
    #print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
    #  "\nRMSE: ", round(RMSE, 2))
    return gen_val

