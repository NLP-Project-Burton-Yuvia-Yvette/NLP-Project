import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

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

def cv_countvectorizer(X_train):
    cv = CountVectorizer()

    return cv
    
    

    ################## YC #############

def get_tree(x_train, y_train, x_validate, y_validate, x_test, y_test,cv):
    '''
    Function gets Decision Tree model accuracy on train and validate data set 
    ''' 
    # create decision tree model using defaults and random state to replicate results
    tree = DecisionTreeClassifier(max_depth=3)

    # fit model on training data
    X_bow = cv.fit_transform(x_train)
    tree.fit(X_bow, y_train)
    train_score= tree.score(X_bow, y_train)
    
    # Whatever transformations we apply to X_train need to be applied to X_test
    X_bow_val = cv.transform(x_validate)
    val_score =tree.score(X_bow_val, y_validate)

    return train_score, val_score
    
def get_forest(x_train, y_train, x_validate, y_validate, x_test, y_test,cv):
    X_bow1 = cv.fit_transform(x_train)
    rf = RandomForestClassifier(max_depth =6, 
                            min_samples_leaf = 2, 
                            random_state=123)
    rf.fit(X_bow1, y_train)
    train_score = rf.score(X_bow1, y_train)
    
    # Whatever transformations we apply to X_train need to be applied to X_test
    X_bow_val = cv.transform(x_validate)
    val_score =rf.score(X_bow_val, y_validate)
    
    return train_score, val_score

def get_knn(x_train, y_train, x_validate, y_validate, x_test, y_test,cv):
    X_bow = cv.fit_transform(x_train)
    knn = KNeighborsClassifier(n_neighbors=6, weights='uniform')
    knn.fit(X_bow, y_train)
    train_score = knn.score(X_bow, y_train)
    
    # Whatever transformations we apply to X_train need to be applied to X_test
    X_bow_val = cv.transform(x_validate)
    val_score =knn.score(X_bow_val, y_validate)
    
    return train_score, val_score


########################## Model Test Functions ##############################


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

