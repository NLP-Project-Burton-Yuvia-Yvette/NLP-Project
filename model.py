


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