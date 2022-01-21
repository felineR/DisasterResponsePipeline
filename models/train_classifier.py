import sys


def load_data(database_filepath):
    
    '''Retrieves precleaned data from database and splits it into X and y values. Returns X, y, and category names.'''
    
    # import libraries
    import sqlite3
    from sqlalchemy import create_engine
    import pandas as pd
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    table_name = "CategorizedMessages"
    df = pd.read_sql_table(table_name=table_name, con=engine)
    X = df["message"]
    y = df.drop(axis = 1, columns=["id", "message", "original", "genre"])
    category_names = y.columns
    
    print("Loading completed.")
    return X, y, category_names


def tokenize(text_message):
    
    '''Takes in text message, and cleans, tokenizes, lemmes and stemmes it. Returns preprocessed text_message as list of tokens. '''
    
    # import libraries
    import re
    
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    
    # Convert to all lower case letters
    text_message.lower()

    # Replace URLs within message with a placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text_message = re.sub(url_regex, "urlplaceholder", text_message)

    # Remove punctuation from message
    text_message = re.sub('[^a-zA-Z0-9]', " ", text_message)
    
    # Tokenzize message, remove stop words as well as leading and trailing white spaces
    tokens = word_tokenize(text_message)
    tokens = [t.strip() for t in tokens if t not in stopwords.words("english")]

    # Stemming and lemmatizing tokens
    tokens = [PorterStemmer().stem(t) for t in tokens]
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    
    return tokens


def build_model():
    
    '''
    Sets up machine learing pipeline including Count Vectorizer, TfidfTransformer, and MultiOutputClassifier with Random Forest Classifier.
    Uses GridSearchCV to tweak the models parameters for an improved model version.
    Returns a machine learning model.
    '''
    
    # import libraries
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.model_selection import GridSearchCV
    
    # create machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize))
        , ('tfidf', TfidfTransformer())
        , ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    # set parameters and use GridSearchCV to improve model
    parameters = {
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10,50,100]
    }
    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    print("Building completed.")
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    
    '''
    Takes in a machine learning model, test data X_test and y_test, as well as category names.
    Calculates the model's prediction quality based on three measures: f1_score, precision_score, recall_score.
    Prints out the metrics per category.
    '''
    
    # import libraries
    import pandas as pd
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    
    # predict categories based on trained model
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = y_test.columns

    # Evaluate prediction results
    evaluation_results = []
    for col in y_test.columns:
        f1 = f1_score(y_test[col], y_pred[col], average='micro')
        precision = precision_score(y_test[col],y_pred[col], average='micro')
        recall = recall_score(y_test[col],y_pred[col], average='micro')
        
        evaluation_results.append([f1, precision, recall])
        
    # Print evaluation results per category    
    print("".ljust(22), '| ', "f1".ljust(6), ' | ', "precision".ljust(9), ' | ', "recall".ljust(8), ' |')
    for i in range(0,36):
        print(y_test.columns[i].ljust(22), '| ', 
          str(round(evaluation_results[i][0],3)).ljust(6), ' | ', 
          str(round(evaluation_results[i][1],3)).ljust(9), ' | ', 
          str(round(evaluation_results[i][2],3)).ljust(8), ' |')
        
    print("Evaluating completed.")
    pass


def save_model(model, model_filepath):
    
    '''Takes in a machine learning model and a filepath. Saves the machine learning model into the specified path.'''
    
    # import libraries
    import pickle
    
    # saving model
    pickle.dump(model, open(model_filepath, 'wb'))
    
    print("Saving completed.")
    pass


def main():
    
    '''
    Loads data from database, splits data into training and test data, builds machine learning model, trains machine learning model,
    makes predictions based on machine learning model, evaluates prediction results, and saves the machine learning model.
    Prints out error message if arguments do not match expected number of arguments (two: database_filepath and model_filepath).
    '''
    
    # import libraries
    from sklearn.model_selection import train_test_split
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        print("Training completed.")
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()