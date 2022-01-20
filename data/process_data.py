import sys


def load_data(messages_filepath, categories_filepath):
    
    import pandas as pd
    
    # Read in csv files for messages and categories
    messages = pd.read_csv(messages_filepath, dtype="str")
    categories = pd.read_csv(categories_filepath, dtype="str")
    
    # Combine messages and categories dataframes 
    df = pd.merge(messages, categories, on="id", how="outer")
    
    print("Loading completed")
    return df


def clean_data(df):
    
    import pandas as pd
    
    # Make each category a separate column
    categories = df["categories"].str.split(";", expand=True)
    
    # Rename all 36 category columns
    extract_category = lambda cat: cat[0:-2]
    categories.columns = categories.iloc[0].apply(func=extract_category)

    # Clean and convert all category columns to numeric
    for column in categories:
        categories[column] = pd.to_numeric(categories[column].str[-1:])
        
    # Merge message data and expanded category columns
    df.drop(columns="categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df = df[~(df.duplicated(keep='first'))]
    
    print("Cleaning completed")
    return df


def save_data(df, database_filename):
    
    import sqlite3
    from sqlalchemy import create_engine
    
    # Saving dataframe to database
    engine = create_engine(('sqlite:///'+database_filename))
    df.to_sql('CategorizedMessages', engine, index=False, if_exists='replace')
    
    print("Saving completed")
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()