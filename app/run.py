import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('CategorizedMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    '''Extracts data and visualizes it, receives messages as user input for model'''
    
    # extract data needed for visuals 
    # (1) message genre    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
        
    # (2) messages related to disaster
    related_counts = pd.Series(data={'Related': sum(df["related"] > 0), 'Not related': sum(df["related"] == 0)})
    related_names = list(related_counts.index)
       
    # (3) messages related to weather disasters
    weather_relation = df[['floods', 'storm', 'fire', 'earthquake', 'cold','other_weather']]
    weather_aggregated = pd.DataFrame(data={'weather_related': weather_relation.sum(axis=1)}
                                     ).groupby('weather_related').agg({'weather_related': 'sum'})['weather_related']
    weather_aggregated.loc[0] = sum(df["weather_related"] == 0)
    weather_counts = pd.Series(data=weather_aggregated)
    weather_names = list(weather_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Related / Not Related to Disaster',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Related"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=weather_names,
                    y=weather_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Related to Weather Disasters',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of Weather Phenomena Related to Message"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    
    '''Classifies user's message using pretrained machine learning model'''
    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    
    '''Runs Web Application Disaster Response'''
    
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()