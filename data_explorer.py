import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import ssl
import nltk
from nltk.stem import WordNetLemmatizer
from googletrans import Translator
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

nltk.download('stopwords')

stop_words = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
analyser = SentimentIntensityAnalyzer()
plt.rcParams.update({'font.size': 20})
translator = Translator()

# Clean data set
# Perform feature selection
def setup_data(data):

    # Rename humor labels to fake
    data['label'] = data['label'].replace('humor', 'fake')
    data['label'] = data['label'].replace('fake', 0)
    data['label'] = data['label'].replace('real', 1)

    """
    Feature extraction
    1. VADER sentiment analysis before preprocessing (explained in docs)
    2. TF-IDF
    """
    data['pos_sentiment'] = data['tweetText'].apply(pos_sentiment)

    """
    Preprocess the data
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove URLs
    4. Remove Emojis
    6. Tokenise
    6. Remove stopwords
    7. Lemmatize 
    """
    data['tweetText'] = data['tweetText'].apply(lower)
    data['tweetText'] = data['tweetText'].apply(remove_duplicate_spaces)
    data['tweetText'] = data['tweetText'].apply(remove_url)
    data['tweetText'] = data['tweetText'].apply(remove_emoji)
    data['tweetText'] = data['tweetText'].apply(remove_punctuation)
    data['tweetText'] = data['tweetText'].apply(remove_stopwords)
    data['tweetText'] = data['tweetText'].apply(remove_apostrophe)
    data['tweetText'] = data['tweetText'].apply(lemmatize)

    # fit the vectorizer to the documents
    scores = vectorizer.transform(data['tweetText'])
    scores_frame = pd.DataFrame(scores.todense(), columns = vectorizer.get_feature_names_out())
    result = pd.concat([data, scores_frame], axis=1, join='inner')

    # # Remove unnecessary features
    result = result.drop(['tweetId', 'userId', 'imageId(s)', 'timestamp', 'tweetText', 'username'], axis=1)
    return result

def setup_train_data(data):
    vectorizer.fit(data['tweetText'])
    return setup_data(data)

# Evaluate frequency of a feature
def eval_frequency(data, feature):
    fig, ax = plt.subplots()
    data[feature].value_counts().plot(ax=ax, kind='bar')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.show()

# If not english will translate
def detect_lang(text):
    # No valid text
    if detect(text) == 'en':
        return text
    else:
        return to_english(text)

# Translate all the text to english
def to_english(text):
    return translator.translate(text).text

# Returns the overall sentiment of the text
def pos_sentiment(text):
    vs = analyser.polarity_scores(text)
    return vs['pos']

# Returns number of occurences in a given string
# Can give a list of chars, strings
def number_of(list, string):
    return len(re.findall(list, string))

def remove_duplicate_spaces(text):
    return ' '.join(text.split())

def remove_punctuation(text):
    text = str(text)
    symbols = '''!()-[]{};:"\,<>./?@#$%^&*_~'''
    for char in text:
        if char in symbols:
            text = text.replace(char, "")
    return text

def remove_stopwords(text):
    text = str(text)
    tokens = nltk.word_tokenize(text)
    filtered = []
    for w in tokens:
        if w not in stop_words:
            filtered.append(w)
    return ' '.join(filtered)

def remove_single_letters(text):
    text = str(text)
    filtered = ""
    for w in text:
        if len(w) > 1:
            filtered = filtered + " " + w

def remove_apostrophe(text):
    text = str(text)
    return text.replace("'", "")

def lower(text):
    text = str(text)
    return text.lower()

def lemmatize(text):
    tokens = nltk.word_tokenize(text)
    for w in tokens:
        w = lemmatizer.lemmatize(w)
    return ' '.join(tokens)

def remove_url(text):
    # regular expression pattern to match URLs
    url_pattern = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
    text = re.sub(url_pattern, "", text)
    return text

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64f"  # emoticons
        u"\U0001F300-\U0001F5ff"  # symbols & pictographs
        u"\U0001F680-\U0001F6ff"  # transport & map symbols
        u"\U0001F1e0-\U0001F1ff"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

""" All unecessary features
data['questionCount'] = data.apply(lambda x: number_of(r'\?', x['tweetText']), axis=1)
data['exclamationCount'] = data.apply(lambda x: number_of(r'\!', x['tweetText']), axis=1)
data['fstPersonPronouns'] = data.apply(lambda x: number_of(r'\b([i,I]|[m,M]e|[m,M]ine|[m,M]yself|[w,W]e|[u,U]s|[o,O]urs|[o,O]urselves)\b', x['tweetText']), axis=1)
data['sndPersonPronouns'] = data.apply(lambda x: number_of(r'\b[y,Y](ou|ours|ourself|ourselves)\b', x['tweetText']), axis=1)
data['thirdPersonPronouns'] = data.apply(lambda x: number_of(r'\b[h,H](e|er|imself|is|ers|erself|im|)\b|\b[i,I](t|tself|ts)\b|\b[t,T](hey|hem|hemself|hemselves)\b', x['tweetText']), axis=1)
data['profanityCount'] = data.apply(lambda x: number_of(r'[f,F]uck|[s,S]hit|[a,A]ss|[b,B]itch|[c,C]ock|[c,C]unt|[w,W]anker|[t,T]wat|[p,P]ussy|[p,P]rick|[d,D]ick|[b,B]astard', x['tweetText']), axis=1)
data['mentionsCount'] = data.apply(lambda x: number_of(r'\@', x['tweetText']), axis=1)
data['emojiCount'] = data.apply(lambda x: number_of(r'[\U0001F600-\U0001F64F]|[\U0001F1E0-\U0001F1FF]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]', x['tweetText']), axis=1)
data['urlCount'] = data.apply(lambda x: number_of(r'http', x['tweetText']), axis=1)
Feature extraction
data['hashtagCount'] = data.apply(lambda x: number_of(r'\#', x['tweetText']), axis=1)
data['sentiment'] = data['tweetText'].apply(sentiment_analyse)
data['length'] = data['tweetText'].apply(len)
Remove unnecessary features
data = data.drop(['tweetId', 'userId', 'imageId(s)', 'timestamp', 'tweetText', 'username'], axis=1) """