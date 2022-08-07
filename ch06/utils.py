import re
import string

import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stemmer = nltk.stem.snowball.EnglishStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Stop Words
stop_words = set(nltk.corpus.stopwords.words("english"))

# Cleaning the corpus
def clean_text(text):
    """
    Makes text lowercase for better comparison.
    Removes punctuation, trailing characters, text between
    square brackets, words containing numbers and links.
    Similar function to the one built for Chapter 2, Recipe 4
    """
    text = str(text).lower()
    text = re.sub("\[.*?\']", '', text)
    text = re.sub("https?://\S+|www\.\S+", '', text)
    text = re.sub("<.*?>+", '', text)
    text = re.sub("[%s]" % re.escape(string.punctuation), '', text)
    text = re.sub("\n", ' ', text)
    text = re.sub("\w*\d\w*", '', text)
    # No emails, so word subject can be interesting and is not removed
    # text = re.sub("subject", '', text)
    text = re.sub("\\r", ' ', text)
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    for p in punct:
        text = text.replace(p, '')
    return text

def process_words(
    text,
    stop_words=stop_words,
    stemmer = stemmer,
    lemmatizer = lemmatizer):
    
    words = nltk.tokenize.word_tokenize(text)
       
    filtered_words_post = []
    
    for word in words:
        
        if word not in stop_words and word.isalpha():
            word = stemmer.stem(word)
            filtered_words_post.append(lemmatizer.lemmatize(word))
    
    return filtered_words_post

def process_words_basic(
    text,
    lemmatizer = lemmatizer):
    
    words = nltk.tokenize.word_tokenize(text)
       
    filtered_words_post = []
    
    for word in words:
        
        if word.isalpha():
            filtered_words_post.append(lemmatizer.lemmatize(word))
    
    return filtered_words_post
