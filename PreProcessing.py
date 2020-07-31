from textblob import TextBlob
from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords
import codecs
import re

from nltk.stem.wordnet import WordNetLemmatizer


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer

contractions = {
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "i had / i would",
"i'd've": "i would have",
"i'll": "i shall / i will",
"i'll've": "i shall have / i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

lem = WordNetLemmatizer()
stemmer= PorterStemmer()


def normalize_review(text):
    for word in text.split():
        if word.lower() in contractions:
            text = text.replace(word, contractions[word.lower()])
    return text


def removeAllConsecutive(text):
    pattern = re.compile(r"(.)\1{2,}")

    for word in text.split():
            text = text.replace(word, pattern.sub(r"\1\1",word))

    # text = TextBlob(text)

    return text




def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [stemmer.stem(word) for word in words]
    return words


def readwords(filename):
    f = open(filename)
    words = [line.rstrip() for line in f.readlines()]
    return words


positive = readwords('positive.txt')
negative = readwords('negative.txt')


def Positive_word_count(text):
    count = Counter(text.split())
    pos = 0

    for key, val in count.items():
        if key in positive:
            pos += val
    return pos


def Negative_word_count(text):
    count = Counter(text.split())
    neg = 0

    for key, val in count.items():
        if key in negative:
            neg += val
    return neg

def sentiment(text):

    analysis = TextBlob(text)
    if analysis.sentiment[0] > 0:
        return 'positive'
    elif analysis.sentiment[0] <= 0:
        return 'negative'


data = pd.read_csv(r"Dataset.csv")

permanent = data[['categories', 'name', 'reviews.title' , 'reviews.rating' , 'reviews.text', 'reviews.doRecommend']]

df2=permanent.dropna(axis=0,how='any')

df2['Combined_feature'] = df2['reviews.title'] + df2['reviews.text']

df2['Labels'] = df2['reviews.doRecommend']

df2['mod_name_1'] = df2['name'].apply(''.join).str.replace('[^A-Za-z\s]+','')

df2['mod_name_2'] = df2['mod_name_1'].replace('\n','', regex=True)

df2['Product_Name'] = df2['mod_name_2'].replace('\r','', regex=True)

df2['review_without_trailing_leading_blank spaces']=df2['Combined_feature'].str.strip()

df2['review_toLower']=df2['review_without_trailing_leading_blank spaces'].str.lower()

df2['review_without_newline'] = df2['review_toLower'].replace('\n','', regex=True)

df2['normalize']=df2['review_without_newline'].apply(normalize_review)

df2['review_without_specialCharacters_alphaNumerical']=df2['normalize'].apply(''.join).str.replace('[^A-Za-z\s]+','')

df2['removed_repetation_chars']=df2['review_without_specialCharacters_alphaNumerical'].apply(removeAllConsecutive)

stop_words = set(stopwords.words('english'))
df2['review_without_stopwords'] = df2['removed_repetation_chars'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

df2['Feature_0'] = df2['review_without_stopwords']

df2['Feature_1'] = df2['review_without_stopwords'].apply(lambda x: TextBlob(x).sentiment[0])

df2['Feature_2'] = df2['reviews.rating']

df2['Sentiment'] = df2['review_without_stopwords'].apply(sentiment)

df2['subjectivity'] = df2['review_without_stopwords'].apply(lambda x: TextBlob(x).sentiment[1])

df2['positive_words'] = df2['review_without_stopwords'].apply(Positive_word_count)

df2['negative_words'] = df2['review_without_stopwords'].apply(Negative_word_count)

export_csv = df2.to_csv (r'Dataset_preprocessed.csv', index = None, header=True)
f = codecs.open("Dataset_preprocessed.csv","r","utf-8")
samplewords = f.read()
f.close()
