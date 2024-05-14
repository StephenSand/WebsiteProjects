### Goals: ###
### Using the reviews data to identify positive and negative reviews
### Then extracting the most significant words/phrases from both the positive and negative reviews

### Warning! ###
### If you actually try to run this, please be warned that it takes a very long time.
### In production I constantly saved pickle files to continue where I left off rather than starting from the beginning.
### As such I commented out a lot of steps in production but will leave them in here.
### This was one of my older projects and could use some cleaning.

## Imports
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.util import *
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import tokenize
from nltk.corpus import stopwords
import argostranslate.package
import argostranslate.translate





## Functions
# A function for using langdetect to detect language
from langdetect import DetectorFactory, detect
DetectorFactory.seed = 0
def detect_lang(x):
    try:
        a = detect(x)
    except:
        a = None
    return a

# A function for translating the review using argos translate
def translate_lang(lang):
    a = foreign2[foreign2['lang'] == lang ]
    to_lang = list(filter(lambda x: x.code == "en", installed_languages))[0]
    from_lang = list(filter(lambda x: x.code == lang, installed_languages))[0]
    translation = from_lang.get_translation(to_lang)
    b = []
    for x in a['comments']:
        b.append(x)
    c = []
    for x in b:
        c.append(translation.translate(x))
    
    a['translated'] = c
    return a

# A function to curb run on sentences from being tokenized as a sentence
retokenizer = tokenize.RegexpTokenizer(pattern = r'\.(?=\S)', gaps=True)
def split_it(x):
    a = x.split('...')
    b = '. '.join(a)
    clean_tokens = retokenizer.tokenize(b)
    clean_str = ". ".join(clean_tokens)

    return clean_str


# A function for using the tfidfvectorizer
def enter_the_tfidfmatrix(sentence_list, stem=True):
    # Stemming
    porter = PorterStemmer()
    # Word Tokenizing
    words = [tokenize.word_tokenize(x) for x in sentence_list]
    stemmed = []
    for y in words: stemmed.append([porter.stem(x) for x in y])
    
    joined = [' '.join(x) for x in stemmed]
    if stem == False:
        lowered = []
        for y in words: lowered.append([x.lower() for x in y])
        
        joined = [' '.join(x) for x in lowered]
    stop_words = list(stopwords.words('english'))

    tv = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,3))
    sparse = tv.fit_transform(joined)
    sparsed_words = tv.get_feature_names_out()
    matrix = pd.DataFrame.sparse.from_spmatrix(sparse, index = sentence_list, columns = sparsed_words)
    return matrix


# A function that returns the sentiment of the text in a review, positive for good, negative for bad
def sentimentize(x):
    sentiment = 0
    for y in x:
        scores = sid.polarity_scores(y)
        sentiment += (scores['pos'] - scores['neg'])
    
    return sentiment




# commented out in production
## Reading the reviews csv file
reviews = pd.read_csv('reviews.csv')
# Taking a look at the dataset
reviews.head()
reviews.dropna(inplace=True)
reviews['comments'] = reviews['comments'].str.replace("</br>"," ")
reviews['comments'] = reviews['comments'].str.replace("<br/>"," ")
reviews['comments'] = reviews['comments'].str.replace("\r","")


#commented out in production
## Detecting language of review
reviews['lang'] = reviews['comments'].apply(detect_lang)
reviews = reviews[reviews['lang'].isna() == False]
reviews['textlang'] = list(zip(reviews['comments'],reviews['lang']))
reviews.to_pickle('reviews_translated.pickle')


# commented out in production - Argostranslate notes
# Download and install Argos Translate package - this is like 20GBs by the way
#argostranslate.package.update_package_index()
#available_packages = argostranslate.package.get_available_packages()
#for x in available_packages:
#    download_path = x.download()
#    argostranslate.package.install_from_path(download_path)
#
# argos packages saved in C:\Users\user\.local\share\argos-translate\packages
#installed_languages = argostranslate.translate.get_installed_languages()
#eng = installed_languages[0]
#dicts = [x.get_translation(eng) for x in installed_languages[1:]]
# Find the position of the language dictionary that matches the text's language in dicts
# EXAMPLE: the French -> English dictionary is 14 in the dicts list so for french you would use dicts[14]
#dictionary = dicts[42]
# Make the translator
#translator = dictionary.translate(<text as a str>)
#
# foreign languages = ['fr', 'zh-cn', 'so', 'ru', 'ko', 'it', 'ro', 'de', 'hr', 'cs',
#       'ca', 'es', 'ja', 'zh-tw', 'af', 'pl', 'vi', 'sv', 'pt', 'nl',
#       'he', 'id', 'fi', 'da', 'tl', 'cy', 'uk', 'no', 'sk', 'tr', 'mk',
#       'th', 'bg', 'hu', 'el', 'ar', 'sw', 'et', 'lv', 'sq', 'lt', 'sl',
#       'ur', 'fa']
# installed argos libraries = ['en', 'sq', 'ar', 'az', 'bn', 'bg', 'ca', 'zh', 'zt', 'cs', 'da', 'nl', 'eo', 'et', 'fi', 'fr', 'de', 'el', 'he', 'hi', 'hu', 'id', 'ga', 'it', 'ja', 'ko', 'lv', 'lt', 'ms', 'nb', 'fa', 'pl', 'pt', 'ro', 'ru', 'sr', 'sk', 'sl', 'es', 'sv', 'tl', 'th', 'tr', 'uk']
# langs not in installed = ['zh-cn', 'so', 'hr', 'zh-tw', 'af', 'vi', 'cy', 'no', 'mk', 'sw', 'ur']


# commented out in production
reviews = pd.read_pickle('reviews_translated.pickle')
foreign = reviews[reviews['lang'] != 'en']
langs = foreign['lang'].unique()
installed_languages = argostranslate.translate.get_installed_languages()
installed = [x.code for x in installed_languages]
# Here I was trying to see how many comments were in cantonese - 'zh-cn' and 'zh-tw'
foreign2 = foreign
foreign2['lang'] = foreign2['lang'].str.replace('zh-cn','zh')
foreign2['lang'] = foreign2['lang'].str.replace('zh-tw','zh')
for x in langs:
    if x not in installed:
        foreign2 = foreign2[foreign2['lang'] != x]

# commented out in production
#len(foreign)
#76845
#len(foreign2)
#33188
#len(reviews)
#263131
## This is where translation begins
## ALl reviews will be separated by language, a translator will be made from that language to English, and the resulting dataframe will be saved in a pickle file
fin_langs = foreign2['lang'].unique()
for x in fin_langs:
    df = translate_lang(x)
    thename = x + '.pickle'
    df.to_pickle(thename)

# commented out in production
# Now we are adding all the pickle files of the translated reviews together into one
foreign3 = pd.DataFrame()
for x in fin_langs:
    a = pd.read_pickle(x+'.pickle')
    foreign3 = pd.concat([foreign3,a],axis=0)

# commented out in production
# I'm adding the translated reviews to the original reviews dataframe
reviews['translated'] = reviews['comments']
b = reviews.copy()
for x in foreign3.index:
    a = foreign3.loc[x]['translated']
    b['translated'][x] = a

# commented out in production
# Pickling the original dataframe with new translations feature
b.to_pickle('reviews_translated_final.pickle')

# Fixing sentences to tokenize later
reviews = pd.read_pickle('reviews_translated_final.pickle')
reviews['translated'] = reviews['translated'].apply(split_it)
## Dropping reviews that could not be translated and pickling resulting df
print(len(reviews))
#263131
reviews['lang'] = reviews['translated'].apply(detect_lang)
reviews = reviews[reviews['lang'].isna() == False]
reviews = reviews[reviews['lang'] == 'en']
print(len(reviews))
#250742
#pd.options.display.max_columns = None
#pd.options.display.max_rows = None
#pd.options.display.max_colwidth = None
reviews.to_pickle('final_df_translated.pickle')

# Picking out some reviews that were translated incorrectly but didn't return Nones
reviews = pd.read_pickle('final_df_translated.pickle')
#some rows weren't caught by langdetect or weren't translated right
rogue_rows = [116596, 29188]
#others in bad_df = [18322, 144044, 245299, 190311, 47324, 177868, 44065, 193796, 174730, 248042, 261150, 108343, 14164, 161363, 219704, 150359]
reviews.drop(rogue_rows, axis=0, inplace=True)
reviews = reviews[reviews['translated'] != 'Adress: No. 5 Yard, 7th Guanggu Street, Badaling Economic Development Zone, Yanqing District, Beijing, China']

# Tokenizing
reviews['tokens'] = reviews['translated'].apply(lambda x: tokenize.sent_tokenize(str(x)))
sid = SentimentIntensityAnalyzer()
reviews['sentiment'] = reviews['tokens'].apply(sentimentize)
# Finding the most positive reviews
good = reviews[reviews['sentiment'] > 0]
good.sort_values(by='sentiment', ascending = False, inplace = True)
# Finding the most negative reviews
bad = reviews[reviews['sentiment'] < 0]
bad['sentiment'] = bad['sentiment'].abs()
bad.sort_values(by='sentiment', ascending=False, inplace=True)
# Not enough memory for all entries - we'll use the top 1000 entries each for both the good and bad reviews
good_df = good[:1000]
bad_df = bad[:1000]

# Creating the corpus each for positive and negative reviews
good_corpus = good_df['tokens'].sum()
bad_corpus = bad_df['tokens'].sum()


## good summary
# not stemmed
stemless_good_matrix = enter_the_tfidfmatrix(good_corpus,stem=False)
stemless_good_words = stemless_good_matrix.sum(axis=0)
stemless_good_words.sort_values(ascending=False, inplace=True)
stemless_good_summary= stemless_good_matrix.sum(axis=1)
stemless_good_summary.sort_values(ascending=False,inplace=True)

# stemmed with porterstemmer
good_matrix = enter_the_tfidfmatrix(good_corpus)
good_words = good_matrix.sum(axis=0)
good_words.sort_values(ascending=False,inplace=True)
good_summary = good_matrix.sum(axis=1)
good_summary.sort_values(ascending=False, inplace=True)



## bad summmary
# not stemmed
stemless_bad_matrix = enter_the_tfidfmatrix(bad_corpus,stem=False)
stemless_bad_words = stemless_bad_matrix.sum(axis=0)
stemless_bad_words.sort_values(ascending=False, inplace=True)
stemless_bad_summary= stemless_bad_matrix.sum(axis=1)
stemless_bad_summary.sort_values(ascending=False,inplace=True)
ghosts = [stemless_bad_summary.index[0],stemless_bad_summary.index[2],stemless_bad_summary.index[5],stemless_bad_summary.index[7]]
stemless_bad_summary.drop(ghosts,axis=0,inplace=True)

# stemmed with porterstemmer
bad_matrix = enter_the_tfidfmatrix(bad_corpus)
bad_words = bad_matrix.sum(axis=0)
bad_words.sort_values(ascending=False,inplace=True)
bad_summary = bad_matrix.sum(axis=1)
bad_summary.sort_values(ascending=False, inplace=True)
ghosts = [bad_summary.index[0], bad_summary.index[4], bad_summary.index[7]]
bad_summary.drop(ghosts,axis=0,inplace=True)



# Final results for good reviews
a = pd.DataFrame(stemless_good_words[:20])
a.rename({0:'Tfidf Weighted Significance'}, axis=1, inplace=True)
a.rename_axis('Word', inplace=True)
good_words_html = a.to_html().replace('\n','')
f = open("good.txt","w")
f.write('Best Words: '+good_words_html+' Summary: '+str(good_summary.index[:5]))
f.close()

# Final results for bad reviews
a = pd.DataFrame(stemless_bad_words[:20])
a.rename({0:'Tfidf Weighted Significance'}, axis=1, inplace=True)
a.rename_axis('Word', inplace=True)
bad_words_html = a.to_html().replace('\n','')
g = open("bad.txt","w",encoding="utf-8")
g.write('Worst Words: '+bad_words_html+' Summary: '+str(bad_summary.index[:5]))
g.close()






