# I think I got the idea from:
#https://medium.com/saturdays-ai/building-a-text-summarizer-in-python-using-nltk-and-scikit-learn-class-tfidfvectorizer-2207c4235548


# Beautiful Soup requires you pip install html5lib
from bs4 import BeautifulSoup as bs4
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
import urllib3

## Scraping the website ##
# Asking user for url to scrape #
url = input('Please enter the full url address:')

# Make the request with urllib #
http = urllib3.PoolManager()
works = False
while works == False:
    try:
        response = http.request('GET', url)
        works = True
    except:
        url = input('That url did not work. Please make sure there are no typos:')

# Parsing the response with Beautiful Soup and extracting the text #
soup = bs4(response.data,'html.parser')
text = soup.get_text()

## General cleaning of messy html text in order to properly tokenize ##
text = text.replace('\n','')
text = text.replace('\t','')
text = text.replace('?', '? ')

# Regex split sentences with words like.This #
retokenizer = RegexpTokenizer(pattern = r'\.(?=\S)', gaps=True)
clean_tokens = retokenizer.tokenize(text)
clean_str = ". ".join(clean_tokens)
text = clean_str

# Regex split sentences with endings likeThis #
retokenizer = RegexpTokenizer(pattern = r'[a-z](?=[A-Z])', gaps=True)
retokenizer_d = RegexpTokenizer(pattern = r'[a-z](?=[A-Z])', gaps=False)
clean_tokens = retokenizer.tokenize(text)
end_letters = retokenizer_d.tokenize(text)
for x in np.arange(0,len(end_letters)):
    clean_tokens[x] = clean_tokens[x] + end_letters[x]

clean_str = ". ".join(clean_tokens)

## Tokenizing ##
# Stemming & Lowering #
porter = PorterStemmer()
# Sentence Tokenizing then Word Tokenizing #
sents = sent_tokenize(clean_str)
words = [word_tokenize(x) for x in sents]
stemmed = []
for y in words: stemmed.append([porter.stem(x) for x in y])

joined = [' '.join(x) for x in stemmed]
# Transforming sentence list into Tfidf matrix #
tv = TfidfVectorizer(stopwords={'english'})
sparse = tv.fit_transform(joined)
sparsed_words = tv.get_feature_names_out()
matrix = pd.DataFrame.sparse.from_spmatrix(sparse, index = sents, columns = sparsed_words)
# Noted words is a Series of the most significant words in the corpus #
noted_words = matrix.sum(axis=0)
# Summed is the final Series of sentences with the summed Tfidf score of all the words in the sentence #
summed = matrix.sum(axis=1)
summed.sort_values(ascending=False, inplace=True)
# The summary is made of the top 5 sentences #
summary = ' '.join(summed.index[:5])
## Final Result ##
print(summary)


