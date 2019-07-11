import os
import nltk
import string
import json
import re
import numpy as np
from argparse import ArgumentParser
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename",
                    help="input dataset in json format", metavar="FILE")
parser.add_argument("-p", "--ppt", dest="ppt",
                    help="ppt text information in json format", metavar="FILE")
args = parser.parse_args()


def tokenize_and_stem(s):
    REMOVE_PUNCTUATION_TABLE = str.maketrans(
        {x: None for x in string.punctuation})
    TOKENIZER = TreebankWordTokenizer()
    STEMMER = PorterStemmer()
    return [STEMMER.stem(t) for t
            in TOKENIZER.tokenize(s.translate(REMOVE_PUNCTUATION_TABLE))]


# load documents from directory
with open(args.ppt, 'r') as read_file:
    documents = json.load(read_file)

# build tfdif matrix

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
# preprocess stopwords
stopwords_stemmed = []
stemmer = PorterStemmer()
stopwords_stemmed = []
for word in stopwords:
    for token in nltk.word_tokenize(word):
        stopwords_stemmed.append(stemmer.stem(token))

stopwords = nltk.corpus.stopwords.words('english')
vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                             min_df=0.05, stop_words=stopwords_stemmed,
                             use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

vocab = []

for doc in documents:
    text = ""
    for slide in doc['slides']:
        text = text + " " + slide['text']
    vocab.append(text)

# measure questions against tfidf
with open(args.filename, "r") as read_file:
    questions = json.load(read_file)

maxFilename = ''
for question in questions:
    temp = {}
    temp["question"] = question["content"]
    temp["context"] = question["slideSet"]
	# find best slideset
    doc_tfidf = vectorizer.fit_transform(vocab)
    query_vector = vectorizer.transform([question["content"]])
    slideSet_similarity = cosine_similarity(query_vector, doc_tfidf)
    scores = slideSet_similarity.ravel()
    maxPos = np.argmax(scores)
    maxFilename = documents[maxPos]['slideSet']
    maxFilename = maxFilename.replace(".pptx","")
    # Determine Correctness -> skip to next question if correct slide set was not found
    try:
        # some cleanup for file names. A bit dirty but slight variations in filenames cause problems
        cleanSlideURL = question['slideURL'].replace("external_assets/presentations/slideImages/ppt/","")
        cleanSlideURL = cleanSlideURL.replace(" ","")
        cleanSlideURL = cleanSlideURL.replace("-","")
        cleanSlideURL = cleanSlideURL.replace("_","")

        cleanFilename = maxFilename.replace("_","")      
        if cleanFilename[:4] not in cleanSlideURL:
            print('TFIDF determined: ', maxFilename, ' but was:', cleanSlideURL)
            continue
    except:
        print('No slideURL in question, skipping...')
        continue 
    # calculate tf-idf scores for individual slides
    slideSet = {}
    for elem in documents:
        if maxFilename in elem['slideSet']:
            slideSet = elem
            break
    docs = []
    for elem in slideSet['slides']:
        docs.append(elem['text'])
    slides_tfidf = vectorizer.fit_transform(docs)
    query_vector = vectorizer.transform([question["content"]])
    slides_similarity = cosine_similarity(query_vector, slides_tfidf)

    # create plot
    plt.clf()
    plt.plot(slides_similarity[0], 'b-')
    plt.axvline(question['slide'],0,1,color='r')
    plt.xlabel('Slide index')
    plt.ylabel('Cosine Similarity TFIDF')
    title = "Question: " + question['content'] + "\nSlideSet: " + maxFilename
    plt.title(title)
    plt.savefig('plots/' + question['_id'] + '.png',bbox_inches='tight')
