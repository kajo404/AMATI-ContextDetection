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

directory = "docs/"
documents = []

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename",
                    help="input dataset in json format", metavar="FILE")
args = parser.parse_args()

def tokenize_and_stem(s):
            REMOVE_PUNCTUATION_TABLE = str.maketrans({x: None for x in string.punctuation})
            TOKENIZER = TreebankWordTokenizer()
            STEMMER = PorterStemmer()
            return [STEMMER.stem(t) for t 
                in TOKENIZER.tokenize(s.translate(REMOVE_PUNCTUATION_TABLE))]

# load documents from directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        doc = open(directory + filename,'r')
        lines = doc.read()
        doc.close()
        documents.append((filename.replace(".txt",""),lines))

# build tfdif matrix

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.05, stop_words = stopwords,
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

vocab = []

for doc in documents:
    vocab.append(doc[1])

doc_tfidf = vectorizer.fit_transform(vocab)

print(doc_tfidf.shape)

# measure questions against tfidf
with open(args.filename, "r") as read_file:
    questions = json.load(read_file)
result = []
correct = 0
for question in questions:
    temp = {}
    temp["question"] = question["content"]
    temp["context"] = question["slideSet"]

    query_vector = vectorizer.transform([question["content"]]) 
    similarity = cosine_similarity(query_vector, doc_tfidf)
    scores = similarity.ravel()
    maxPos = np.argmax(scores)
    maxFilename = documents[maxPos][0]
    temp["cosSim"] = int(re.findall(r'\d+',maxFilename)[0])
    result.append(temp)
    if temp["context"] == temp["cosSim"]:
        correct = correct + 1

accuracy = correct/len(questions)

print("Accuracy: " + str(accuracy))


