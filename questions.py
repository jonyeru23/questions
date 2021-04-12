import nltk
import sys
import os
from string import punctuation
from math import log, e
import numpy as np
from time import sleep


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)
    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # make a dict for the txt
    corpus = {}
    for txt in os.listdir(directory):
        with open(os.path.join(directory, txt), encoding="utf8") as file:
            corpus[txt] = file.read()
    return corpus


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document)
    return [word.lower() for word in words if word not in punctuation and
            word not in nltk.corpus.stopwords.words("english")]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.

    how to do:
    loop over each doc, when i see a new word i add it to the dict and i++ to it's value,
    i will keep track of words i saw in each doc to make sure i dont dubble increment
    after i loopes over all the docs, i will make all the words and give them the log value
    """
    # the returned dict
    idf = {}
    # get the number of the docs
    num_docs = len(documents)

    # loop over each doc
    for doc in documents:
        seen = set()
        # loop over each word
        for word in documents[doc]:
            # if the word was seen in the doc continue
            if word in seen:
                continue
            # else, check if it exists in dict, if so i++ if not i = 0
            else:
                if word in idf:
                    idf[word] += 1
                else:
                    idf[word] = 1
                seen.add(word)

    # make it logarithmic
    for word in idf:
        idf[word] = log((num_docs/idf[word]), e)

    return idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.

    the how:
    make a dict with ranking for each doc (doc as key and grade as value)
    iterate over all the words in the query,
    if a word is stopword, continue
    else, calculate her tf-idf and add it to the value of the dict
    the dict with the hiesght value wins
    """
    # make the dict
    ranking = {}
    for file in files:
        ranking[file] = 0

    # iterate over the query
    for word in query:
        if word in nltk.corpus.stopwords.words("english") or word not in idfs:
            continue

        # term frequency and idf
        for file in files:
            ranking[file] += files[file].count(word) * idfs[word]

    keys = list(ranking.keys())
    values = list(ranking.values())
    the_keys = []

    for i in range(n):
        # getting the index of the highest value
        ind = np.argmax(values)
        # appending it to the keys we will return
        the_keys.append(keys[ind])
        # removing the reminders
        values.remove(values[ind])
        keys.remove(keys[ind])

    return the_keys


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # make a dict
    rankings = {}
    for sentence in sentences:
        rankings[sentence] = 0

    for word in query:
        if word in nltk.corpus.stopwords.words("english") or word not in idfs:
            continue

        for sentence in sentences:
            # in other words, if the word is in a sentence rank it higher
            if word in sentences[sentence]:
                rankings[sentence] += idfs[word]

    keys = list(rankings.keys())
    values = list(rankings.values())
    the_keys = []

    for i in range(n):
        # getting the index of the highest value
        ind = np.argmax(values)
        # appending it to the keys we will return
        the_keys.append(keys[ind])
        # removing the reminders
        values.remove(values[ind])
        keys.remove(keys[ind])

    return the_keys


if __name__ == "__main__":
    main()
