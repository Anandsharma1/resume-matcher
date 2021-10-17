import unicodedata
import nltk
from nltk.corpus import stopwords
import re
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import os

from src.wordDocParser import getTextWord
from src.pdfDocParser import getTextPDFMiner


def preProcess(text):
    # convert the encoding to ASCII, replacing any non-translatable characters
    # (those not in the 0-127 ASCII range) to nothingness. End decode is needed
    # to convert it back into string
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # convert to lower case
    text = text.lower()

    # lemmatize
    text = [nltk.WordNetLemmatizer().lemmatize(word)
            for word in nltk.word_tokenize(text)]
    text = ' '.join(text)

    # stop word removal
    stop = stopwords.words('english')
    text = [word for word in nltk.word_tokenize(text) if word not in stop]
    text = ' '.join(text)

    # return list of pre-processed strings
    return text


def tokenize(text, maxNgram):
    # split sentences
    sents = text.split('\n')

    # strip white spaces from beginnning and ending and remove empty strings
    text = [sent.strip() for sent in sents if sent.strip() != '']

    # detect sentence boundary and split it accordingly. This returns list of
    # list of strings
    text = [nltk.sent_tokenize(sent) for sent in text]

    # remove trailing characters from each token
    tokensList = []
    for para in text:
        for sent in para:
            tokensList.append(sent.rstrip('.!?!:;'))
    # text = [sent.rstrip('.!?!:;') for sent in text]

    # remove brackets and comma
    # text = [re.sub('[(){}\[\],]', '', sent) for sent in text]
    text = [re.sub('[(){}\[\],]', '', sent) for sent in tokensList]

    tokensList = []
    for sent in text:
        tokens = [token for token in sent.split(' ') if token != '']
        if maxNgram == 5:
            for token in ngrams(tokens, 1): tokensList.append(' '.join(token))
            for token in ngrams(tokens, 2): tokensList.append(' '.join(token))
            for token in ngrams(tokens, 3): tokensList.append(' '.join(token))
            for token in ngrams(tokens, 4): tokensList.append(' '.join(token))
            for token in ngrams(tokens, 5): tokensList.append(' '.join(token))
        elif maxNgram == 4:
            for token in ngrams(tokens, 1): tokensList.append(' '.join(token))
            for token in ngrams(tokens, 2): tokensList.append(' '.join(token))
            for token in ngrams(tokens, 3): tokensList.append(' '.join(token))
            for token in ngrams(tokens, 4): tokensList.append(' '.join(token))
        elif maxNgram == 3:
            for token in ngrams(tokens, 1): tokensList.append(' '.join(token))
            for token in ngrams(tokens, 2): tokensList.append(' '.join(token))
            for token in ngrams(tokens, 3): tokensList.append(' '.join(token))
        elif maxNgram == 2:
            for token in ngrams(tokens, 1): tokensList.append(' '.join(token))
            for token in ngrams(tokens, 2): tokensList.append(' '.join(token))
        else:
            for token in ngrams(tokens, 1): tokensList.append(' '.join(token))

    # return list of tokens
    return tokensList


def getDTM(docNameList, strList, isTFIDF=False, maxNgram=5):
    vectorizer = None
    args = {'maxNgram': maxNgram}
    if (isTFIDF):
        vectorizer = TfidfVectorizer(analyzer='word', lowercase=None,
                                     preprocessor=preProcess,
                                     tokenizer=lambda text: tokenize(text, **args))
    else:
        vectorizer = CountVectorizer(analyzer='word', lowercase=None,
                                     preprocessor=preProcess,
                                     tokenizer=lambda text: tokenize(text, **args))

    mat = vectorizer.fit_transform(strList)

    dtm = pd.DataFrame(mat.toarray(), columns=vectorizer.get_feature_names(),
                       index=docNameList)

    return dtm


def getFilesFromDir(dirPath):
    files = []
    fileStrs = []
    if dirPath is not None and os.path.isdir(dirPath):
        for file in os.listdir(dirPath):
            files.append(file)
            file = os.path.join(dirPath, file)
            try:
                if file.endswith('.pdf'):
                    fileStrs.append(getTextPDFMiner(file))
                elif file.endswith('.docx'):
                    fileStrs.append(getTextWord(file))
                else:
                    print('Unsupported file type found ' + file)
            except Exception as e:
                raise e
                print('No files found in directory ' + dirPath)

    return [files, fileStrs]


# returns list of 2 elements. The first element is list of document names and
# second element is list of strings from the parsed supported documents (docx,
# pdf)
def getFileStrings(jobReqDir, resumeDir):
    jobDetails = getFilesFromDir(jobReqDir)
    resumeDetails = getFilesFromDir(resumeDir)

    return [jobDetails[0] + resumeDetails[0], jobDetails[1] + resumeDetails[1]]