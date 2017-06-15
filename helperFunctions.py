# -*- coding: utf-8 -*-
"""
Function used in Review Analysis
"""

# Load necessary libraries
import os
import pandas as pd
from pathlib import Path
import io
import numpy as np
from collections import OrderedDict

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
#nltk.download('punkt')
#nltk.download('maxent_treebank_pos_tagger')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
sys.path.append('vaderSentiment-master')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk import ContextIndex
from nltk.corpus import wordnet as wn
from nltk import FreqDist

#------------------------------------------------------------------------------------------------

## Read reviews data
def readReviewsData(dataPath, city, year):
    dataPath = dataPath+'/'+city
    
    # Get the hotel files
    hotelReviewList = os.listdir(dataPath)

    # Create data frame to store reviews of all hotels
    reviewsDF = pd.DataFrame()

    # Loop through all hotel files and extract reviews
    for filename in hotelReviewList:
        filePath = dataPath+'/'+filename
        dataStr = Path(filePath).read_text()                                    # Read data from file as string (unicode issue)
        dataDf = pd.read_csv(io.StringIO(dataStr),sep='\t',skipinitialspace = True,
                            names=['Date','Title','Review','Junk'], quoting=3)  # Convert to reviews in string to data frame
        dataDf = dataDf[['Date','Title','Review']]                              # Data frame must contain 3 cols - Date, Title, Review
        dataDf['FullReview'] = dataDf['Title']+'. '+dataDf['Review']            # Create a column called FullReview that combines the title and the review
        hotelName = filename.split(sep='_')
        hotelName = '_'.join(hotelName[3:])
        dataDf['Hotel'] = hotelName                                             # Append Hotel name as another column
        reviewsDF = reviewsDF.append(dataDf)                                    # Append data to reviews Data Frame
    
    # Filter by year    
    reviewsDF.index = pd.to_datetime(reviewsDF['Date']) # Make the index date type
    reviewsDF = reviewsDF[year]                         # Filter data based on year provided
    reviewsDF = reviewsDF.dropna(how='any')
    
    # Get review count for each hotel
    reviewsDF['Hotel'] = reviewsDF['Hotel'].astype('category')  # Convert Hotel names to categorical type
    hotelReviewCount = reviewsDF.groupby('Hotel').size()        # Review count per hotel
    hotelReviewCount = pd.DataFrame({'Hotel':hotelReviewCount.index, 'ReviewCount':hotelReviewCount.values})    # Convert series to dataframe
    return hotelReviewCount,reviewsDF


#------------------------------------------------------------------------------------------------

## Create a stemmed word dictionary for extracting the original word from its root word
def unstem(x, unstemDict):
    stemx = ps.stem(str(x))
    if stemx in unstemDict:
        unstemDict[stemx].append(str(x))
    else:
        unstemDict[stemx] = [str(x)]

unstem = np.vectorize(unstem)


## Vectorized Stemming Function for faster processing
def fastStem(x):
    x = ps.stem(str(x))
    return x

fastStem = np.vectorize(fastStem)


## Vectorized UnStemming Function for faster processing
def fastUnStem(x,unstemDict):
    x = unstemDict[x][0] if x in unstemDict else x
    return x

fastUnStem = np.vectorize(fastUnStem)


#------------------------------------------------------------------------------------------------
## Get length of string (vectorized for an array)
def getLen(x):
    return len(x)

getLen = np.vectorize(getLen)


#------------------------------------------------------------------------------------------------

## Preprocess text data
def preprocessText(txt,onlyChar=True,lower=True,stem = True, stopw = True, minLen = True):
    processTxt = txt
    
    if onlyChar:
        processTxt = re.sub('[^a-zA-Z]', ' ', processTxt)
        
    if lower:
        processTxt = processTxt.lower()
               
    # tokenize sentences
    processTxt = processTxt.split()
    
    if minLen:
        processTxt = np.array(processTxt)
        wordLen = getLen(processTxt)
        processTxt = processTxt[wordLen > 1]
    
    if stopw:
        processTxt = np.array(processTxt)
        stopWords = stopwords.words('english')
        stopWords.remove('not')
        stopWords.extend(['quot','el','il'])
        stopList = np.array(stopWords)
        excludeList = np.in1d(processTxt, stopList)
        processTxt = list(processTxt[~excludeList])
    
    if stem:
        processTxt = np.array(processTxt)
        processTxt = fastStem(processTxt)
        
    processTxt = ' '.join(processTxt)
            
    return processTxt


#------------------------------------------------------------------------------------------------

## Get the most frequent words in a given corpus
def getMostFrequentWords(corpus,unstemDict,minThresh=1):
    mostFreqWordsSeries = pd.Series(' '.join(corpus).split()).value_counts()
    mostFreqWordsSeries = mostFreqWordsSeries[mostFreqWordsSeries.values >= minThresh]
    freqWordsUnstem = [unstemDict[x][0] for x in mostFreqWordsSeries.index if x in unstemDict]
    mostFreqWordsDF = pd.DataFrame({'FreqWord' : freqWordsUnstem,'Count' : mostFreqWordsSeries.values})
    mostFreqWordsDF = mostFreqWordsDF.set_index('FreqWord')
    return mostFreqWordsDF


#------------------------------------------------------------------------------------------------


## For a given word return a list of similar words (appearing in same context)

# Define context function with window size 3
def contextFunc3W(tokens,i):
    window = 3
    left = (tokens[(i-window):i] if (i-window) >= 0 else '*START*')
    right = (tokens[(i+1):(i+window+1)] if (i+window) <= len(tokens) - 1 else '*END*')
    left = ' '.join(left)
    right = ' '.join(right)
    comb = left+' '+right
    return tuple(comb.split())

# Define context function with window size 2
def contextFunc2W(tokens,i):
    window = 2
    left = (tokens[(i-window):i] if (i-window) >= 0 else '*START*')
    right = (tokens[(i+1):(i+window+1)] if (i+window) <= len(tokens) - 1 else '*END*')
    left = ' '.join(left)
    right = ' '.join(right)
    comb = left+' '+right
    return tuple(comb.split())

# filter function
filterFunc = lambda w: len(w) > 2 and w.lower() not in set(stopwords.words('english'))

# Get the list of similar words for a given word
def getSimilarWords(word, reviewContextFull3W, reviewContextFull2W, reviewContextStop2W, numWords=20):
    similarList1 = reviewContextFull3W.similar_words(word)
    similarList2 = reviewContextStop2W.similar_words(word)
    similarList3 = reviewContextFull2W.similar_words(word)    
    combinedList = similarList1
    combinedList.extend(similarList2)
    combinedList.extend(similarList3)
    combinedList = list(OrderedDict.fromkeys(combinedList)) # remove duplicates
    combinedList = [x for x in combinedList if x not in set(stopwords.words('english'))] # remove stopwords
    combinedList = combinedList[0:numWords]
    
    wordScores = reviewContextFull3W.word_similarity_dict(word)
    score1 = np.array([wordScores[w] for w in combinedList])
    wordScores = reviewContextStop2W.word_similarity_dict(word)
    score2 = np.array([wordScores[w] for w in combinedList])
    wordScores = reviewContextFull2W.word_similarity_dict(word)
    score3 = np.array([wordScores[w] for w in combinedList])
    combScore = score1 + score2 + score3
    combScore = combScore/np.sum(combScore)
    
    similarWordsDF = pd.DataFrame({'Word' : combinedList, 'Score': combScore})
    similarWordsDF = similarWordsDF.set_index('Word')
    similarWords = similarWordsDF.to_dict()
    similarWordsDict = [similarWords[di] for di in similarWords][0]
    return similarWordsDict
    

#------------------------------------------------------------------------------------------------

## Find Collocated Words
def getCollocatedWords(BigramFinder,unstemDict,numPairs=20):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    collocations = BigramFinder.score_ngrams(bigram_measures.likelihood_ratio)
    collocations = collocations[:numPairs]
    collocations = [(unstemDict[x[0][0]][0]+'-'+unstemDict[x[0][1]][0],x[1]) for x in collocations]
    collocations = dict(collocations)
    return collocations


#------------------------------------------------------------------------------------------------

## Get Noun Phrases for a given sentence
def getNounPhrase(sentence):
    grammar = "NP: {<DT><JJ>*<NN><NN>*}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(nltk.pos_tag(sentence.split()))
    NP = [res.leaves() for res in result if type(res) != tuple]
    NPlist = []
    if NP:
        for idx in range(0,len(NP)):
            phrase = [phraseList for phraseList in NP[idx]]
            NPlist.append(' '.join([w for w,t in phrase]))
            NPlist = [preprocessText(np,stem=False,stopw=False,minLen=False) for np in NPlist]
    return NPlist


#------------------------------------------------------------------------------------------------

## Extract the contextual sentiment
def getContextualSentiment(review,domainWords):
    reviewLines = tokenize.sent_tokenize(review)
    sentiAnalyze = SentimentIntensityAnalyzer()
    sentimentDict = {}
    
    # Get Sentiment for each review sentence
    for sentence in reviewLines:
        ss = sentiAnalyze.polarity_scores(sentence)
        if ss['compound'] < -0.2:
            rating = 'negative'
        elif ss['compound'] >= -0.0 and ss['compound'] <= 0.5:
            rating = 'neutral'
        else:
            rating = 'positive'
        
        # Get the noun phrases in the sentence
        NP = getNounPhrase(sentence)
        nounList = []
        nounList.extend(NP)
                
        # Get the domin words present in the sentence (nouns related to domain)
        posTags = nltk.pos_tag(sentence.split())
        domainword = [w for w,t in posTags if t=='NN' and w in domainWords and w not in set(stopwords.words('english'))]
        if not domainword:
            domainword = [w for w,t in posTags if t=='NN']       
        
        # Remove words already present in noun phrases
        for npw in NP:
            domainword = [re.sub('[^a-zA-Z]', ' ', w).strip() for w in domainword if npw.rfind(w) == -1]
        
        # Create dictionary of rating for each sentence represented as domain context/words/noun phrases
        nounList.extend(domainword)
        nounList = list(OrderedDict.fromkeys(nounList))
        sentimentDict[','.join(nounList)] = rating
        
    # Aggregate the rating for each context separately
    sentimentDictFinal = {}
    sentimentDictFinalKeys = []
    sentimentDictFinalKeysRaw = []
    for key in sentimentDict.keys():
        if key:
            for word in key.split(sep=','):                
                word = preprocessText(word,stem=False,stopw=False,minLen=False)
                if word.rfind(' ') == -1:                                        
                    nounWord = word
                    nounWord = ps.stem(nounWord)
                else:
                    nounWord = nltk.pos_tag(word.split())
                    nounWord = [w for w,t in nounWord if t == 'NN']
                    if not nounWord:
                        continue
                    else:
                        if len(nounWord) > 1:
                            nounWord = nounWord[0]
                        nounWord = ps.stem(nounWord[0])
                       
                if nounWord in sentimentDictFinalKeys:
                    nounIdx = sentimentDictFinalKeys.index(nounWord)
                    keyWord = sentimentDictFinalKeysRaw[nounIdx]                    
                    rating = sentimentDictFinal[keyWord]
                    if rating == 'neutral': # Update to new rating if earlier rating is neutral
                        sentimentDictFinal[keyWord] = sentimentDict[key]
                else:
                    sentimentDictFinal[word] = sentimentDict[key]
                    sentimentDictFinalKeys.append(nounWord)
                    sentimentDictFinalKeysRaw.append(word)  

    # Reconstruct the sentiment dictionary with non-stemmed keys  
    sentimentDictFinal = dict(zip(sentimentDictFinalKeysRaw, sentimentDictFinal.values()))
    
    # Wrap the sentiments in HTML for output

    htmlStrList = []
    htmlStr = ''
    for subject in sentimentDictFinal:
        htmlStr = htmlStr + '<p><b>' + str(subject) + '</b>'
        senti = sentimentDictFinal[subject]
        if senti == 'positive':
            htmlStr = htmlStr + '<img src="www/positive.jpg" alt="Positive" width="70" height="70"></p></br><hr>'
        elif senti == 'neutral':
            htmlStr = htmlStr + '<img src="www/neutral.jpg" alt="Neutral" width="70" height="70"></p></br><hr>'
        else:
            htmlStr = htmlStr + '<img src="www/negative.jpg" alt="Negative" width="70" height="70"></p></br><hr>'
        htmlStrList.append(htmlStr)
    
                                                            
    return htmlStr,sentimentDictFinal


#------------------------------------------------------------------------------------------------

# Get the sentiment for the title of the review
def getTitleSentiment(title):
    sentiAnalyze = SentimentIntensityAnalyzer()
    ss = sentiAnalyze.polarity_scores(title)
    sentiment = ''
    if ss['pos'] >= ss['neg']:
        sentiment = 'positive'
    else:
        sentiment = 'negative'
        
    return sentiment
    
#------------------------------------------------------------------------------------------------

## Get Cluster of similar words and assign a topic/theme to them
def getThemeClusters(corpus,mostFreqWords,unstemDict,numClusters):
    reviewContextFull3W = ContextIndex(tokens=corpus.split(),context_func=contextFunc3W)
    freqWords = np.array(list(mostFreqWords.index))
    freqStem = fastStem(freqWords)
    freqStem = list(freqStem)
    wordContextList = [' '.join([re.sub('[^a-zA-Z]', ' ', str(t)) for t in list(reviewContextFull3W._word_to_contexts[w])]) for w in freqStem]
    vectorizer = TfidfVectorizer(min_df=5, max_features = 10000)
    tfidfFit = vectorizer.fit_transform(wordContextList)
    
    kmeansModel = MiniBatchKMeans(n_clusters=numClusters, init='k-means++', n_init=1, init_size=10000, batch_size=10000, verbose=False, max_iter=1000)    
    kmeansFit = kmeansModel.fit(tfidfFit)
    kmeansClusters = kmeansModel.predict(tfidfFit)
    kmeansDistances = kmeansModel.transform(tfidfFit)    
    sortedCentroids = kmeansFit.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    terms = np.array(terms)

    hyper = lambda s: s.hypernyms()
    wnSynsetExcludeList = ['artifact','physical','entity','structure','object','whole']
    clusterVec = np.unique(kmeansClusters)
    clusterDict = {}
    for clusterVal in clusterVec:
        clusterWords = terms[sortedCentroids[clusterVal, ]]
        clusterWords = fastUnStem(clusterWords,unstemDict)
        clusterIdx = kmeansClusters == clusterVal
        clusterWordsFreq = freqWords[clusterIdx]
        clusterWords = [w for w in clusterWords if w in clusterWordsFreq]
        clusterWordCount = mostFreqWords.ix[clusterWords[:20]]
        clusterWordCount = clusterWordCount.sort_values(by="Count",ascending=False)
           
        synsetList = []
        for word in clusterWordCount.index:            
            wsn = wn.synsets(word)
            if wsn:
                wordSynset = wn.synset(wsn[0].name())
                synsetList.append(wordSynset.lemma_names()[0])
                hypoSets = list(wordSynset.closure(hyper))
                hypoNames = [hyponame.lemma_names()[0] for hyponame in hypoSets]
                hypoNames = hypoNames[:-4]
                hypoNames = [hn for hn in hypoNames if hn.strip() not in wnSynsetExcludeList]
                synsetList.extend(hypoNames) # eliminate last 4 hyponames as they are very generic

        if synsetList:
            fdist = FreqDist(synsetList)
            clusterTheme = fdist.most_common(1)[0][0] if type(fdist.most_common(1)) == list else fdist.most_common(1)[0]
            clusterDict[clusterTheme] = clusterWords[:20]
        else:
             clusterDict[clusterWords[0]] = clusterWords[:20]
 
    return clusterDict
