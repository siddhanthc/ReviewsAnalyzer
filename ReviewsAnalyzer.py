"""
Testing script for reviews (make final later)

author: Siddhanth
"""

# Load necessary libraries
import numpy as np
import nltk

# Load necessary functions
import helperFunctions as hf

# Read Hotel Reviews Data into a data frame
dataPath = 'OpinRankDataSet/hotels/'
city = 'chicago'
year = '2007'
hotelReviewCount, reviewsDF = hf.readReviewsData(dataPath, city, year)
print('Done reading data')

# Create combined reviews corpus from all reviews for different applications
combinedCorpus = reviewsDF['FullReview'].str.cat(sep=' ')
corpusOnlyChar = hf.preprocessText(combinedCorpus,onlyChar=True,lower=True,stopw=False,stem=False)
corpusNoStop   = hf.preprocessText(corpusOnlyChar,onlyChar=False,lower=False,stopw=True,stem=False)
corpusStem     = hf.preprocessText(corpusNoStop,onlyChar=False,lower=False,stopw=False,stem=True)
print('Done corpus processing')

# Create the unstem dictionary for the corpus
unstemDict = {}
corpusTrim = np.array(corpusNoStop.split())
hf.unstem(corpusTrim,unstemDict)
corpusTrim = list(corpusTrim)
print('Done creating unstem dictionary')

# Extract the most frequent words (excluding stopwords)
cleanReview = reviewsDF['FullReview'].apply(hf.preprocessText)
mostFreqWords = hf.getMostFrequentWords(cleanReview,unstemDict,5)
print('Done extracting most frequent words')

# Find Similar Words (appearing in similar contexts)
from nltk import ContextIndex
corpusOnlyChar = hf.preprocessText(combinedCorpus,onlyChar=True,lower=True,stopw=False,stem=False)
reviewContextFull3W = ContextIndex(tokens=corpusOnlyChar.split(),context_func=hf.contextFunc3W)
reviewContextFull2W = ContextIndex(tokens=corpusOnlyChar.split(),context_func=hf.contextFunc2W)
reviewContextStop2W = ContextIndex(tokens=corpusNoStop.split(),context_func=hf.contextFunc2W)
word = 'suite'
similarWords = hf.getSimilarWords(word, reviewContextFull3W, reviewContextFull2W, reviewContextStop2W, numWords=20)
print('Done finding similar words')

# Find Collocated Words (words appearing together in a phrase)
from nltk.collocations import BigramCollocationFinder
windowSize = 3
finder = BigramCollocationFinder.from_words(corpusStem.split(), windowSize)
collocationWords = hf.getCollocatedWords(finder,unstemDict,numPairs=10)
print('Done Finding Collocated words')

# Find Contextual Sentiments
reviewsDF['cleanReview'] = reviewsDF['FullReview'].apply(hf.preprocessText)
mostFreqWords = hf.getMostFrequentWords(reviewsDF['cleanReview'],unstemDict,5)
posTaggedWords = nltk.pos_tag(list(mostFreqWords.index))
hotelWords = [w for w,tag in posTaggedWords if tag == 'NN']
htmlStr, reviewSentiments = hf.getContextualSentiment(reviewsDF['FullReview'][1], domainWords = hotelWords)
print('Done building sentiment analyzer')

# Get similar word clusters along with topic
cleanReview = reviewsDF['FullReview'].apply(hf.preprocessText,stopw=True,minLen = False)
cleanReview = cleanReview.str.cat(sep=' ')
maxClusters = 10
clusterDict = hf.getThemeClusters(cleanReview,mostFreqWords,unstemDict,maxClusters)
print('Done clustering similar terms and assigning topic')


htmlStr = ''
for i in range(10):
    htmlStr = htmlStr + ' hello'
    
    
