
# coding: utf-8

# # Reviews Analyzer

# The objective of this analysis is threefold:
# 1. Word association identification
# 2. Contextual sentiment extraction
# 3. Theme/topic assignment to word clusters
# 
# The dataset used for the purpose of this analysis consists of [hotel reviews](https://code.google.com/archive/p/dataset/downloads) mined from [Tripadvisor.com](http://tripadvisor.com)

# ### Lets load the data
# Hotel reviews are available for the cities of Beijing, Chicago, Dubai, Las-Vegas, London, Montreal, New-Delhi, New-York-City, San-Francisco and Shanghai spanning the years 2003 - 2010. For the purpose of this analysis, the city has been set to *chicago* and year to *2007*. These settings however **can** be changed. 

# In[1]:


# Load necessary libraries
import numpy as np
import nltk
from nltk import ContextIndex
from nltk.collocations import BigramCollocationFinder
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.core.display import display_html


# Load custom module
import helperFunctions as hf

# Read Hotel Reviews Data into a data frame
dataPath = 'OpinRankDataSet/hotels/'
city = 'chicago'
year = '2007'
hotelReviewCount, reviewsDF = hf.readReviewsData(dataPath, city, year)
print('Done reading data')


# ### Clean up the text
# Cleaning mainly involves removal of noisy characters, stopwords, case normalization and stemming. A mapping is also created between words and their stemmed forms, so that extracting the actual word is possible.

# In[2]:


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
print('Done creating unstem dictionary')


# ### A quick look at the most frequent terms in the corpus

# In[3]:


# Extract the most frequent words (excluding stopwords)
cleanReview = reviewsDF['FullReview'].apply(hf.preprocessText)
mostFreqWords = hf.getMostFrequentWords(cleanReview,unstemDict,5)

print('Here is a list of the most frequently words in the corpus:')
freqWordsDict = mostFreqWords[:20].to_dict()['Count']
freqCloud = WordCloud(background_color='white',
                      width=1500,
                      height=1200,
                      scale = 1,
                      random_state=10,
                      min_font_size=6).generate_from_frequencies(freqWordsDict)
plt.figure(figsize=(10,10))
plt.imshow(freqCloud)
plt.axis('off')
plt.show()


# ### Word Association
# 
# Although word association has many connotations, it can be broadly classified into two types:
# 1. **Paradigmatic Association**: These words are associated because they appear in a similar context. Paradigmatically associated words can be used interchangeably in a sentence without affecting its semantics. Eg: "I bumped into my friend while on my way to work". Here *work* can be replaced with *college* without affecting the semantic correctness of the sentence.
# 2. **Syntagmatic Association**: These words are associated because they are typically collocated. Syntagmatically associated words often occur together in a sentence. Eg: "The person at the front desk assited me". The words *front desk* are collocated and often occur together.
# 
# Both these associations have been mined for in the reviews.

# #### Paradigmatic Association
# Paradigmatic associations have been mined by looking at the context around words for window size upto 3. Results obtained by including and excluding stopwords have been combined (stopwords like prepositions and determinants can help while looking at context).
# 
# Here is an example for paradigmatically associated words in the corpus. The example word **breakfast** can be replaced with another word to finds its associations in the corpus

# In[4]:


# Find Similar Words (appearing in similar contexts)
reviewContextFull3W = ContextIndex(tokens=corpusOnlyChar.split(),context_func=hf.contextFunc3W)
reviewContextFull2W = ContextIndex(tokens=corpusOnlyChar.split(),context_func=hf.contextFunc2W)
reviewContextStop2W = ContextIndex(tokens=corpusNoStop.split(),context_func=hf.contextFunc2W)

word = 'breakfast' #<------------change word here to see other associations
similarWords = hf.getSimilarWords(word, reviewContextFull3W, reviewContextFull2W, reviewContextStop2W, numWords=20)

print('Here is a list of words associated with '+ word)
similarWordCloud = WordCloud(background_color='white',
                      width=1500,
                      height=1200,
                      scale = 1,
                      random_state=10,
                      min_font_size=6).generate_from_frequencies(similarWords)
plt.figure(figsize=(10,10))
plt.imshow(similarWordCloud)
plt.axis('off')
plt.show()


# #### Syntagmatic Association
# Syntagmatic associations have been mined by looking at how frequently do bi-grams occur in the corpus. The bi-grams need not be coniguous. Stop words have been ignored.
# 
# Here is an example for syntagmatically associated words in the corpus. The top ten most frequently occurring bi-grams are shown.

# In[5]:


# Find Collocated Words (words appearing together in a phrase)
windowSize = 3 # Increase this parameter to discover more spaced out bi-grams. Typically syntagmatic words lie in close proximity
finder = BigramCollocationFinder.from_words(corpusStem.split(), windowSize)
collocationWords = hf.getCollocatedWords(finder,unstemDict,numPairs=10)

print('Here is a list of top frequently occurring collocated words')
collocationWordCloud = WordCloud(background_color='white',
                      width=1500,
                      height=1200,
                      scale = 1,
                      random_state=10,
                      min_font_size=6).generate_from_frequencies(collocationWords)
plt.figure(figsize=(10,10))
plt.imshow(collocationWordCloud)
plt.axis('off')
plt.show()


# ### Contextual Sentiment Analysis
# Contextual sentiments are typically extracted from smaller phrases which could have either a postive, negative or neutral sentiment. So, there could be many different sentiments expressed in the same paragraph of text. From a *reviews* perspective, it make sense to look at what aspect of the product/service was the consumer happy/unhappy about. That is exactly what is being done in this analysis.
# 
# 
# Since the dataset at our disposal is not labelled for sentiment, a rule-based classifier *(VADER-Valence Aware Dictionary and sEntiment Reasoner)* is used to extract sentiments from each sentence. Once the sentiment is known, the *object* that the sentiment is referring to is identified by looking for nouns and noun phrases. This is further filtered by comparing with a list of frequently occuring noun words in the corpus that could be associated with either a product or service. The sentiments are then aggregated from the entire review.
# 
# Have a look!

# In[6]:


# Find Contextual Sentiments
cleanReview = reviewsDF['FullReview'].apply(hf.preprocessText)

# Get a list of frequent terms that could be associated with the domains (in this case 'hotel') product/service
mostFreqWords = hf.getMostFrequentWords(cleanReview,unstemDict,minThresh=5)
posTaggedWords = nltk.pos_tag(list(mostFreqWords.index))
hotelWords = [w for w,tag in posTaggedWords if tag == 'NN']

# Select a review and identify the associated sentiments
REVIEW = reviewsDF['FullReview'][80] #<---------------- Pick another review by chaning the index value
sentimentHtml,reviewSentiments = hf.getContextualSentiment(REVIEW, domainWords = hotelWords)
print('Sentiments expressed in this review:\n' + REVIEW)
display_html(sentimentHtml,raw=True)


# ### Theme/Topic Identifcation
# The objective here is to assign a theme to a cluster of associated words extracted from the corpus. The words could be describing a related phenomenon. There are two steps here: 
# 1. To identify clusters of associated terms. This is done by first building a TFIDF matrix of frequently occuring terms and the contexts in which they occur. K-means is then used to partition these terms into clusters that are homogenous.
# 2. Once the clusters of terms are obtained, the theme/subject/topic that each term in the cluster could be referring to is identified. This is done with the help of the WordNet lexicon. For each term all the hyponyms (parent terms from which the term is derived) are identified. Then the parent term that is most common across all the terms in the cluster is chosen as the theme for the cluster. This seems to work pretty well.
# 
# Lets have a look at the clusters and what themes they are ascribing to.

# In[7]:


# Get similar word clusters along with topic
cleanReview = reviewsDF['FullReview'].apply(hf.preprocessText,stopw=True,minLen = True)
cleanReview = cleanReview.str.cat(sep=' ')
maxClusters = 10
clusterDict = hf.getThemeClusters(cleanReview,mostFreqWords,unstemDict,maxClusters)

# Modify cluster dict for plotting
clusterList = []
nameList = []
for dic in clusterDict:
    nameList.append(dic)
    themeList = [(dic,5)]
    termList = [(term,2.5) for term in clusterDict[dic]]
    themeList.extend(termList)
    themeDict = dict(themeList)
    clusterList.append(themeDict)
    
print('Here are the theme-based clusters')
numClusters = len(clusterList)
f, sbplts = plt.subplots(numClusters,figsize=(20,50))
for idx in range(numClusters):
    termCloud = WordCloud(background_color='white',scale=1,relative_scaling=0.9,width=1500,height=1200,random_state=10,min_font_size=10).generate_from_frequencies(clusterList[idx])
    sbplts[idx].imshow(termCloud)
    sbplts[idx].axis('off')
    sbplts[idx].set_title(nameList[idx])
plt.show()


# 
# 
# 
# 
# 
# # Hotel Comparisons
# We can use the techniques from above to compare trends across different hotels. Lets pick two hotels from Chicago which have almost the same number of reviews:
# 1. Congress Plaza Hotel
# 2. Sheraton Chicago Hotel and Towers
# 
# 

# In[8]:


# Extract data for the two hotels
hotelCongress = 'congress_plaza_hotel'
hotelSheraton = 'sheraton_chicago_hotel_and_towers'

hotelCongressDF = reviewsDF[reviewsDF['Hotel'] == hotelCongress]
hotelSheratonDF = reviewsDF[reviewsDF['Hotel'] == hotelSheraton]


# ### Most frequent words
# Lets have a look at the most used words in the reviews of the two hotels.

# In[9]:


# Extract the most frequent words for hotelCongress
cleanReviewCongress = hotelCongressDF['FullReview'].apply(hf.preprocessText)
mostFreqWordsCongress = hf.getMostFrequentWords(cleanReviewCongress,unstemDict,5)

f,subplts = plt.subplots(2,figsize=(10,10))

freqWordsDict = mostFreqWordsCongress[10:30].to_dict()['Count']
freqCloud = WordCloud(background_color='white',
                      width=1500,
                      height=1200,
                      scale = 1,
                      random_state=10,
                      min_font_size=6).generate_from_frequencies(freqWordsDict)
subplts[0].imshow(freqCloud)
subplts[0].axis('off')
subplts[0].set_title('Common words used in reviews for Hotel Congress')


# Extract the most frequent words for hotelSheraton
cleanReviewSheraton = hotelSheratonDF['FullReview'].apply(hf.preprocessText)
mostFreqWordsSheraton = hf.getMostFrequentWords(cleanReviewSheraton,unstemDict,5)

freqWordsDict = mostFreqWordsSheraton[10:30].to_dict()['Count']
freqCloud = WordCloud(background_color='white',
                      width=1500,
                      height=1200,
                      scale = 1,
                      random_state=10,
                      min_font_size=6).generate_from_frequencies(freqWordsDict)
subplts[1].imshow(freqCloud)
subplts[1].axis('off')
subplts[1].set_title('Common words used in reviews for Hotel Sheraton')

plt.show()


# Congress is being labeled as *old* whereas Sheraton promises a good *view*.

# ### Collocation
# Lets also have a look at some of the collocated words pairs from both reviews of both hotels

# In[10]:


# Extract Collocated Words for Hotel Congress
congRev = ' '.join(cleanReviewCongress)
windowSize = 3 
finder = BigramCollocationFinder.from_words(congRev.split(), windowSize)
collocationWords = hf.getCollocatedWords(finder,unstemDict,numPairs=10)

f,subplts = plt.subplots(2,figsize=(10,10))

collocationWordCloud = WordCloud(background_color='white',
                      width=1500,
                      height=1200,
                      scale = 1,
                      random_state=10,
                      min_font_size=6).generate_from_frequencies(collocationWords)
subplts[0].imshow(collocationWordCloud)
subplts[0].axis('off')
subplts[0].set_title('Collocated words used in reviews for Hotel Congress')


# Extract collocated words for Hotel Sheraton
sherRev = ' '.join(cleanReviewSheraton)
windowSize = 3 
finder = BigramCollocationFinder.from_words(sherRev.split(), windowSize)
collocationWords = hf.getCollocatedWords(finder,unstemDict,numPairs=10)

collocationWordCloud = WordCloud(background_color='white',
                      width=1500,
                      height=1200,
                      scale = 1,
                      random_state=10,
                      min_font_size=6).generate_from_frequencies(collocationWords)
subplts[1].imshow(collocationWordCloud)
subplts[1].axis('off')
subplts[1].set_title('Collocated words used in reviews for Hotel Sheraton')

plt.show()


# Both these hotels seem to have a good proximity to *Michigan Avenue*. Also, Sheraton visitors seem to frequent *Navy Pier*.

# ### Sentiment Analysis
# Lets have a quick look at the positive/negative sentiments mentioned in a few sample reviews from both these hotels

# In[11]:


# Find Contextual Sentiments for Hotel Congress

# Select a few reviews and identify the associated sentiments
REVIEW = hotelCongressDF['FullReview'][:5] #<---------------- Pick another review by chaning the index value
REVIEW = ' '.join(REVIEW)
sentimentHtml,reviewSentiments = hf.getContextualSentiment(REVIEW, domainWords = hotelWords)
display_html('<h1>Hotel Congress</h1>'+sentimentHtml,raw=True)


# Find Contextual Sentiments for Hotel Sheraton

# Select a few reviews and identify the associated sentiments
REVIEW = hotelSheratonDF['FullReview'][:5] #<---------------- Pick another review by chaning the index value
REVIEW = ' '.join(REVIEW)
sentimentHtml,reviewSentiments = hf.getContextualSentiment(REVIEW, domainWords = hotelWords)
display_html('<h1>Hotel Sheraton</h1>'+sentimentHtml,raw=True)


# Hotel Sheraton definitely seems to have more positive sentiments as compared to Hotel Congress in the few reviews that we compared.

# ### Themes in the Hotel Reviews
# Finally, lets see what themes emerge in the two different hotels

# In[13]:


# Get similar word clusters along with topic for Hotel Congress
cleanReview = cleanReviewCongress.str.cat(sep=' ')
maxClusters = 7
clusterDict = hf.getThemeClusters(cleanReview,mostFreqWords,unstemDict,maxClusters)

# Modify cluster dict for plotting
clusterList = []
nameList = []
for dic in clusterDict:
    nameList.append(dic)
    themeList = [(dic,5)]
    termList = [(term,2.5) for term in clusterDict[dic]]
    themeList.extend(termList)
    themeDict = dict(themeList)
    clusterList.append(themeDict)
    
print('Here are the theme-based clusters for Hotel Congress')
numClusters = len(clusterList)
f, sbplts = plt.subplots(numClusters,figsize=(10,30))
for idx in range(numClusters):
    termCloud = WordCloud(background_color='white',scale=1,relative_scaling=0.9,width=1500,height=1200,random_state=10,min_font_size=10).generate_from_frequencies(clusterList[idx])
    sbplts[idx].imshow(termCloud)
    sbplts[idx].axis('off')
    sbplts[idx].set_title(nameList[idx])
plt.show()


# In[14]:


# Get similar word clusters along with topic for Hotel Sheraton
cleanReview = cleanReviewSheraton.str.cat(sep=' ')
maxClusters = 7
clusterDict = hf.getThemeClusters(cleanReview,mostFreqWords,unstemDict,maxClusters)

# Modify cluster dict for plotting
clusterList = []
nameList = []
for dic in clusterDict:
    nameList.append(dic)
    themeList = [(dic,5)]
    termList = [(term,2.5) for term in clusterDict[dic]]
    themeList.extend(termList)
    themeDict = dict(themeList)
    clusterList.append(themeDict)
    
print('Here are the theme-based clusters for Hotel Sheraton')
numClusters = len(clusterList)
f, sbplts = plt.subplots(numClusters,figsize=(10,30))
for idx in range(numClusters):
    termCloud = WordCloud(background_color='white',scale=1,relative_scaling=0.9,width=1500,height=1200,random_state=10,min_font_size=10).generate_from_frequencies(clusterList[idx])
    sbplts[idx].imshow(termCloud)
    sbplts[idx].axis('off')
    sbplts[idx].set_title(nameList[idx])
plt.show()


# The clusters being produced are non-deterministic, since kmeans is resampling in its various iterations. A clearer trend for clusters was visible when the entire corpus was considered, which isnt emerging for the smaller datasets of the two hotels. We could do with more data for these hotels, instead of just focussing on a particular year.

# # Conclusion
# The techniques developed for this analysis are **domain agnostic** and could well be applied to reviews from any other domain (eg: mobile phones). These algorithms could be combined effectively for trend analysis across different segments and time as well. The positive/negative ratings of hotels could be analyzed with respect to time. In the case of hotels, key interests of travelers from different cities could be picked up. We could also look at what differenciates certain hotels from others even they though belong to the same budget segment, and discover many other such insights.