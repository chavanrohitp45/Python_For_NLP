# -*- coding: utf-8 -*-
"""
@author: Rohit Chavan
"""

##########################Text Mining###############################
sentence="We are Learning Textmining from Sanjivani AI"
##If we want to know the position
sentence.index("Learning")  
#op=7
##It will show lwarning is at 7 position
##This is going to show character position from 0 including
 
####################################################################
#We want to know position textMinning word
sentence.split().index("Textmining") 
#op=3
##It will split the words in list and count the position
#If you want to see the list select sentence.split() and
##It will show 3
###################################################
#Suppose we want to print any word in reverse order
sentence.split()[2][::-1]  
###[start:end end:-1(start)] it will start-1,-2,-3 till the end
##learnng will be printed as gninraeL
##############################################
#suppose we want to print first and last word of the sentence
words=sentence.split()  
# Tokenization: It will prepare the list of words by spliting the sentence 
first_word=words[0]  
first_word
last_word=words[-1]
last_word  
##now we want to conat the first and last word
cancat_word=first_word+" "+last_word
cancat_word
#####################################################################
#we want to print even words from sentences
#use list comprehension
[words[i] for i in range(len(words))if i%2==0] 

#here same way but combined in one sentence
for i in range(len(words)):
    list1=[]
    if i%2==0:
      list1.append(words[i])
      list1
##words having odd length will not be printed
################################################
sentence 
#now we want to display only AI
sentence[-3:]#AI 
sentence[:-3]#'We are Learning Textmining from Sanjivani'
##Suppose we want to display entire sentence into reverse order
sentence[::-1] 
#IA inavijnaS morf gninimtxeT gninraeL era eW 
########################################
#suppose we want to select each word and print in reversed order
words
print(" ".join(word[::-1]for word in words))
##eW era gninraeL gninimtxeT morf inavijnaS IA
#####################################################################
#tokonization
import nltk 
nltk.download('punkt')
from nltk import word_tokenize
words=word_tokenize("I am reading NLP fundamentals")
print(words)
#o/p=['I', 'am', 'reading', 'NLP', 'fundamentals']
####################################################################

#parts of speech(pos) tagging
nltk.download('averaged_perceptron_tagger') 
nltk.pos_tag(words) 
#It is going mention parts os speech
'''
PRP: Personal pronoun. Examples include "I", "he", "she", "we".
VBP: Verb, non-3rd person singular present. Examples include "am", "are", "do".
VBG: Verb, gerund or present participle. Examples include "reading", "swimming", "running".
NNP: Proper noun, singular. Examples include "NLP", "John", "London".
NNS: Noun, plural. Examples include "fundamentals", "dogs", "cars".

I: PRP (Personal pronoun)
am: VBP (Verb, non-3rd person singular present)
reading: VBG (Verb, gerund or present participle)
NLP: NNP (Proper noun, singular)
fundamentals: NNS (Noun, plural)
'''
###################################################################
'''
CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: “there is” … think of it like “there exists”)
FW foreign word
IN preposition/subordinating conjunction
JJ adjective ‘big’
JJR adjective, comparative ‘bigger’
JJS adjective, superlative ‘biggest’
LS list marker 1)
MD modal could, will
NN noun, singular ‘desk’
NNS noun plural ‘desks’
NNP proper noun, singular ‘Harrison’
NNPS proper noun, plural ‘Americans’
PDT predeterminer ‘all the kids’
POS possessive ending parent’s
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO, to go ‘to’ the store.
UH interjection, errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
'''
####################################################################
#stop words from NLTK library
from nltk.corpus import stopwords
stop_words=stopwords.words('English')
#you can verify 179 stop words in variable explorer
print(stop_words)
sentence1="I am learning NLP:It is one of the most popular library in python"
#first we will tokenize the sentence
sentence_words=word_tokenize(sentence1)
print(sentence_words)
#now let us filter the sentence1 using stop words
sentence_no_stops=" ".join([words for words in sentence_words if words not in stop_words])
print(sentence_no_stops)
sentence1 
######################################################################
#suppose we want to repalce words in string
sentence2="I visited MY from IND on 14-02-19"
normalized_sentence=sentence2.replace("MY","Malaysia").replace("IND","India")
normalized_sentence=normalized_sentence.replace("-19","-2020")
print(normalized_sentence)

#####################################################################

#suppose we want auto correction in the sentence
from autocorrect import Speller
#declare the function speller define for english
spell=Speller(lang='en')
spell("English")

spell("Engiilish")
########################################################################
import nltk
nltk.download('punkt')
from nltk import word_tokenize
#suppose we want to correct whole sentence
sentence3="Ntural lanagage processin deals within the aart of extracting sentiiiments"
#let us first tokenize this sentence
sentence3=word_tokenize(sentence3)
corrected_sentence=" ".join([spell(word) for word in sentence3])
print(corrected_sentence)

########################################################################
#Stemming 
import nltk
nltk.download('punkt') 
  
stemmer=nltk.stem.PorterStemmer()
stemmer.stem("programming")
stemmer.stem("programmed")
stemmer.stem("Jumping")
stemmer.stem("Jumped")
 
#########################################################################

#Lematizer
#lematizer looks into dictionary words
import nltk
nltk.download("wordnet")

nltk.download('omw-1.4')

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()  
lemmatizer.lemmatize("programed")
lemmatizer.lemmatize("programs")
lemmatizer.lemmatize("battling")
lemmatizer.lemmatize("amazing")

#########################################################################
#chunking (shallow parsing) Identifying named entities
import nltk
nltk.download("maxent_ne_chunker")
nltk.download('words')
sentence4="We are learning NLP in python by SanjivaniAI by"
##first we will tokenize
words=word_tokenize(sentence4) 
words=nltk.pos_tag(words)
i=nltk.ne_chunk(words,binary=True)
[a for a in i if len(a)==1]
#################################################################

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk


nltk.download("maxent_ne_chunker")
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


sentence4 = "We are learning NLP in python by SanjivaniAI"

# Tokenize the sentence
words = word_tokenize(sentence4)

# Part-of-Speech tagging
pos_tags = pos_tag(words)

# Named Entity Recognition
named_entities = ne_chunk(pos_tags, binary=True)

# Filter the results to get single-token named entities
single_token_entities = [chunk for chunk in named_entities if hasattr(chunk, 'label') and len(chunk) == 1]

# Print the single-token named entities
for entity in single_token_entities:
    print(entity)

##########################################################################

#senetence tokenization
from nltk.tokenize import sent_tokenize
sent=sent_tokenize("we are learning NLP in python.Delivered by SanjivaniAI")
sent

print(sent)
############################################################################

from nltk.wsd import lesk
sentence1="keep your savings in the bank"
print(lesk(word_tokenize(sentence1),'bank'))
#synset('savings_bank.n.02')
sentence2="It is so risky to drive over the the banks of river "
print(lesk(word_tokenize(sentence2),'bank'))
#Synset('bank.v.07')

#Synset ('bank.n.07') a slope in the turn of a road or track;
#the outside is higher than the inside in order to reduce the 

#"bank" as multiple meanings if you want to find exact meaning
#execute following code

#the definition for "bank" can be seen here:

from nltk.corpus import wordnet as wn
from ss in wn.synsets('bank'):print(ss,ss.definition())


########################################################################

""" Regex 101 """

import re
sentence5= "sharat twitted ,Wittnessing 70th republic day India from Rajpath,now Delhi mesmorizing  performance by Indian Army: "            
re.sub(r'([^\s\w]|_)+',' ',sentence5).split()

#Extracting N-grams
#n-grams can be extracted using three techniques
#1.custom defined functions
#2.NLTK
#3.textBlob

##################################################################

#extracting n_grams using custom defined function

import re
def n_gram_extractor(input_str, n):
    tokens = re.sub(r'([^\s\w]|_)+', ' ',input_str).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])

n_gram_extractor("the cute little boy is playing with kitten",2)
n_gram_extractor("the cute little boy is playing with kitten",3)

#################################################################

from nltk import ngrams
#extracting n grams with nltk
list(ngrams("the cute little boy is playing with kitten".split(),2))
list(ngrams("the cute little boy is playing with kitten".split(),3))

###################################################################
#pip install textblob 
from textblob import TextBlob
blob=TextBlob("the cute little boy is playing with kitten")
blob.ngrams(n=2)
blob.ngrams(n=3)
##################################################################

#Tokenization using keras,
#pip install keras
#pip install textblob
sentence5
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(sentence5)

######################## TYPES OF TOKENIZER ##################################

#Tokenization using textBlob
from textblob import TextBlob
blob =TextBlob(sentence5)
blob.words

#################################################################

#Tweet Tokenizer
from nltk.tokenize import TweetTokenizer
tweet_tokenizer=TweetTokenizer()
tweet_tokenizer.tokenize(sentence5)

#################################################################

#Multi_word_expresstion
from nltk.tokenize import MWETokenizer
sentence5
mwe_tokenizer=MWETokenizer([('republic','day')])
mwe_tokenizer.tokenize(sentence5.split())
mwe_tokenizer.tokenize(sentence5.replace('!',' ').split())

###################################################################

#Regular expression Tokenizer
from nltk.tokenize import RegexpTokenizer
reg_tokenizer=RegexpTokenizer('\w+|\$[\d\.]+|\s+')
reg_tokenizer.tokenize(sentence5)

###################################################################

#white space Tokenizer

from nltk.tokenize import WhitespaceTokenizer
wh_tokenizer =WhitespaceTokenizer()
wh_tokenizer.tokenize(sentence5)

#####################################################################

#Word Punct Tokenizer
from nltk.tokenize import WordPunctTokenizer
wp_tokenizer=WordPunctTokenizer()
wp_tokenizer.tokenize(sentence5)

#####################################################################

#stemmer
sentence6="I love playing cricket.Cricket players practices hard in there inning "
from nltk.stem import RegexpStemmer
regex_stemmer = RegexpStemmer('ing$')
' '.join(regex_stemmer.stem(wd) for wd in sentence6.split())

########################################################################

sentence7-"Before eating , it would be nice to sanitize your hand with a sanitizer "
from nltk.porter import PorterStemmer
ps_stemmer=PorterStemmer()
words=sentence7.split()
" ".join([ps_stemmer.stem(wd) for wd in words])

########################################################################

#Lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()
sentence8="The codes executed today are far better than what are execute generally"
words=word_tokenize(sentence8)
" ".join([lemmatizer.lemmatize(word) for word in words])

##########################################################################

#singularize and pluaralization

from textblob import TextBlobsentence9=TextBlob("she shell seashells on the seashore")
words=sentence9.words
#we want to make word[2] i.e seashells in singular form
sentence9.words[2].singularize()
#we want word 5 i.e seashore in plural form
sentence9.words[5].pluralize()

########################################################################
#language translation from spanish to English
from textblob import TextBlob
en_blob=TextBlob(u'muy bien')
en_blob.translate(from_lang='es',to='en')
#es:spanish  en:English

#######################################################################
#custom stopwords removal
from nltk import word_tokenize
sentence9="she sells seashells on the seashore"
custom_stop_word_list=['she','on','the','am','is']
words=word_tokenize(sentence9)
" ".join([word for word in words if word.lower() not in custom_stop_word_list])

#select words which are not in defined list

#extracting general feature from raw text number of words
#detect presence of wh word polarity,subjectivity,language identification
#to identify the number of words
import panads as pd
df=pd.DataFrame([['the vaccine for covid-19 will be announced on 1st August'],['Do you know how much expection the world population is having from this research?'],['the risk of vie=rus will come to an end on 31st July']])
df.columns=['text']
df

#now let us measure the number of words
from textblob import TextBlob
df['number_of_words ']=df['text'].apply(lambda x:len(TextBlob(x).words))
df['number_of_words']


###############################################################################

# detect the presence of wh words
wh_words = set(['why','who','which','what','where','when','how'])
df['is_wh_words_present'] = df['text'].apply(lambda x:True if len(set(TextBlob(str(x)).words).intersection(wh_words))>0 else False)
df['is_wh_words_present']

###############################################################################

#polarity of the sentence
df['polarity']=df['text'].apply(lambda x:TextBlob(str(x)).sentence.polarity)
df['polarity']

sentence10='i like this example very much '
pol=TextBlob(sentence10).sentiment.polarity
pol
sentence10="this is fantastic example and i like it very much"
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10="This was helpful example but I would have prefer another one"
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence="this is my personal opinion that it was helpful example but i would prefer another one"
pol=TextBlob(sentence10).sentiment.polarity
pol

###############################################################################

#subjectivity of the dataframe df and check whether there is  personal opinion or not
df['subjectivity']=df['text'].apply(lambda x:TextBlob(str(x)).sentence.subjectivity)
df['subjectivity']

###############################################################################

# to find language of the sentence,this part of code will get http error
df['language']=df['text'].apply(lambda x:TextBlob(str(x)).detect_language())
df['language']=df['text'].apply(lambda x:TextBlob(str(x)).detect_language())


###############################################################################

#bag of words
#This bow converts unstructured data to structured form
import pandas as pd
from sklearn.feature_extraction.test import CountVectorizer
corpus=["At least seven indian pharma companies are working to develop  vaccine againest the corona virus.The deadly virus that has already infected more than 14 billions globally Bharat Biotech is among the domastic pharma firm working on the corona virus vaccine in india"]

bag_of_words_model=CountVectorizer()
print(bag_of_words_model.fit_transform(corpus).todense)
bag_of_word_df=pd.DataFrame(bag_of_words_model.fit_transform(corpus).todense())
#This will create database
bag_of_word_df.columns=sorted(bag_of_words_model.vocabulary_)
bag_of_word_df.head()
###############################################################################

bag_of_word_small=CountVectorizer(max_features=5)
bag_of_word_df=pd.DataFrame(bag_of_word_model_small.fit_transform(corpus).todense())
bag_of_word_df.columns=sorted(bag_of_words_model.vocabulary_)
bag_of_word_df.head()

###############################################################################


import pandas as pd
import numpy as np
#read the csv
df=pd.read_csv("C:\Data Set\spam.csv")
#check first 10 records
df.head()
#Total number of spam and ham
df.Category.value_counts()
#create one more colum comprises 0 and 1
#name of column is spam
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.shape

###############################################################################

#Train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.2)

#let us check the shape of x train data and x test data
x_train.shape
x_test.shape

#Let us check the type of x_train and y_train
type(x_train)
type(y_train)

###############################################################################

#Create a bag of words representation using CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
v= CountVectorizer()
x_train_cv=v.fit_transform(x_train.values)
x_train_cv
#After creation of BOW let us check the shape
x_train_cv.shape

###############################################################################

#Train the naive bayes model
from sklearn.naive_bayes import MultinomialNB
#initilize the model
model=MultinominalNB()
#Train the model
model.fit(x_train_cv,y_train)

###############################################################################

#create bag of words representation using CountVectorizer of x_test
x_test_cv=v.transform(x_test)


#Evaluate Performance
from sklearn.metrics import classification_report
y_pred=model.predict(x_test_cv)
print(classification_report(y_test,y_pread))

####################################################################################

#how to use TFIDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
corpus =['The mouse had a tiny little mouse',' The cat saw the mouse ',' The  ']

#step 1 initialize count vector
cv=CountVectorizer()


#to count the total no of TF
word_count_vector=cv.fit_transform(corpus)
word_count_vector.shape
'''
Out[88]: (3, 7)
'''

#now next step is apply IDF
tfidf_transformer= TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)


#this matrix is in the raw matrix from let us convert it in dataframe 
df_idf=pd.DataFrame(tfidf_transformer.idf_,index=cv.get_feature_names_out(),columns=["idf_weights"])


#sort ascending
df_idf.sort_values(by=["idf_weights"])
'''
Out[97]: 
        idf_weights
the        1.000000
mouse      1.287682
cat        1.693147
had        1.693147
little     1.693147
saw        1.693147
tiny       1.693147
'''
################################################################################

import pandas as pd
#read the data into a pandas dataframe
df =pd.read_csv("C:\Data Set\Ecommerce_data.csv")
print(df.shape)
df.head(5)
#check the Distribution of labels
df['label'].value_counts()
#add the new column which gives a unique number to each of these label
df['label_num']=df['label'].map({
    'Household':0,
    'Books':1,
    'Electronics':2,
    'Cloothing & Accessories':3
})
#checking the result
df.head(5)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(df.Text,df.label_num,test_size=0.2,random_state=2022,stratify=df.label_num)

#20% samples will go to test dataset

print("Shape of x_train:", x_train.shape)
print("Shape of x_test:" x_test.shape)
y_train.value_counts()
y_test.value_counts()

###############################################################################
#Apply to classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

#1. create a pipeline object
clf=Pipeline([('vectorizer_tfidf',TfidfVectorizer()),('KNN',KneighborsClassifier())])

#2. fit with x_train and y_train
clf.fit(x_train, y_train)


#3.get the predictions for x_test and store it in y_pred
y_pred = clf.predict(x_test)



###############################################################################

#pip install gensim
#pip install  =python_levenshtin
import gensim
import pandas as pd
df=pd.read_json("C:\Data Set\Cell_Phones_and_Accessories_5.json",lines=True)
dfdf.shape

#simple Preprocessing & Tokenization
review_text = df.reviewText.apply(gensim.utils.simple_preprocess)
review_text
#let us check first word of each review

review_text.loc[0]
#let us check first row of dataFrame

df.reviewText.loc[0]

#training the Word2Vec Model
model=genism.models.Word@Vec(
    window=10,
    min_count=2,
    workers=4)
'''
where windows is how many words you are going to consider as sliding windows
you can choose any count min_count there must min 2 words in each sentence
workers:no.of threads'''

#Build Vocabulary
model.build_vocab(review_text, progress_per=1000)
#progress_per:after 1000 words it show progress
#train the Word2Vec model
#it will take time,have patience
model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)
#save the model
model.save("word2vec-amazon-cell-accessories-reviews-short.model")
#finding Similar words and similarity between words
model.wv.mos_similar("bad")
model.wv.similarity(w1="cheap",w2="inexpensive")
model.mv.similarity(w1="great",w2="good")




