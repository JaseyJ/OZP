from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd
import nltk
from keras.preprocessing.text import Tokenizer
import timeit

start = timeit.default_timer() #start timer
    
dataset = pd.read_csv('D:\School\OZP\Coderen\Datasets\DatasetReduced.csv', sep=';',
    	header = None)             #import dataset
text = dataset.drop([1], axis=1) #separate text from score
score = dataset.drop([0], axis=1) #separate score from text
sentences = text[0].tolist() #transform dataframe to list
    
cvec = CountVectorizer(stop_words='english', min_df=0, max_df=1.0, ngram_range=(1,1))
sf = cvec.fit_transform(sentences)
    
#assign score (weight) to all the words 
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
    
importantWords = [] #create array to store the words with a higher score than the average score in
i = 0
while i < weights_df.shape[0]: #while loop to loop over all weights
    if weights_df['weight'][i] >= weights_df['weight'].mean(): #if score of term is higher or equal to score mean:
        importantWords.append(weights_df['term'][i]) #add these terms to list of important words
    i += 1
    
reviewsFiltered = [] #empty list to store all filtered reviews in
    
for entry in sentences: #loop through all reviews
    nltk_tokens = nltk.word_tokenize(entry) #tokenize review into individual words
    text = '' #empty string to add words to
    for word in nltk_tokens: #loop through all words in reviews
        if word in importantWords: #loop to add all words from review that are also in importantWords to the text string
            text = word + ' ' + text
    reviewsFiltered.append(text) #append filtered reviews to list
   
# using tokenizer on the filtered reviews
model = Tokenizer()
model.fit_on_texts(reviewsFiltered)
     
#create bag of words representation 
rep = model.texts_to_matrix(reviewsFiltered, mode='count')
rep = np.delete(rep, 0, 1)  # delete first column of rep
rep = np.append(score, rep, axis=1) #Append score in front of rep
    
mat = np.matrix(rep)
with open('OutfileTFIDF.csv','wb') as f:    #write data into file
    for line in mat:
        np.savetxt(f, line, fmt='%.0f', delimiter="; ")
            
stop = timeit.default_timer() #end timer
print('Time: ', stop - start) #print elapsed time