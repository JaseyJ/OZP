import sklearn.feature_extraction.text
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from collections import Counter
import timeit

start = timeit.default_timer() #timer to determine runtime
    
dataset = pd.read_csv('D:\School\OZP\Coderen\Datasets\DatasetReduced.csv', sep=';', #read dataset
    	header = None)
text = dataset.drop([1], axis=1) #drop scores
score = dataset.drop([0], axis=1) #drop text
ngram_size = 3 #determine window size of n-gram
    
arr = text.to_numpy() #convert text reviews from dataframe to numpy array
entire= [] #create empty list to store processed n-grams in
vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(ngram_size,ngram_size))
    
i=0
while i < arr.size:  #while loop to n-gramify each entry separatly and append corresponding score to front of array
    lengtevar = str(arr[i])
    lengte = len(lengtevar.split()) #determine length (amount of words) of entry
        
    if lengte > ngram_size: #apply standard procedure when length of input is greater or equal to window size
        vect.fit(arr[i]) #apply only on current index
        values = vect.get_feature_names() #insert n-grams to values
            
    else:   #if the length is smaller than the n-gram size, just return the original entry
        word = arr[i].tolist()
        values = word
        
    entire.append(values) #append list with current index n-gramified
    i += 1
    
vocab = []
for entry in entire:    #collect all n-grams that have been created
    for each in entry:
        vocab.append(each)
    
c = Counter(vocab)
selected_vocab = Counter({k: c for k, c in c.items() if c >= 4}) #Count amount of times n-gram appears, add to selected vocab if higher than given value
    
final = []
emptyEntries = 0
j=0
while j < len(entire):              #Loop to filter out all n-grams that are not part of selected vocab
    words = []
    for each in entire[j]:
        if each in selected_vocab:
            words.append(each)
    final.append(words)       
    j += 1
    
# using tokenizer 
model = Tokenizer()
model.fit_on_texts(final)
     
#create bag of words representation 
rep = model.texts_to_matrix(final, mode='count')
rep = np.delete(rep, 0, 1)  # delete first column of rep, since it is always 0
rep = np.append(score, rep, axis=1) #Append score in front of rep
print(rep)
mat = np.matrix(rep)
with open('OutfileNgram.csv','wb') as f: #write data into file
    for line in mat:
        np.savetxt(f, line, fmt='%.0f', delimiter="; ")
    
stop = timeit.default_timer()
print('Time: ', stop - start)
