from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import timeit


start = timeit.default_timer() #start timer
     
dataset = pd.read_csv('D:\School\OZP\Coderen\Datasets\DatasetReduced.csv', sep=';',
    	header = None)             #import Dataset
text = dataset.drop([1], axis=1)    #drop scores
score = dataset.drop([0], axis=1)   #drop text
textList = text[0].tolist() #transform dataframe to list
    
# using tokenizer 
model = Tokenizer()
model.fit_on_texts(textList)
     
#print keys 
vocab = (list(model.word_index.keys()))
vocab.insert(0, 'score')
     
#create bag of words representation 
rep = model.texts_to_matrix(textList, mode='count')
rep = np.delete(rep, 0, 1)  # delete first column of rep
rep = np.append(score, rep, axis=1) #Append score in front of rep
mat = np.matrix(rep)
with open('OutfileBoW.csv','wb') as f:  #write data into file
    for line in mat:
        np.savetxt(f, line, fmt='%.0f', delimiter="; ")
            
stop = timeit.default_timer() #end timer
print('Time: ', stop - start) #print elapsed time
