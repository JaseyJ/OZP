#The base of this code is from Abhishek Sharma, posted on geeksforgeeks.org
# Link: https://www.geeksforgeeks.org/decision-tree-implementation-python/

# Importing the required packages 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import timeit
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import collections

n = 0;
while n < 20:
    start = timeit.default_timer()
    print(n)
    
    # Function importing Dataset 
    def importdata(): 
    	balance_data = pd.read_csv('D:\School\OZP\Coderen\Code\OutfileNgram.csv' ,
    	sep= ';', header = None) 
 
    	return balance_data 
    
    # Function to split the dataset 
    def splitdataset(balance_data): 
    
    	# Separating the target variable 
    	X = balance_data.values[:, 1:] #[1] to [5] are attributes
    	Y = balance_data.values[:, 0]  #[0] is target variable
    
    	# Splitting the dataset into train and test 
    	X_train, X_test, y_train, y_test = train_test_split( 
    	X, Y, test_size = 0.3, stratify=Y) 
    	
    	#Apply SMOTE oversampling to the training data
    	oversample = SMOTE(sampling_strategy='auto')
    	steps = [('o', oversample)]
    	pipeline = Pipeline(steps=steps)
    	X_TrainRe, y_TrainRe = pipeline.fit_resample(X_train, y_train)
        
    	return X, Y, X_TrainRe, X_test, y_TrainRe, y_test 
    	
    # Function to perform training with giniIndex. 
    def train_using_gini(X_train, X_test, y_train): 
    
    	# Creating the classifier object 
    	clf_gini = DecisionTreeClassifier(criterion = "gini", 
    			random_state = 100,max_depth=3, min_samples_leaf=5) 
    
    	# Performing training 
    	clf_gini.fit(X_train, y_train) 
    	return clf_gini     
    
    # Function to make predictions 
    def prediction(X_test, clf_object): 
    
    	# Predicton on test with giniIndex 
    	y_pred = clf_object.predict(X_test) 
    	print("Predicted values:") 
    	print(collections.Counter(y_pred)) 
    	return y_pred 
    	
    # Function to calculate accuracy 
    def cal_accuracy(y_test, y_pred): 
    	
        #Calculate accuracy per class
    	matrix = confusion_matrix(y_test, y_pred)
    	accuracies = matrix.diagonal()/matrix.sum(axis=1)
    	print(accuracies)
         
    	#Calculate overall accuracy
    	print ("Accuracy : ", 
    	accuracy_score(y_test,y_pred)*100) 
    	
    	print("Report : ", 
    	classification_report(y_test, y_pred))
    
    # Driver code 
    def main(): 
    	
    	# Building Phase 
    	data = importdata() 
    	X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    	clf_gini = train_using_gini(X_train, X_test, y_train) 
    	
    	# Operational Phase 
    	print("Results Using Gini Index:") 
    	
    	# Prediction using gini 
    	y_pred_gini = prediction(X_test, clf_gini) 
    	cal_accuracy(y_test, y_pred_gini) 
    	
    	
    # Calling main function 
    if __name__=="__main__": 
    	main() 
        
    
    stop = timeit.default_timer()
    print('Time: ', stop - start, '/n')
    n += 1
    

