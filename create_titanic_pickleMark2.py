#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd
import pickle
import random

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''




def create_feature_sets_and_labels(X,y,test_size = 0.1):
    train_x,test_x,train_y,test_y = cross_validation.train_test_split(X,y,test_size=0.2)
	#lexicon = create_lexicon(pos,neg)
        
    featureset = []
    print(y)
    for i in range(len(y)):
        if y[i] == 1:
            #print('bing')
            temp =[1,0]
            tempnp = list(temp)
            #features +=X[i],tempnp
            features = list(X[i])
            featureset.append([features,temp])
        if y[i] == 0:
            
            #print('bing')
            temp =[0,1]
            tempnp = list(temp)
            #features +=X[i],tempnp
            features = list(X[i])
            featureset.append([features,temp])
            #print('ping')
            #features +=X[i],([0,1])
    
    #print(featureset)
    
                       #features += 
                   
	
	#features += sample_handling('pos.txt',lexicon,[1,0])
	#features += sample_handling('neg.txt',lexicon,[0,1])

    random.shuffle(featureset)
    features = np.array(featureset)

    testing_size = int(test_size*len(features))
    #train_x,test_x,train_y,test_y = cross_validation.train_test_split(X,y,test_size=0.2)
    

    

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    return train_x,train_y,test_x,test_y

def handle_non_numerical_data(df):
    #colums df
    columns = df.columns.values
    for column in columns:
        #each column create a list of unique values
        text_digit_vals = {}
        def convert_to_int(val):
           return text_digit_vals[val]
#       looking for data type within the olumn which is not a number value
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
           # print(df[column].name)
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            #unique values given number id as new value to replace text with a int
            for unique in unique_elements:
               # print(unique)
                #check if value exists in list of unique values
                 #eg [bob,1],[glob,2],[frog,3]
                if unique not in text_digit_vals:
                    #increment new unique value within the list
                   
                    text_digit_vals[unique] = x
                    x+=1
#           Call convert to int inside columns 
            df[column] = list(map(convert_to_int, df[column]))

    return df




if __name__ == '__main__':
    df = pd.read_csv('titanic.csv')
    #print(df.head())
    df.drop(['body','name'], 1, inplace=True)
    #print(df.head())
    df.convert_objects(convert_numeric=True)
    df.fillna(0, inplace=True)
    #print(df.head())
    df = handle_non_numerical_data(df)
    df.to_csv('TitanicNum.csv')
    print(df.head())
    #X is data - survived
    X = np.array(df.drop(['survived'], 1).astype(float))
    #convert to scala for improved accuracy
    X = preprocessing.scale(X)
    #y is the value of survived to be checked
    #against after the classifier makes a prediction eg test data for the model
    y = np.array(df['survived'])

#

#with open('titanic_set.pickle','wb') as f:
#    pickle.dump([train_x,train_y,test_x,test_y],f)
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels(X,y)
    # if you want to pickle this data:
    with open('titanic_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)
