import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 
# compare ensemble to each baseline classifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot

df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/ready12_24_train.csv')  
del df['Health_Camp_ID']
from preprocessing import drop_cols , one_hot_encoding , scale , drop_cols_specific

dataframe_1 = scale(df) #Scale all features that are not binary 



# get the dataset
def get_dataset(dataframe):
    y= dataframe.pop('y_target')

    X = dataframe 

    return X, y

# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LogisticRegression()))
	level0.append(('knn', KNeighborsClassifier()))
	level0.append(('cart', DecisionTreeClassifier()))
	level0.append(('svm', SVC()))
	level0.append(('bayes', GaussianNB()))
	# define meta learner model
	level1 = LogisticRegression()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model
 
# get a list of models to evaluate
def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores




if __name__ == '__main__':
    
    df_encode= one_hot_encoding(dataframe_1, columns = ['Category 1','Category 2','Category 3','Job Type_x', 'online_score','City_Type2_x']) # for testing differnt cities dont drop City_Type
    #df_encode2= one_hot_encoding(dataframe_2, columns = ['Category 1','Category 2','Category 3','Job Type', 'online_score']) #, 'City_Type2_x'
    
    df_encode1 = df_encode.copy() 
    #df_encode4 = df_encode2.copy() 

    df_test = df_encode1.drop(['Category 1','Category 2','Category 3','Job Type_x', 'online_score','City_Type2_x'],axis=1) #,'City_Type2_x'
    df_test1 = df_test.copy()  
    print(df_test1.columns)


        
    # define dataset
    X, y = get_dataset(df_test1)
    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()