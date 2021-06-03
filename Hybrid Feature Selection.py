#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
import re,string,unicodedata
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import string


# In[2]:


stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "arent", 'as', 'at',
 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
 'can', "cant", 'cannot', 'could', "couldnt", 'did', "didnt", 'do', 'does', "doesnt", 'doing', "dont", 'down', 'during',
 'each', 'few', 'for', 'from', 'further', 
 'had', "hadnt", 'has', "hasnt", 'have', "havent", 'having', 'he', "hed", "hell", "hes", 'her', 'here', "heres",
 'hers', 'herself', 'him', 'himself', 'his', 'how', "hows",
 'i', "id", "ill", "im", "ive", 'if', 'in', 'into', 'is', "isnt", 'it', "its", 'its', 'itself',
 "lets", 'me', 'more', 'most', "mustnt", 'my', 'myself',
 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours' 'ourselves', 'out', 'over', 'own',
 'same', "shant", 'she', "shed", "shell", "shes", 'should', "shouldnt", 'so', 'some', 'such', 
 'than', 'that',"thats", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "theres", 'these', 'they', "theyd", 
 "theyll", "theyre", "theyve", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 
 'was', "wasnt", 'we', "wed", "well", "were", "weve", 'were', "werent", 'what', "whats", 'when', "whens", 'where',
 "wheres", 'which', 'while', 'who', "whos", 'whom', 'why', "whys",'will', 'with', "wont", 'would', "wouldnt", 
 'you', "youd", "youll", "youre", "youve", 'your', 'yours', 'yourself', 'yourselves', 
 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', '1st', '2nd', '3rd',
 '4th', '5th', '6th', '7th', '8th', '9th', '10th','say']


# In[3]:


import pandas as pd 
  
imdb_data = pd.read_csv("IMDB.csv") 
print(imdb_data)


# In[4]:


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(denoise_text)


# In[5]:


import string
import re
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 

def clean(text):
    words=text.split()
    cleaned_words=[]
    for word in words:
        word=word.lower()
        word=word.translate(str.maketrans('', '', string.punctuation))
        word=re.sub(r"""\w*\d\w*""", '', word)
        if word in stopwords:
            continue
        if word=='':
            continue
        word=ps.stem(word)
        cleaned_words.append(word)
    return ' '.join(cleaned_words)
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(clean)


# In[6]:


data=[]
target=[]

for i in range(50000):
    data.append(imdb_data.iloc[i]['review'])
    if(imdb_data.iloc[i]['sentiment']=='negative'):
        target.append(0)
    else:
        target.append(1)


# In[7]:


x_train=data[0:40000]
y_train=target[0:40000]
x_test=data[40000:]
y_test=target[40000:]


# In[8]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

vect = CountVectorizer(max_df=0.9, min_df=2, max_features=15000, stop_words= stopwords)
X_train = vect.fit_transform(x_train)
X_test=vect.transform(x_test)
Y_train=np.array(y_train)
Y_test=np.array(y_test)


# In[9]:


#Information Gain
res = dict(zip(mutual_info_classif(X_train, Y_train, discrete_features=True),vect.get_feature_names()))
X_train = X_train.toarray()
df_train=pd.DataFrame(X_train, columns = vect.get_feature_names())
res = sorted(res.items(),reverse=True)
selected_features=[res[i][1] for i in range(10000)]
features=vect.get_feature_names()
for i in range(len(selected_features)):
    features.remove(selected_features[i])
df_train=df_train.drop(columns=features)
X_test=X_test.toarray()
df_test=pd.DataFrame(X_test, columns = vect.get_feature_names())
df_test=df_test.drop(columns=features)


# In[10]:


"""
#Chi Square
X_train = X_train.toarray()
X_test=X_test.toarray()
df_train=pd.DataFrame(X_train, columns = vect.get_feature_names())
df_test=pd.DataFrame(X_test, columns = vect.get_feature_names())
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
kbest = SelectKBest(score_func = chi2, k = 10000)
X_train_kbest=kbest.fit_transform(X_train,Y_train)
X_test_kbest=kbest.transform(X_test)
mask = kbest.get_support()
new_features = df_train.columns[mask]
df_train_chi=pd.DataFrame(X_train_kbest, columns = new_features)
df_test_chi=pd.DataFrame(X_test_kbest, columns = new_features)
df_train=df_train_chi
df_test=df_test_chi
"""


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from deap import creator, base, tools, algorithms
from scoop import futures
import random
import numpy
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

X_train=df_train
X_test=df_test
y_train=Y_train
y_test=Y_test

# Feature subset fitness function
def getFitness(individual, X_train, X_test, y_train, y_test):

	# Parse our feature columns that we don't use
	cols = [index for index in range(len(individual)) if individual[index] == 0]
	X_trainParsed = X_train.drop(X_train.columns[cols], axis=1)
	X_trainOhFeatures = pd.get_dummies(X_trainParsed)
	X_testParsed = X_test.drop(X_test.columns[cols], axis=1)
	X_testOhFeatures = pd.get_dummies(X_testParsed)

	sharedFeatures = set(X_trainOhFeatures.columns) & set(X_testOhFeatures.columns)
	removeFromTrain = set(X_trainOhFeatures.columns) - sharedFeatures
	removeFromTest = set(X_testOhFeatures.columns) - sharedFeatures
	X_trainOhFeatures = X_trainOhFeatures.drop(list(removeFromTrain), axis=1)
	X_testOhFeatures = X_testOhFeatures.drop(list(removeFromTest), axis=1)

	# Apply NB on the data, and calculate accuracy
	clf = MultinomialNB()
	clf.fit(X_trainOhFeatures, y_train)
	predictions = clf.predict(X_testOhFeatures)
	accuracy = accuracy_score(y_test, predictions)

	# Return calculated accuracy as fitness
	return (accuracy,)


# Create Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(df_train.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Continue filling toolbox...
toolbox.register("evaluate", getFitness, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

#========

def getHof():

	# Initialize variables to use eaSimple
	numPop = 20
	numGen = 10
	pop = toolbox.population(n=numPop)
	hof = tools.HallOfFame(numPop * numGen)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", numpy.mean)
	stats.register("std", numpy.std)
	stats.register("min", numpy.min)
	stats.register("max", numpy.max)

	# Launch genetic algorithm
	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=numGen, stats=stats, halloffame=hof, verbose=True)

	# Return the hall of fame
	return hof

def getMetrics(hof):

	# Get list of percentiles in the hall of fame
	percentileList = [i / (len(hof) - 1) for i in range(len(hof))]
	
	# Gather fitness data from each percentile
	testAccuracyList = []
	validationAccuracyList = []
	individualList = []
	for individual in hof:
		testAccuracy = individual.fitness.values
		validationAccuracy = getFitness(individual, X_train, X_test, y_train, y_test)
		testAccuracyList.append(testAccuracy[0])
		validationAccuracyList.append(validationAccuracy[0])
		individualList.append(individual)
	testAccuracyList.reverse()
	validationAccuracyList.reverse()
	return testAccuracyList, validationAccuracyList, individualList, percentileList


if __name__ == '__main__':

	'''
	First, we will apply NB using all the features to acquire a baseline accuracy.
	'''
	individual = [1 for i in range(len(df_train.columns))]
	testAccuracy = getFitness(individual, X_train, X_test, y_train, y_test)
	validationAccuracy = getFitness(individual, X_train, X_test, y_train, y_test)
	print('\nTest accuracy with all features: \t' + str(testAccuracy[0]))
	print('Validation accuracy with all features: \t' + str(validationAccuracy[0]) + '\n')

	'''
	Now, we will apply a genetic algorithm to choose a subset of features that gives a better accuracy than the baseline.
	'''
	hof = getHof()
	testAccuracyList, validationAccuracyList, individualList, percentileList = getMetrics(hof)

	# Get a list of subsets that performed best on validation data
	maxValAccSubsetIndicies = [index for index in range(len(validationAccuracyList)) if validationAccuracyList[index] == max(validationAccuracyList)]
	maxValIndividuals = [individualList[index] for index in maxValAccSubsetIndicies]
	maxValSubsets = [[list(X_train)[index] for index in range(len(individual)) if individual[index] == 1] for individual in maxValIndividuals]

	print('\n---Optimal Feature Subset(s)---\n')
	for index in range(len(maxValAccSubsetIndicies)):
		print('Validation Accuracy: \t\t' + str(validationAccuracyList[maxValAccSubsetIndicies[index]]))
		print('Individual: \t' + str(maxValIndividuals[index]))
		print('Number Features In Subset: \t' + str(len(maxValSubsets[index])))
		print('Feature Subset: ' + str(maxValSubsets[index]))


# In[ ]:


nb_ga_chi2_acc_imdb=np.load('nb_ga_chi2_acc_imdb.npy')
nb_ga_chi2_pl_imdb=np.load('nb_ga_chi2_pl_imdb.npy')
nb_ga_ig_acc_imdb=np.load('nb_ga_ig_acc_imdb.npy')
nb_ga_ig_pl=np.load('nb_ga_ig_pl_imdb.npy')


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(nb_ga_chi2_pl_imdb,nb_ga_chi2_acc_imdb,label="GA+Chi2")
plt.plot(nb_ga_ig_pl,nb_ga_ig_acc_imdb,label="GA+IG")
plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
#plt.plot(percentileList, ynew, color='b')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Performance of Filter Method+GA on Naive Bayes")
plt.show()


# In[ ]:


svc_ga_chi2_acc_imdb=np.load('svc_ga_chi2_acc_imdb.npy')
svc_ga_chi2_pl_imdb=np.load('svc_ga_chi2_pl_imdb.npy')
svc_ga_ig_acc_imdb=np.load('svc_ga_ig_acc_imdb.npy')
svc_ga_ig_pl=np.load('svc_ga_ig_pl_imdb.npy')


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(svc_ga_chi2_pl_imdb,svc_ga_chi2_acc_imdb,label="GA+Chi2")
plt.plot(svc_ga_ig_pl,svc_ga_ig_acc_imdb,label="GA+IG")
plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Performance of Filter Method+GA on Linear SVC")
plt.show()


# In[ ]:




