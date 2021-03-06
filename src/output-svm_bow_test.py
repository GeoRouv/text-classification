import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing

import time
import numpy as np
from nltk.corpus import stopwords

MyStopWords = set(stopwords.words('english'))

#------------Read the csv file and change the encoding----------------
df_train = pd.read_csv('../datasets/q1/train.csv', encoding='utf-8')
df_train['Combined']  = 4*(df_train['Title'] + ' ')  + df_train['Content']
#---------------------------------------------------------------------

#--------- INITIALIZATIONS -------------
le = preprocessing.LabelEncoder()
y = le.fit_transform(df_train['Label'])
clf = LinearSVC(random_state=42)
#----------------------------------------

#---------- CREATE VECTORIZER -----------
vectorizer = TfidfVectorizer(stop_words=MyStopWords)
X = vectorizer.fit_transform(df_train['Combined'])
#----------------------------------------

print('Running on test set...')
#------------Read the csv file and change the encoding----------------
df_test = pd.read_csv('../datasets/q1/test_without_labels.csv', encoding='utf-8')
df_test['Combined']  = 3*(df_test['Title'] + ' ')  + df_test['Content']
#---------------------------------------------------------------------

print('Training Linear SVM...')
clf.fit(X, y)
print('Finished training...')

X = vectorizer.transform(df_test['Combined'])
predictions = clf.predict(X)

predictions = le.inverse_transform(predictions)

result = pd.DataFrame({'Id':df_test['Id'],'Predicted':predictions})
result.to_csv('testSet_categories.csv', sep=',', index=False)