import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing

from nltk.corpus import stopwords
import time


stop_words = set(stopwords.words('english'))

# ========= Read Dataset =============
df = pd.read_csv('../datasets/q1/train.csv', encoding='utf-8')
df['Combined'] = 3*(df['Title'] + ' ') + df['Content']
# ====================================

# ======== Initialisation ============
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['Label'])
clf = KNeighborsClassifier(n_neighbors=9)
# ====================================

# ==================  BOW ==================
total_time = time.time()
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df['Combined'])

folds = 5
print('Starting {}-fold for BOW'.format(folds))
kfold_time = time.time()
kf = KFold(n_splits=folds)
accuracy = 0
precision = 0
recall = 0
fmeasure = 0

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    accuracy += accuracy_score(y_test, predictions)
    precision += precision_score(y_test, predictions, average='macro')
    recall += recall_score(y_test, predictions, average='macro')
    fmeasure += f1_score(y_test, predictions, average='macro')

accuracy /= folds
precision /= folds
recall /= folds
fmeasure /= folds

print('accuracy = {}, precision = {}, recall = {}, f1-measure = {}'.format(
    round(accuracy, 4), round(precision, 4), round(recall, 4), round(fmeasure, 4)))
print('{}-fold time: {} s'.format(folds, time.time() - kfold_time))
print('BOW time: {} s'.format(time.time() - total_time))
# =======================================================

# ====================- SVD ==========================
total_time = time.time()
svd = TruncatedSVD(n_components=20, random_state=42)
X = svd.fit_transform(X)

folds = 5
print('Starting {}-fold for SVD'.format(folds))
kfold_time = time.time()
kf = KFold(n_splits=folds)
accuracy = 0
precision = 0
recall = 0
fmeasure = 0

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    accuracy += accuracy_score(y_test, predictions)
    precision += precision_score(y_test, predictions, average='macro')
    recall += recall_score(y_test, predictions, average='macro')
    fmeasure += f1_score(y_test, predictions, average='macro')

accuracy /= folds
precision /= folds
recall /= folds
fmeasure /= folds

print('accuracy = {}, precision = {}, recall = {}, f1-measure = {}'.format(
    round(accuracy, 4), round(precision, 4), round(recall, 4), round(fmeasure, 4)))
print('{}-fold time: {} s'.format(folds, time.time() - kfold_time))
print('SVD time: {} s'.format(time.time() - total_time))
