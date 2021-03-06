# text-classification
Text classification of news articles using different classification methods

## Preprocessing

For providing more weight to the title of each article we multiplied it by a relatively small number and integrated it to its content.
Plus, not any extra stopwords were used beside nltk’s.
For vectorizing the text, we used sklearn’s TfidfVectorizer. This tool weights the word counts by a measure of how often they appear in the documents.
  
## Machine Learning Models
● SVM:  
LinearSVC was chosen which is similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
We didn’t tune any of its parameters because it performed nicely from the beginning.  
  
● Random Forest:  
Only parameter we changed its default value is maximum depth to 18.
Generally, more trees will result in better accuracy. However, more trees also mean more computational cost and after a certain number of trees, the improvement is negligible.
  
● KNN:  
Although this model becomes significantly slower as the volume of our data increases, it was assumed that if SVM works pretty well, KNN could do better.
In general, if training data is much larger than no. of features(m>>n), KNN is better than SVM. SVM outperforms KNN when there are large features and lesser training data. In this project, however, SVM was slightly better than KNN. As regards value k, its value was chosen experimentally.
  
