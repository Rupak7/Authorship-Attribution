import csv
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
csv.register_dialect('myDialect', delimiter = '|')

age_20=[]
csv.field_size_limit(sys.maxsize)
with open('training_blogs_data.csv') as myFile:
   reader = csv.reader(myFile, dialect='myDialect')
   for row in reader:
       #print(row[1])
       if row[1] == "20":
           age_20.append(row[3])

#print(age_20)

str="";
for i in age_20:
    str += i;

Age=[]
#print(str)
text = word_tokenize(str)
#
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # vectorizer = TfidfVectorizer()
# # X = vectorizer.fit_transform(age_20)
# # print(X)
#
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


#lectures = ["this is some food", "this is some drink"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text)
indices = np.argsort(vectorizer.idf_)[::-1]
features = vectorizer.get_feature_names()
top_n = 10
top_features = [features[i] for i in indices[:top_n]]
print top_features
