from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import grid_search
import codecs

#initialize
data=[]
target = []


#read file and split data and target labels
with codecs.open("Twitter_Spam_Dataset1.csv", 'r' ,encoding='ISO-8859-1') as inpfile:
	for line in inpfile:
		container = line.split(",")
		data.append(container[0])
		target.append(container[-1].replace("\n",""))

vectorizer =   TfidfVectorizer(min_df=1,stop_words="english",ngram_range=(1,3),token_pattern="\w+",analyzer = "word")
X_title = vectorizer.fit_transform(data)
Y = target

print X_title[0]


X_train, X_test, y_train, y_test = cross_validation.train_test_split(
X_title, Y, test_size=0.1, random_state=4563)
print "split complete"
print X_train.shape


clf = linear_model.LogisticRegression(C = 100)
clf.fit(X_train,y_train)


print"predicting"
pred = clf.predict(X_test)

accuracy = accuracy_score(y_test,pred)
print (accuracy)



