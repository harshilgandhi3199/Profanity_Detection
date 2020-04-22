#Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.20, random_state=0)

"""#naive bayes classifier
from sklearn.regre import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train, y_train)"""

#logistic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#Predicting the test set 
y_pred=classifier.predict(X_test)

#Creating a confusion matrix and evaluating model's performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
cm=confusion_matrix(y_test, y_pred)
score=accuracy_score(y_test, y_pred)
print(score)
print(classification_report(y_test, y_pred))

#Applying k-cross fold validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
