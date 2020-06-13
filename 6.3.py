from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
cancer = load_iris()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
        cancer.target, stratify=cancer.target, random_state=42)

svc = SVC(kernel='linear',degree=1,gamma="auto",C=1)
svc.fit(X_train,y_train)
print("The accuracy on training set(svc) under linear:{:3f}".format(svc.score(X_train, y_train)))
print("The accuracy on training set(svc) under linear:{:3f}".format(svc.score(X_test, y_test)))

svc = SVC(kernel='rbf',degree=1,gamma="auto",C=1)
svc.fit(X_train,y_train)
print("The accuracy on training set(svc) under rbf(gauss):{:3f}".format(svc.score(X_train, y_train)))
print("The accuracy on training set(svc) under rbf(gauss):{:3f}".format(svc.score(X_test, y_test)))

mlp=MLPClassifier(random_state=0,max_iter=1000,alpha=1).fit(X_train,y_train)
print("Training set score(mlp): {:.3f}".format(mlp.score(X_train, y_train)))
print("Test set score(mlp): {:.3f}".format(mlp.score(X_test, y_test)))

tree = DecisionTreeClassifier(random_state=0,max_depth=4).fit(X_train,y_train)
print("Accuracy on training set(tree):{:3f}".format(tree.score(X_train,y_train)))
print("Accuracy on test set(tree):{:3f}".format(tree.score(X_test,y_test)))