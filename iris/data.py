from sklearn import datasets
from sklearn.model_selection import train_test_split

def get_data():
    # 1.
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # 2.1
    (X_train, X_test, y_train, y_test) = holdout(X, y)
    # 3.2
    return (X_train, X_test, y_train, y_test)

# data utility function
def holdout(X, y):
    # 2.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # 3.1
    return (X_train, X_test, y_train, y_test)

