def homework(): 
    import pandas as pd
    import numpy as np  
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_iris
    iris = load_iris()
    y = iris.target
    X = iris.data
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X,y)
    scoreModel = dt.score(X,y)
    # YOUR CODE HERE

    return scoreModel


if __name__ == '__main__':
    print(homework())
