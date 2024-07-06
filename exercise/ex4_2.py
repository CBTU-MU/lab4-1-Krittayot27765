def homework(): 
    import pandas as pd
    import numpy as np  
    import matplotlib.pyplot as plt
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import load_iris
    iris = load_iris()
    y = iris.target
    X = iris.data
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(10,10),
                   random_state= 10,
                   max_iter = 5000)
    nn.fit(X,y)
    scoreModel = nn.score(X,y)
    # YOUR CODE HERE
    
    return scoreModel


if __name__ == '__main__':
    print(homework())
