def homework(): 
    import pandas as pd
    import numpy as np  
    import matplotlib.pyplot as plt

    import pandas as pd
    data = pd.read_csv("https://raw.githubusercontent.com/toche7/DataSets/main/admit.csv")
    from sklearn.neural_network import MLPClassifier

    y = data.Label
    X = data[['SubjectA','SubjectB']]
    NN =  MLPClassifier(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(10,10),
                   random_state= 10,
                   max_iter = 5000)
    NN.fit(X,y)
    scoreModelNN = NN.score(X,y)
    # YOUR CODE HERE

    return scoreModelNN


if __name__ == '__main__':
    print(homework())
