#region imports
from AlgorithmImports import *
from sklearn.neural_network import MLPClassifier
#endregion

def GetModelMLPClassifier(hidden_layers):
    
    return MLPClassifier(
                hidden_layer_sizes = hidden_layers,
                activation='tanh',#'tanh',#'logistic', 'relu', modificado para logistic em 260123
                solver='sgd',
                alpha=1,
                batch_size='auto', #If the solver is ‘lbfgs’, the classifier will not use minibatch
                learning_rate='adaptive',
                learning_rate_init=0.001,#Only used when solver=’sgd’ or ‘adam’, default: 0.001
                #power_t= 0.5, # It is used in updating effective learning rate when the learning_rate is set to ‘invscaling’. Only used when solver=’sgd’.
                max_iter=20000,#4000, #20000, #200
                random_state=None,
                verbose=True,
                momentum=0.9, #Should be between 0 and 1. Only used when solver=’sgd’
                warm_start=False, #modificado para true em 260123, pois train agora é weekly
                tol=0.0001,
                n_iter_no_change=100, # Maximum number of epochs to not meet tol improvement.
                )