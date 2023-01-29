from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV

def classify(train, train_target, test, test_target):
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(train, train_target)
    pred = naive_bayes_classifier.predict(test)
    return metrics.classification_report(test_target, pred, target_names=['Fake', 'Real'])

def tune(nb_classifier, train, train_target):
    param_grid = {'alpha': [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]}
    grid_search = RandomizedSearchCV(nb_classifier, param_grid)
    grid_search.fit(train, train_target)
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)