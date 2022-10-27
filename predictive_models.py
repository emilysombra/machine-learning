from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd


def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn

# KNN
num_neighbors = [3, 5, 7]
distance_algorithms = ['ball_tree', 'kd_tree', 'brute']

# decision tree
criterions = ['gini', 'entropy']

# naive_bayes
naive_bayes = [GaussianNB, MultinomialNB, BernoulliNB]

df = pd.read_csv('datasets/water_potability.csv')
df.fillna(df.mean(), inplace=True)
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
x = df.loc[:, features]
y = df.loc[:, ['Potability']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

for i in range(3):
    clf = KNN(n_neighbors=num_neighbors[i], algorithm=distance_algorithms[i])
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    print('Classificador: KNN')
    print(f'Atributos: K={num_neighbors[i]}, métrica de distância={distance_algorithms[i]}')
    print(f'Acurácia: {accuracy}')
    print(f'Precisão: {precision}')
    print(f'F-Score: {f1}')
    print(f'Recall: {recall}')
    print(f'ROC (área): {roc}')
    print('------')

for criterion in criterions:
    clf = DecisionTreeClassifier(criterion=criterion)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    print('Classificador: Decision Tree')
    print(f'Critério: {criterion}')
    print(f'Acurácia: {accuracy}')
    print(f'Precisão: {precision}')
    print(f'F-Score: {f1}')
    print(f'Recall: {recall}')
    print(f'ROC (área): {roc}')
    print('------')

for NaiveBayes in naive_bayes:
    clf = NaiveBayes()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    print(f'Classificador: {clf}')
    print(f'Acurácia: {accuracy}')
    print(f'Precisão: {precision}')
    print(f'F-Score: {f1}')
    print(f'Recall: {recall}')
    print(f'ROC (área): {roc}')
    print('------')