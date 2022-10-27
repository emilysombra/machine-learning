import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

df = pd.read_csv('datasets/water_potability.csv')
df.fillna(df.mean(), inplace=True)
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
x = df.loc[:, features]
y = df.loc[:, ['Potability']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# general
MAX_ITER = 25000

# SVM
kernel_list = ['linear', 'poly', 'rbf']
degree_list = [1, 2, 3]
c_list = [1.0, 2.0]
gamma_list = [1.0, 2.0]

# MLP
learning_rates = [0.001, 0.01, 0.05]
momentum_list = [0.99, 0.95, 0.9, 0.7]

df = pd.read_csv('datasets/water_potability.csv')
df.fillna(df.mean(), inplace=True)
features = [
    'ph',
    'Hardness',
    'Solids',
    'Chloramines',
    'Sulfate',
    'Conductivity',
    'Organic_carbon',
    'Trihalomethanes',
    'Turbidity'
]
x = df.loc[:, features]
y = df.loc[:, ['Potability']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)


def test_classifier(clf):
    clf.fit(x_train, y_train.values.ravel())
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    with open('results.txt', 'a') as f:
        f.write('Acuracia: {:.3f}\n'.format(accuracy))
        f.write('Precisao: {:.3f}\n'.format(precision))
        f.write('F-Score: {:.3f}\n'.format(f1))
        f.write('Recall: {:.3f}\n'.format(recall))
        f.write('ROC (area): {:.3f}\n'.format(roc))
    print('Acuracia: {:.3f}'.format(accuracy))
    print('Precisao: {:.3f}'.format(precision))
    print('F-Score: {:.3f}'.format(f1))
    print('Recall: {:.3f}'.format(recall))
    print('ROC (area): {:.3f}'.format(roc))



s = 'SVM - kernel={}, c={}, gamma={}'
for kernel in kernel_list:
    for c in c_list:
        for gamma in gamma_list:
            if kernel == 'poly':
                for degree in degree_list:
                    with open('results.txt', 'a') as f:
                        s_ = s.format(kernel, c, gamma) + f', degree={degree}\n'
                        f.write(s_)
                    print(s.format(kernel, c, gamma), f', degree={degree}', sep='')
                    clf = SVC(random_state=0, kernel=kernel, gamma=gamma, C=c, degree=degree, max_iter=MAX_ITER)
                    test_classifier(clf)
            else:
                with open('results.txt', 'a') as f:
                    s_ = s.format(kernel, c, gamma) + '\n'
                    f.write(s_)
                print(s.format(kernel, c, gamma))
                clf = SVC(random_state=0, kernel=kernel, gamma=gamma, C=c, max_iter=MAX_ITER)
                test_classifier(clf)

s = 'MLP - lr={}, momentum={}'
for lr in learning_rates:
    for momentum in momentum_list:
        with open('results.txt', 'a') as f:
            s_ = s.format(lr, momentum) + '\n'
            f.write(s_)
        print(s.format(lr, momentum))
        clf = MLP(random_state=0, learning_rate_init=lr, momentum=momentum, max_iter=MAX_ITER)
        test_classifier(clf)
