import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def run(path='datasets/water_potability.csv'):
    df = pd.read_csv(path)
    # pré processamento
    df.fillna(df.mean(), inplace=True)
    # chi-square
    print('-----\nchi-square\n-----')
    features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    x = df.loc[:, features]
    y = df.loc[:, ['Potability']]
    x_chi = SelectKBest(score_func=chi2, k=4)
    x_chi.fit(x, y)
    print(x_chi.get_feature_names_out())
    # ## PCA ##
    print('-----\nPCA\n-----')
    pca = PCA(n_components=4)
    pca_fitted = pca.fit(x)
    pca_transformed = pca_fitted.transform(x)
    # covariancia
    print('-----\ncovariancia\n-----')
    df_cov = df.cov()
    print(df_cov)
    plt.figure(figsize=(18, 13))
    sns.heatmap(df_cov, xticklabels=df.columns, yticklabels=df.columns, annot=True)
    plt.show()
    

    # autovalores e autovetores
    autovalores, autovetores = np.linalg.eig(df_cov)
    print('-----\nautovalores\n-----')
    for autovalor in autovalores:
        print(autovalor)
    
    print('-----\nautovetores\n-----')
    for autovetor in autovetores:
        print(autovetor)

    # variancia explicada
    explained_variances = []
    for autovalor in autovalores:
        explained_variances.append(autovalor / np.sum(autovalores))
    
    print('-----\nvariância explicada\n-----')
    print(np.sum(explained_variances), '\n', explained_variances)

    ev = []
    s = 0
    for i in explained_variances:
        s = s + i
        ev.append(s)

    plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, len(ev) + 1)), ev, marker='D')
    plt.grid(b=True)

    for i in range(0,len(ev)):
        s = str(round(ev[i], 4))
        plt.annotate(s, xy=(i + 1, ev[i]))

    plt.xlabel('Explained Variance')
    plt.ylabel('N of dimensions')
    plt.show()

    # aplicação da matriz
    print('-----\naplicação da matriz\n-----')
    print(pca_fitted.components_)
    # matriz projetada
    print('-----\nmatriz projetada\n-----')
    print(pca_transformed)
    print(f'Dimensões da nova matriz: {pca_transformed.shape}')
    print(f'Dimensões da matriz original: {x.shape}')


if __name__ == '__main__':
    run()
