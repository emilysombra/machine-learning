import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def run(path='datasets/water_potability.csv'):
    df_original = pd.read_csv(path)
    df = df_original[['Hardness', 'Conductivity', 'Organic_carbon', 'Turbidity', 'Potability']]
    for col in df:
        print(col)
        print('média:', df[col].mean())
        print('moda:', df[col].mode()[0])
        print('mediana:', df[col].median())
        print('quartil Q1:', df[col].quantile(0.25))
        print('quartil Q2:', df[col].quantile(0.5))
        print('quartil Q3:', df[col].quantile(0.75))
        print('percentil 20:', df[col].quantile(0.2))
        print('percentil 50:', df[col].quantile(0.5))
        print('percentil 70:', df[col].quantile(0.7))
        print('variância:', df[col].var())
        print('desvio padrão:', df[col].std())
        print('\n----------------')
        _, ax1 = plt.subplots()
        ax1.set_title('Boxplot ' + col)
        ax1.boxplot(df[col])
        plt.show()

    # histograma
    df.hist()
    plt.show()
    # scatter plots
    sns.pairplot(df, hue='Potability', palette='mako')
    plt.show()
    # correlação
    corr = df_original.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
    plt.show()
    # covariancia
    cov = df_original.cov()
    sns.heatmap(cov, xticklabels=cov.columns, yticklabels=cov.columns, annot=True)
    plt.show()


if __name__ == '__main__':
    run()
