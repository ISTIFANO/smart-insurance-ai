import pandas as pd
import seaborn as sns
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = r"C:\Users\aamir\Desktop\YC\P\mart-insurance-ai\src\infrastructure\data\assurance-maladie-68d92978e362f464596651.csv"
data = pd.read_csv(path)

print(data.describe())

#  analyse descriptive des données (moyennes, médianes, écarts-types pour les variables numériques ; fréquences pour les catégoriques)
statis = pd.DataFrame({
    "max": data.max(numeric_only=True),
     "min" :  data.min(numeric_only=True),
      "Mediane" : data.median(numeric_only=True),
       "Moyenne " : data.mean(numeric_only=True),
        "ecarts types" : data.std(numeric_only=True)})

print(statis)

categorical = data.select_dtypes(include=['object']).columns

for colu in categorical :
    print(data[colu].value_counts())
    print(data[colu].unique())


#  les valeurs manquantes et les doublons

print(data.isnull().sum())

print(data.duplicated().sum())

# Analyser la distribution des variables numériques (ex. : histogrammes avec Matplotlib/Seaborn).

cols = data.select_dtypes(include=['int64', 'float64']).columns

for col in cols:
    plt.figure(figsize=(6,4))
    plt.hist(data[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(f"distribution de {col}")
    plt.xlabel(col)
    plt.ylabel("Frequence")
    plt.show()

# les relations entre variables à l'aide de matrices de corrélation et de visualisations (ex. : pairplots ou heatmaps).

mat_cor = data.corr(numeric_only=True)

plt.figure(figsize=(9,7))
sns.heatmap(mat_cor,annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.show()
    

sns.pairplot(data,hue='age')
plt.show()


# Gestion des valeurs manquantes

numeric_values = data.select_dtypes(include=["int64","float64"])
object_values = data.select_dtypes(include=["object"])

for col_num in numeric_values : 
    print("les valeur qui onrt vide est " , data[col_num].isnull().sum())
    data[col_num].fillna(data[col_num].median(),inplace=True)



for col_obj in object_values :
    print("les valeur qui onrt vide est " , data[col_obj].isnull().sum())
    data[col_obj].fillna(data[col_obj].mode()[0],inplace=True)

# Suppression des doublons

data.drop_duplicates(inplace=True)

# Détection et gestion des valeurs aberrantes : Utiliser des techniques statistiques (ex. : boîte à moustaches avec Seaborn, z-score > 3, ou IQR pour identifier les outliers) et gérer les lignes contenant des valeurs aberrantes (suppression).

numeric_cols = data.select_dtypes(include=["int64","float64"]).columns

for col in numeric_cols:
    Q1 = data[col].quantile(0.25)  
    Q3 = data[col].quantile(0.75) 
    IQR = Q3 - Q1                 
    
    lowerbound = Q1 - 1.5 * IQR
    upperbound = Q3 + 1.5 * IQR
    
    outliers = data[(data[col] < lowerbound) | (data[col] > upperbound)]
    print(f"{col} → {len(outliers)} valeurs aberrantes detecte")
    
data01 = data[(data[col] >= lowerbound) & (data[col] <= upperbound)]

# Z score methode 

z_score =  np.abs(zscore(data[numeric_cols]))

mask = (z_score<3).all(axis=1)

print("nombre de valeurs aberrantes :", (~mask).sum())

data02 = data[mask]

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=data02[col])
    plt.title(f"Boîte a moustaches pour {col}")
    plt.show()

encode = LabelEncoder()

for column01 in ["smoker","children","region","sex"] :
    data[column01] = encode.fit_transform(data[column01])

print(data.head())

# Diviser les données en ensembles d'entraînement et de test (80% / 20%) avec traintestsplit de Scikit-learn.

X = data.drop("children", axis=1)
y = data["children"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Taille X_train :", X_train.shape)
print("Taille X_test  :", X_test.shape)
print("Taille y_train :", y_train.shape)
print("Taille y_test  :", y_test.shape)


# Appliquer une normalisation (MinMaxScaler) ou une standardisation (StandardScaler) sur les variables numériques pour harmoniser les échelles.

numeric_col = data.select_dtypes(include=["int64", "float64"]).columns

minMax_scaler = MinMaxScaler()
data_minMax =  data.copy()
data_minMax[numeric_col] = minMax_scaler.fit_transform(data[numeric_col])

std_scaler = StandardScaler()
data_std =  data.copy()

data_std[numeric_col] = std_scaler.fit_transform(data[numeric_col])


print("=== Normalisation MinMax ===")
print(data_minMax.head())

print("=== Standardisation ===")
print(data_std.head())

