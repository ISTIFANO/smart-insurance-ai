import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

path = r"C:\Users\aamir\Desktop\YC\P\mart-insurance-ai\src\infrastructure\data\assurance-maladie-68d92978e362f464596651.csv"
data = pd.read_csv(path)

# print(data.describe())

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

# cols = data.select_dtypes(include=['int64', 'float64']).columns

# for col in cols:
#     plt.figure(figsize=(6,4))
#     plt.hist(data[col], bins=20, color='skyblue', edgecolor='black')
#     plt.title(f"distribution de {col}")
#     plt.xlabel(col)
#     plt.ylabel("Frequence")
#     plt.show()

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
