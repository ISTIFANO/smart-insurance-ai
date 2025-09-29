import pandas as pd

path = r"C:\Users\aamir\Desktop\YC\P\mart-insurance-ai\src\infrastructure\data\assurance-maladie-68d92978e362f464596651.csv"
data = pd.read_csv(path)

# print(data.describe())

statis = pd.DataFrame({
    "max": data.max(numeric_only=True),
     "min" :  data.min(numeric_only=True),
      "Mediane" : data.median(numeric_only=True),
       "Moyenne " : data.mean(numeric_only=True),
        "ecarts types" : data.std(numeric_only=True)})

print(statis)
