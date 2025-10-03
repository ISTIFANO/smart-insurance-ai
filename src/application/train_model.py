import matplotlib.pyplot as plt
import pandas as pd
import joblib as lib
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from evaluate_model import evaluate_model

path = r"C:\Users\aamir\Desktop\YC\P\mart-insurance-ai\src\infrastructure\data\assurance-maladie-68d92978e362f464596651.csv"
data = pd.read_csv(path)

X = data.drop("charges", axis=1)
y = data["charges"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


def exportpreprocessor():
    return lib.dump(preprocessor,r"C:\Users\aamir\Desktop\YC\P\mart-insurance-ai\src\domain\preprocessor.pkl")


def loadpreprocessor():
    return lib.load(r"C:\Users\aamir\Desktop\YC\P\mart-insurance-ai\src\domain\preprocessor.pkl")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0),
    "SVR": SVR()
}


for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    print(pipeline)
    r2, rmse,mae = evaluate_model(pipeline, X_test, y_test)

    print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
   

   #Definir une grille de recherche (GridSearchCV) ou aleatoire (RandomizedSearchCV) avec validation croisee (5 folds) pour les hyperparametres (ex. : pour Random Forest : nestimators, maxdepth, minsamplessplit ; pour XGBoost : learningrate, maxdepth, subsample).
# Comparer les performances des modeles avant et apres optimisation (RMSE, MAE, R²).
rfPipline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

rf_Parame = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 5, 10, 20],
    'model__min_samples_split': [2, 5, 10]
}

rfGrid = GridSearchCV(
    estimator=rfPipline,
    param_grid=rf_Parame,
    scoring='neg_root_mean_squared_error',
    cv=5
)

rfGrid.fit(X_train, y_train)

# print("\nRandom Forest -best hyperparametre :", rfGrid.best_params_)
# print(f"Random Forest - best mse cv : {-rfGrid.best_score_:.4f}")

best_rf_model = rfGrid.best_estimator_
r2_rf_tuned, rmse_rf_tuned, mae_rf_tuned = evaluate_model(best_rf_model, X_test, y_test)
print(f"Random Forest optimise : R² = {r2_rf_tuned:.4f}, RMSE = {rmse_rf_tuned:.2f}, MAE = {mae_rf_tuned:.2f}")

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(random_state=42, verbosity=0))
])

xg_Param = {
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7],
    'model__subsample': [0.7, 1.0],
    'model__n_estimators': [100, 200]
}

xgb_grid = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=xg_Param,
    scoring='neg_root_mean_squared_error',
    cv=5
)

xgb_grid.fit(X_train, y_train)

# print("\nXGBoost - best hyperparametres :", xgb_grid.best_params_)
# print(f"XGBoost - best rmse cv : {-xgb_grid.best_score_:.4f}")

bestmodelXG = xgb_grid.best_estimator_
r2XG, rmseXG, maeXG = evaluate_model(bestmodelXG, X_test, y_test)
print(f"XGBoost optimise : R² = {r2XG:.4f}, mse = {rmseXG:.2f}, MAE = {maeXG:.2f}") 

YP_RF = best_rf_model.predict(X_test)
Y_P_XG = bestmodelXG.predict(X_test)

print(y_test)
print(Y_P_XG)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest
axes[0].scatter(YP_RF, y_test - YP_RF, alpha=0.6)
axes[0].hlines(y=0, xmin=min(YP_RF), xmax=max(YP_RF), colors='red', linestyles='--')
axes[0].set_title("Residus - Random Forest")
axes[0].set_xlabel("Predictions")
axes[0].set_ylabel("Residus")

# XGBoost
axes[1].scatter(Y_P_XG, y_test - Y_P_XG, alpha=0.6, color="orange")
axes[1].hlines(y=0, xmin=min(Y_P_XG), xmax=max(Y_P_XG), colors='red', linestyles='--')
axes[1].set_title("Residus - XGBoost")
axes[1].set_xlabel("Predictions")
axes[1].set_ylabel("Residus")

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, YP_RF, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  
axes[0].set_title("Predictions vs Reelles - Random Forest")
axes[0].set_xlabel("Valeurs reelles")
axes[0].set_ylabel("Valeurs predites")

axes[1].scatter(y_test, Y_P_XG, alpha=0.6, color="orange")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_title("Predictions vs Reelles - XGBoost")
axes[1].set_xlabel("Valeurs reelles")
axes[1].set_ylabel("Valeurs predites")

plt.tight_layout()
plt.show()

r2_rf, rmse_rf, mae_rf = evaluate_model(best_rf_model, X_test, y_test)

r2_xgb, rmse_xgb, mae_xgb = evaluate_model(bestmodelXG, X_test, y_test)

results_df = pd.DataFrame({
    "Modele": ["Random Forest Optimisé", "XGBoost Optimisé"],
    "R²": [r2_rf, r2_xgb],
    "RMSE": [rmse_rf, rmse_xgb],
    "MAE": [mae_rf, mae_xgb]
})

print(results_df)


#save model 

lib.dump(bestmodelXG, r"C:\Users\aamir\Desktop\YC\P\mart-insurance-ai\src\domain\best_model_xg.pkl")


def exportModel():
    return lib.load(r"C:\Users\aamir\Desktop\YC\P\mart-insurance-ai\src\domain\best_model_xg.pkl")

fake_data = pd.DataFrame({
        "age":21,
        "sex": "female",
        "bmi": 25,
        "children": 2,
        "smoker": "yes",
        "region": "southwest" 
        })
    
input_preprocessed = preprocessor.transform(fake_data)

prediction = bestmodelXG.predict(input_preprocessed)
print(f"Charge estimee : {prediction[0]:.2f}")