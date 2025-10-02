import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
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
   

   #Définir une grille de recherche (GridSearchCV) ou aléatoire (RandomizedSearchCV) avec validation croisée (5 folds) pour les hyperparamètres (ex. : pour Random Forest : nestimators, maxdepth, minsamplessplit ; pour XGBoost : learningrate, maxdepth, subsample).
# Comparer les performances des modèles avant et après optimisation (RMSE, MAE, R²).
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