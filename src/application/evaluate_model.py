from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import numpy as np

def evaluate_model(pipeline, X_test, y_test):

    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    return r2, rmse,mae
