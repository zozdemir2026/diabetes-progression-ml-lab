#Calculating metrics 

from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
def regression_scores(model, X_subset, y_true):
    pred= model.predict(X_subset)
    r2=r2_score(y_true, pred)
    mae=mean_absolute_error(y_true, pred)
    mape=100 * mean_absolute_percentage_error(y_true, pred)
    return {
         "R2":r2,
         "MAE":mae,
         "MAPE (%)": mape
    }

