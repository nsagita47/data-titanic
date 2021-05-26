import pandas as pd
import pickle

# from sklearn.utils.extmath import randomised_svd

from helper.data_check_preparation import read_and_check_data
from helper.feature_engineering import feature_engineering
from helper.constant import TRAIN_COLUMN, PATH
from helper.models import LINEAR_MODEL_CLF
from helper.preprocessing import standard_scaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score,recall_score, f1_score

def train_model():
#     read data
    df = read_and_check_data(PATH, TRAIN_COLUMN)
    
#     feature engineering
    df_transformed = feature_engineering(df)
    x = df_transformed.drop(columns=['Survived'])
    y = df_transformed['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)
    
    train_scaler, X_train_scaled = standard_scaler(X_train)
    pickle.dump(train_scaler, open("artifacts/train_scaler.pkl", "wb"))
    X_test_scaled = train_scaler.transform(X_test)
    
#     modelling
    clf_model = LINEAR_MODEL_CLF["logreg_cv"]
    clf_model.fit(X_train_scaled, y_train)
    
    y_pred_class = clf_model.predict(X_test_scaled)
    y_pred_proba = clf_model.predict_proba(X_test_scaled)[:, 1]
    
    pickle.dump(clf_model, open("artifacts/logreg_model.pkl", "wb"))
    
#     performance

    print("------------------------------")
    print("Model Performance")
    print(f"ROC AUC : {roc_auc_score(y_test, y_pred_proba)}")
    print(f"Recall : {recall_score(y_test, y_pred_class)}")
    print(f"Precision : {precision_score(y_test, y_pred_class)}")
    print(f"F1 Score : {f1_score(y_test, y_pred_class)}")
    
    
if __name__ == "__main__":
    print("START RUNNING PIPELINE")
    train_model()
    print("FINISH RUNNING PIPELINE")
    