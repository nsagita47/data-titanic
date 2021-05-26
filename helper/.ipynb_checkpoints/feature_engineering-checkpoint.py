import pandas as pd 
import numpy as np
from helper.preprocessing import CategoricalFeatures 

def feature_engineering(df):
    # encode gender variable
    df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "male" else 0)

    # has cabin or not
    df["Has_Cabin"] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # split and cleansing title name
    df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    title_names = (df["Title"].value_counts() < 10)
    df["Title"] = df["Title"].apply(lambda x: "Misc" if title_names.loc[x] == True else x)

    # impute age by title    
    age_by_title = df.groupby(["Title"])["Age"].mean()
    df["Age"] = df.apply(lambda row: age_by_title[row["Title"]] if np.isnan(row["Age"]) else row["Age"], axis=1)

    # impute fare by passenger class
    fare_by_class = df.groupby(["Pclass"])["Fare"].median()
    df["Fare"] = df.apply(lambda row: fare_by_class[row["Pclass"]] if np.isnan(row["Fare"]) else row["Fare"], axis=1)

    # add several new variables
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    df["Fare_Bin"] = pd.qcut(df["Fare"], 4)
    df["Age_Bin"] = pd.qcut(df["Fare"], 4)
    df["Is_Alone"] = df["Family_Size"].apply(lambda x: 1 if x <= 1 else 0)
    
    # set index and drop unused columns
    df_transformed = df.set_index("PassengerId")
    df_transformed = df_transformed.drop(["Name","Ticket","Cabin"], axis=1)

    # applied CategoricalFeatures()
    cols = [c for c in df_transformed.columns if (df_transformed[c].dtype != "float" and df_transformed[c].dtype != "int")]
    cat_feats = CategoricalFeatures(df_transformed, categorical_features=cols, encoding_type="one_hot", handle_na=True)
    df_transformed = cat_feats.fit_transform()

    return df_transformed