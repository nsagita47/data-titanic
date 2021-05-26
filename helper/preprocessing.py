import pandas as pd
import numpy as np
from sklearn import preprocessing

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of categorical column names e.g. nominal, ordinal data type
        encoding_type: type of encoding e.g. label, one_hot
        handle_na: handle the missing values or not e.g. True/False
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type  = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.one_hot_encoders = None

        if self.handle_na is True:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

#     def _one_hot_encoding(self):
#          one_hot_encoders = preprocessing.OneHotEncoder()
#          one_hot_encoders.fit(self.df[self.cat_feats].values)
#          return one_hot_encoders.transform(self.df[self.cat_feats].values)

    def _get_dummies(self):
        self.output_df = pd.get_dummies(self.df, columns=self.cat_feats, dummy_na=True)
        return self.output_df

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "one_hot":
            return self._get_dummies()
        else:
            raise Exception("Encoding type not supported!")
         
    def transform(self, dataframe):
        if self.handle_na is True:
            for c in self.cat_feats:
                dataframe[:, c] = dataframe.loc[:, c].astype(str).fillna("-9999999")

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe
        elif self.enc_type == "one_hot":
            self.one_hot_encoders.transform(dataframe[self.cat_feats].values)
    
        else:
            raise Exception("Encoding type not supported!") 
            
            
def standard_scaler(df: pd.DataFrame):
    """Scaling standard scaler transform."""
    index_cols = df.index
    scaler = preprocessing.StandardScaler()
    np_scaler = scaler.fit_transform(df)
    df_transformed = pd.DataFrame(
        np_scaler, index=index_cols, columns=df.columns
    )

    return scaler, df_transformed

def polynomial_transform(df: pd.DataFrame, degree: int, interaction_only: bool):
    """Transform to polynomial."""
    df_idx = df.index
    poly = preprocessing.PolynomialFeatures(degree=degree, interaction_only=interaction_only)
    np_clean_polynomial = poly.fit_transform(df)
    column_name = poly.get_feature_names(df.columns)
    df_clean_polynomial = pd.DataFrame(
        np_clean_polynomial, columns=column_name
    )
    df_clean_polynomial.set_index(df_idx, inplace=True)
    df_clean_polynomial.drop(columns=["1"], inplace=True)
    return df_clean_polynomial