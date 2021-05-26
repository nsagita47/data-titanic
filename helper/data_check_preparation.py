from os import path
import pandas as pd 
import numpy as np

def read_data(PATH):
    '''
    Read data from dataset from path
   
    Parameters
    ----------
    PATH : str
        path source of training data, csv.
    
    Returns
    -------
    data : pd.DataFrame
        Data for modeling
    '''
    data = pd.read_csv(PATH)
    
    return data

def check_and_set_columns(data_input, COLUMN):
    '''
    Check data input consistency with predefined COLUMN
    Set data input columns as COLUMN
    
    Parameters
    ----------
    data_input: pd.DataFrame
        DataFrame for modeling
    COLUMN: set
        A set of columns which will be used for modeling
        
    Returns
    -------
    data_input: pd.DataFrame
        Checked dataset for columns consistency
    '''
    COLUMN = set(COLUMN.keys())
    columns_in_data = set(data_input.columns)
    
    if not COLUMN.issubset(columns_in_data):
        with open("warning_msg.txt", "a") as writer:
            writer.write("There is at least one column not in the data")
        raise ValueError("There is at least one column not in the data")
        
    data_input = data_input[list(COLUMN)]
    return data_input

def set_dtypes(data_input, KOLOM):
    '''
    Check data input datatypes consistency with predefined DTYPES
    Set data datatypes as DTYPE
    
    Parameters
    ----------
    data_input: pd.DataFrame
        DaraFrame for modeling
    
    Returns
    -------
    data: pd.DataFrame
        Checked dataset for columns consistency
    '''
    data = data_input.astype(KOLOM)
    return data

def check_missing_passenger_id(data_input):
    '''
    Check is there any missing passenger_id in observation
    The row with missing passenger_id will be dropped to artifacts
    
    Parameters
    ----------
    data_input: pd.DataFrame
        DaraFrame for modeling
    
    Returns
    -------
    data_input: pd.DataFrame
        Checked dataset for columns consistency
    '''
    
    null_order = data_input[data_input["PassengerId"].isnull()]
    
    if not null_order.shape[0] ==0:
        with open("warning_msg.txt", "a") as writer:
            writer.write("You have missing PassengerId in the data")
        
        null_order.to_csv("artifacts/missing_PassengerId.csv")
    
    
    data_input = data_input[data_input["PassengerId"].notnull()]
    return data_input

def check_read_data_success(data_input):
    '''
    Sanity check for data success
    
    Parameters
    ----------
    data
    
    '''
    if not data_input.notnull().sum().sum() > 0:
        with open("warning_msg.txt", "a") as writer:
            writer.write("You have missing values in full in at least one column")
            
    return data_input

def read_and_check_data(path, column):
    """
    Read and checking data
    
    Parameters
    ----------
        PATH (['string']): file location
        COLUMN (['string']): list of columns
    
    """
    
    print("start import data")
    df = read_data(path)
    print("done import data")
    print("start shecking and set columns")
    df = check_and_set_columns(df, column)
    print("done checking and set columns")
    print("start set dtypes")
    df = set_dtypes(df, column)
    print("done set dtypes")
    print("start checking missing passenger_id")
    df = check_missing_passenger_id(df)
    print("done checking missing passenger_id")
    print("start checking read data success")
    df = check_read_data_success(df)
    print("done checking read data success")
    
    return df
    