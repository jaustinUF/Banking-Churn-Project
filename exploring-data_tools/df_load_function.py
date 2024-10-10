import pandas as pd

def df_info(df, unq_cnt_limit ):
    print(f'Dataframe shape: {df.shape}')
    column_info = {                                                # create data information file
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum(),
        "Null Count": (df.isnull().sum()),
        "% null": (df.isnull().sum()/len(df) * 100).astype(int),   # change type to integer
    #   "% null": (df.isnull().sum()/len(df) * 100).round(1),      # control decimal places
        "Unique (<15)": [df[col].nunique() if df[col].nunique() < unq_cnt_limit else ' ' for col in df.columns],
        "Unique (object)": [df[col].nunique() if df[col].dtype == 'object' else '' for col in df.columns],
        "Dtype": df.dtypes
    }
    return pd.DataFrame(column_info).reset_index(drop=True)
