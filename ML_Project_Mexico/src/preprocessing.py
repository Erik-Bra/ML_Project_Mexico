import numpy as np
import pandas as pd

def preprocess_mexico_air_quality(csv_path:str, output_csv: str=None)-> pd.DataFrame:
    #Function to load and clean the mexico_air_quality dataset.
    
    df=pd.read_csv(csv_path) #Read CSV File
    

    #Change names of columns for easier handling later
    df.rename(columns={      
        "PM2.5 [ug/m3]": "PM2.5",
        "PM10[ug/m3]": "PM10",
        "Ozone [ppb]": "O3",
        "Carbon_Monoxide [ppb]": "CO",
        "Temperature [°C]": "Temperature",
        "Relative_Humidity [%]": "Humidity",
    }, inplace=True)

    #Convert Timestamp to Datetime to extract hours, dayofweek and month later
    df["datetime"]=pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S") 
    df.set_index("datetime",inplace=True)
    df.drop(columns=["Timestamp"],inplace=True)
    
    # Fill missing numerical values
    num_cols=df.select_dtypes(include=[np.number]).columns
    df[num_cols]=df[num_cols].interpolate(method="time").fillna(df[num_cols].median())#fill empty columns with median
    df.dropna(inplace=True)

    #Now Extract the time features
    df["hour"]=df.index.hour
    df["dayofweek"]=df.index.dayofweek
    df["month"]=df.index.month

    #Remove outliers and measuring errors from the dataset
    num_cols=df.select_dtypes(include=[np.number]).columns
    lower,upper=0.01,0.99
    for col in num_cols:
        low_val=df[col].quantile(lower)
        high_val=df[col].quantile(upper)
        df=df[(df[col]>=low_val) & (df[col]<=high_val)]

    if output_csv:
        df.to_csv(output_csv)

    return df.reset_index()


if __name__=="__main__":
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "../data/AirQualityIBEROCDMX.csv")
    clean_csv = os.path.join(os.path.dirname(__file__), "../data/mexico_air_quality_cleaned.csv")

    df_cleaned=preprocess_mexico_air_quality(csv_path, clean_csv)
    print((df_cleaned['PM2.5'] == df_cleaned['PM10']).sum(), len(df_cleaned))
    print(df_cleaned.head())
