import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set paths relative to this script
BASE_DIR=os.path.dirname(__file__)
DATA_PATH=os.path.join(BASE_DIR, "../data/mexico_air_quality_cleaned.csv")
RESULTS_DIR=os.path.join(BASE_DIR, "../results")

def prepare_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert relevant columns to numeric and drop any invalid rows."""
    numeric_cols = ["PM2.5", "PM10", "O3", "CO", "Temperature", "Humidity"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=numeric_cols, inplace=True)
    return df

def plot_pm25_trend(df: pd.DataFrame):
    #Plot Daily Avg. PM2.5 levels over time.
    #Saves the Plot in the Results folder.

    df["date"]=pd.to_datetime(df["datetime"])
    df["date_only"]=df["date"].dt.date    
    daily_avg=df.groupby("date_only")["PM2.5"].mean()

    plt.figure(figsize=(12,6))
    plt.plot(daily_avg.index, daily_avg.values,linestyle="-")
    plt.title("Daily Average PM2.5")
    plt.xlabel("Date")
    plt.ylabel("PM2.5")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "eda_pm25_trend.pdf"))
    plt.close()


def plot_correlation(df:pd.DataFrame):
    #Plots a correlation heatmap for all numeric features
    #Drop Dayofweek(categorical data)
    #Saves the Plot in the Results folder.

    df_num=df.select_dtypes(include=[np.number])
    df_num=df_num.drop(columns=["dayofweek"])
    correlation=df_num.corr()

    fig,ax=plt.subplots(figsize=(12,10))
    corr_ax=ax.matshow(correlation,cmap="coolwarm")
    fig.colorbar(corr_ax)

    ticks=np.arange(len(correlation.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(correlation.columns, rotation=90)
    ax.set_yticklabels(correlation.columns)

    for (i,j), val in np.ndenumerate(correlation.values):
        ax.text(i,j, f"{val:.2f}", ha="center", va="center",color="black")

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "eda_correlation_heatmap.pdf"))
    plt.close()

def plot_histogram_pm25(df: pd.DataFrame):
    #Plot a histogram of daily Avg Pm2.5 values.
    #Shows distribution and frequency of pollution.
    #Saves the Plot in the Results folder.
    df["date"]=pd.to_datetime(df["datetime"])
    df["date_only"]=df["date"].dt.date
    daily_avg2=df.groupby("date_only")["PM2.5"].mean()
    
    plt.figure(figsize=(10,6))
    plt.hist(daily_avg2.values, bins=30, edgecolor="black")
    plt.title("PM2.5 Histogram")
    plt.xlabel("PM2.5")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(RESULTS_DIR, "eda_pm25_histogram.pdf"))
    plt.close()

if __name__=="__main__":
    df=pd.read_csv(DATA_PATH)
    print(df.columns)
    plot_pm25_trend(df)
    plot_correlation(df)
    plot_histogram_pm25(df)