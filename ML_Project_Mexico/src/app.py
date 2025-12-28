import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from shiny import reactive
from shiny.express import render, ui, input
import numpy as np

#Quickly define the LSTM class again for use later
class PM_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_layers=2):
        super().__init__()
        self.lstm=nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc=nn.Linear(hidden_dim, 1) #Final layer maps hidden state to PM 2.5 Pred

    def forward(self, x):
        out, _=self.lstm(x)
        out=out[:, -1, :]  
        out=self.fc(out)
        return out

#set the reactive values
df_data=reactive.Value(None)
model_data=reactive.Value(None)
scaler_data=reactive.Value(None)

#Design the UI Layout and Sidebar of the App
with ui.layout_sidebar():
    with ui.sidebar():
        ui.input_file("file", "Upload File", multiple=False)
        ui.input_selectize("Select_Pollutants", "Select Pollutants", choices=[], multiple=True)
        ui.input_slider("window", "Smoothing window (days)", min=1, max=30, value=1)
        ui.input_checkbox("show_pred", "Show PM2.5 Predicition")
    
        with ui.panel_conditional("input.show_pred==true"):
            ui.input_file("Scaler", "Upload Scaler")
            ui.input_file("Model", "Upload Model")
    with ui.navset_pill():
        with ui.nav_panel("Pollution Levels"):
            @render.plot
            #Define how the Plots gets visualized
            def pollutant_plot():
                df=df_data.get()
                if df is None:
                    return

                window=input.window()
                selected_columns=input.Select_Pollutants()

                #Convert datetime column to index
                if "datetime" in df.columns:
                    df["datetime"]=pd.to_datetime(df["datetime"])
                    df=df.set_index("datetime")

                #Plotting
                fig,ax=plt.subplots(figsize=(12,7))
                ax.set_facecolor("#cce6ff")
                ax.grid(True, color="white", linewidth= 1.0, linestyle="--")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                #Plot selected Pollutant
                for col in selected_columns:
                    if col not in df.columns:
                        continue
                    df_plot=df[[col]].rolling(window, min_periods=1).mean()
                    ax.plot(df_plot.index, df_plot[col], label=col, alpha=0.6,linewidth=1.0)  
                
                #Define Predictions Plot for PM 2.5 if ticked in App
                if input.show_pred() and model_data.get() is not None and scaler_data.get()is not None:
                    model=model_data()
                    scaler=scaler_data()

                    feature_cols=get_feature_columns(df)
                    mean=scaler["mean"]
                    std=scaler["std"]
                    std_fixed=np.where(std==0,1,std)

                    SEQ_LEN=24 #seq length during training

                    #Scale features using the trainig statistics
                    X_raw=df[feature_cols].values.astype(np.float32)
                    X_scaled=(X_raw - mean) / std_fixed

                    #If not enough data to form a full seq
                    if len(X_scaled)<=SEQ_LEN:
                        return fig

                    #Sliding window
                    X_seq = []
                    for i in range(SEQ_LEN, len(X_scaled)):
                        X_seq.append(X_scaled[i-SEQ_LEN:i])

                    X_seq=np.array(X_seq)   # (N, 24, num_features)
                    X_tensor=torch.tensor(X_seq, dtype=torch.float32)
                    with torch.no_grad():
                        preds=model(X_tensor).squeeze().numpy()
                    
                    #Align preds with the timestamps
                    pred_index=df.index[SEQ_LEN:]
                    preds_series=pd.Series(preds, index=pred_index)
                    preds_smoothed=preds_series.rolling(window,min_periods=1).mean()
                    ax.plot(preds_smoothed.index, preds_smoothed, label="PM2.5 Prediction",color="Red", linewidth=0.5, linestyle="--", alpha=0.7)                
                
                xmin=df.index.min()
                xmax=df.index.max()
                ax.set_xlim(xmin,xmax)
                
                #Plotting
                ax.set_xlabel("Datetime")
                ax.set_ylabel("Pollutant Value (Rolling Avg)")
                ax.set_title("Pollutant Levels")
                ax.legend()
                
                fig.autofmt_xdate()
                fig.tight_layout()
                return fig

#Loading CSV
@reactive.effect
def load_file():
    fileinfo=input.file()
    if fileinfo:
        df=pd.read_csv(fileinfo[0]["datapath"])
        df_data.set(df)
        update_selects(df)

#Loading Scaler
@reactive.effect
def load_scaler():
    fileinfo=input.Scaler()
    if not fileinfo:
        scaler_data.set(None)
        return
    with open(fileinfo[0]["datapath"], "rb")as f:
        scaler=pickle.load(f)
    scaler_data.set(scaler)

#Loading Model
@reactive.effect
def load_Model():
    fileinfo=input.Model()
    df=df_data()
    if not fileinfo or df is None:
        model_data.set(None)
        return
    
    feature_cols=get_feature_columns(df)
    input_dim=len(feature_cols)
    model=PM_LSTM(input_dim)
    state_dict=torch.load(fileinfo[0]["datapath"], map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    model_data.set(model)

#Update selected columns (Pollutants)
def update_selects(df):
    columns=df.columns.tolist()
    excluded_columns=["datetime", "hour", "dayofweek","month","Temperature","Humidity"]
    pollutant_columns=[col for col in columns if col not in excluded_columns]
    selected=["PM2.5"] if "PM2.5" in columns else[]
    ui.update_selectize("Select_Pollutants", choices=pollutant_columns, selected=selected)

def get_feature_columns(df):
    excluded_columns=["datetime","PM2.5"]
    return [col for col in df.columns if col not in excluded_columns]