import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Path of this script
BASE_DIR=os.path.dirname(os.path.abspath(__file__))

# Data folder is one level up from src
DATA_PATH=os.path.join(BASE_DIR, "../data/mexico_air_quality_cleaned.csv")
DATA_PATH=os.path.abspath(DATA_PATH)  # convert to absolute path

# Model folder
MODEL_DIR=os.path.join(BASE_DIR, "../model")
MODEL_DIR=os.path.abspath(MODEL_DIR)

#Results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")

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


def train_model(model: nn.Module, train_loader: DataLoader, val_loader:
DataLoader, test_loader: DataLoader):
    #train the model with specified hyperparameters
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    epochs=200
    lr=0.01

    optimizer=torch.optim.Adam(model.parameters(),lr=lr)#use Adam Optimizer
    loss_function=nn.MSELoss()#Mean squared Error Loss

    for epoch in range(epochs):
        model.train()
        total_train_loss=0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch= X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()                       #clear gradients
            outputs=model(X_batch)                      #forward pass
            loss=loss_function(outputs, y_batch)        
            loss.backward()                             #Backpropagation
            optimizer.step()                            #Update the weights
            total_train_loss+=loss.item()*X_batch.size(0) #Sum up batch losses

        avg_train_loss=total_train_loss/len(train_loader.dataset)#Average train loss

        model.eval()
        total_val_loss=0
        with torch.no_grad():
            for X_val,y_val in val_loader:
                X_val,y_val=X_val.to(device), y_val.to(device)
                val_outputs=model(X_val)
                val_loss=loss_function(val_outputs,y_val)
                total_val_loss+=val_loss.item()*X_val.size(0)

        avg_val_loss=total_val_loss/len(val_loader.dataset)

        if epoch %50==0:
            print(f"Epoch {epoch}: Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt")) #save model later for the app

    model.eval()
    y_true, y_pred=[],[]
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test=X_test.to(device)
            predictions=model(X_test).cpu().numpy()
            y_pred.append(predictions)
            y_true.append(y_test.numpy())

    y_true=np.concatenate(y_true).flatten()
    y_pred=np.concatenate(y_pred).flatten()

    plt.figure(figsize=(10,6))
    plt.plot(y_pred,label="Predicted",color="blue" ,alpha=0.7)
    plt.plot(y_true, label="Actual", color="orange",alpha=0.7)
    plt.title("PM2.5 Prediction")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_prediction.pdf"))#save plot in the results
    plt.close()

if __name__=="__main__":
    import pandas as pd
    from src.MexicoDataLoader import get_sequence_loaders
    print("Import works!", get_sequence_loaders)
    
    df_mexico=pd.read_csv(DATA_PATH)
    feature_cols=[col for col in df_mexico.columns if col not in ["datetime", "PM2.5"]]
    print(df_mexico.columns)
    train_loader, val_loader,test_loader,mean,std=get_sequence_loaders(df_mexico,feature_cols=feature_cols,seq_len=24,batch_size=32)
    
    X_batch, y_batch=next(iter(train_loader))
    input_dim=X_batch.shape[2]  
    model=PM_LSTM(input_dim)

    train_model(model, train_loader, val_loader, test_loader)