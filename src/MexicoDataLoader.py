import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import pandas as pd
import os

BASE_DIR=os.path.dirname(__file__)
DATA_PATH=os.path.join(BASE_DIR, "../data/mexico_air_quality_cleaned.csv")
MODEL_DIR=os.path.join(BASE_DIR, "../model")

def create_sequences(df, feature_cols, target_col="PM2.5", seq_len=24):
    #Create Sequences of past time steps for LSTM.

    X,y=[],[]
    data=df[feature_cols].values.astype(np.float32)
    target=df[target_col].values.astype(np.float32)
    
    for i in range(seq_len,len(df)):
        X.append(data[i-seq_len:i])
        y.append(target[i])
    
    X=np.array(X)      
    return X, y

def get_sequence_loaders(df, feature_cols, seq_len=24, batch_size=32):
    #Data Loader to prepare Training with the respective train/val/test splits.
    X,y=create_sequences(df, feature_cols, seq_len=seq_len)
    
    n=len(X)
    train_idx =int(n*0.8) #80 percent training
    val_idx =int(n*0.9)   #10 percent val, 10 percent test

    X_train, y_train=X[:train_idx], y[:train_idx]
    X_val, y_val=X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test=X[val_idx:], y[val_idx:]

    #Normalize by training set
    mean=X_train.mean(axis=(0,1))
    std=X_train.std(axis=(0,1))
    std[std==0] = 1

    X_train=(X_train - mean)/std
    X_val=(X_val - mean)/std
    X_test=(X_test - mean)/std

    #Convert arrays to tensors
    train_dataset=TensorDataset(torch.tensor(X_train),torch.tensor(y_train).unsqueeze(1))
    val_dataset=TensorDataset(torch.tensor(X_val),torch.tensor(y_val).unsqueeze(1))
    test_dataset=TensorDataset(torch.tensor(X_test),torch.tensor(y_test).unsqueeze(1))

    #Use DataLoader for batching during training
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader, val_loader, test_loader, mean, std


if __name__=="__main__":
    df_mexico=pd.read_csv(DATA_PATH)
    feature_cols = [col for col in df_mexico.columns if col not in ["datetime", "PM2.5"]]
    train_loader, val_loader,test_loader,mean,std=get_sequence_loaders(df_mexico,feature_cols=feature_cols,seq_len=24,batch_size=32)

    scaler={"mean": mean, "std": std}
    with open(os.path.join(MODEL_DIR, "scaler.pk"), "wb") as f:
        pickle.dump(scaler, f) #save the scaler for use in the app later

    X_batch, y_batch=next(iter(train_loader)) 
    
    #check shapes 
    print("X Batch shape:",X_batch.shape)
    print("y Batch shape:",y_batch.shape)
