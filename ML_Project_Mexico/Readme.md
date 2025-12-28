# PM2.5 Prediction in Mexico City

## Project Overview
This project predicts PM2.5 air pollution levels in Mexico City using an LSTM neural network. It includes a complete workflow:

1. Data preprocessing and cleaning
2. Exploratory Data Analysis (EDA) with plots
3. LSTM model training
4. Interactive app for visualization and PM2.5 predictions

The project started as a **university assignment**, but I extended it independently by using **different data**, **custom LSTM architecture**, **enhanced training procedures**, and **interactive visualizations**.

---

## Step-by-Step Description

### 1. Data Preprocessing
- **Script:** `src/preprocessing.py`
- **Actions:**
  - Loaded Mexico City air quality dataset (`AirQualityIBEROCDMX.csv`)
  - Renamed columns for easier handling (e.g., `"PM2.5 [ug/m3]" → "PM2.5"`)
  - Converted timestamps to `datetime` and extracted features: `hour`, `dayofweek`, `month`
  - Filled missing numeric values using **time interpolation** and **median imputation**
  - Removed **outliers** (1st and 99th percentile)
  - Saved cleaned dataset as `mexico_air_quality_cleaned.csv`

- **Result:** Cleaned, ready-to-use dataset for modeling.

---

### 2. Exploratory Data Analysis (EDA)
- **Script:** `src/eda.py`
- **Actions:**
  - Converted relevant columns to numeric types
  - Plotted **daily PM2.5 trends**
  - Plotted **correlation heatmap** for all numeric features
  - Plotted **histogram of daily PM2.5** to show distribution

- **Key insights:**
  - PM2.5 and PM10 are **very strongly correlated (~0.9999)**
  - Other pollutants and weather features may help slightly improve prediction

- **Output:** Plots saved in `results/` folder (`.pdf` format)

---

### 3. Sequence Creation for LSTM
- **Script:** `src/MexicoDataLoader.py`
- **Actions:**
  - Created sequences of length 24 (past 24 hours) for LSTM training
  - Split data into **train (80%)**, **validation (10%)**, and **test (10%)**
  - Normalized features using **training set statistics**
  - Saved scaler for later use in the app (`model/scaler.pk`)

- **Result:** `DataLoader` objects for PyTorch training

---

### 4. LSTM Model Training
- **Script:** `src/Model_Training.py`
- **Actions:**
  - Defined custom LSTM class (`PM_LSTM`) with:
    - 2 LSTM layers
    - Hidden size 16
    - Dropout 0.2
    - Fully connected output layer
  - Trained model using **MSE loss** and **Adam optimizer**
  - Trained for 200 epochs, tracked **train & validation loss**
  - Saved trained model to `model/model.pt`
  - Plotted **predicted vs actual PM2.5** values in `results/`

- **Observation:** Model performs well, mainly due to **strong correlation between PM2.5 and PM10**.

---

### 5. Interactive App
- **Script:** `src/app.py`
- **Actions:**
  - Built with **Shiny Express for Python**
  - Features:
    - Upload CSV files for air quality data
    - Select pollutants for visualization
    - Optional: Upload trained LSTM model and scaler
    - Show rolling average plots of pollutants
    - Show PM2.5 predictions using sliding window sequences
  - Supports **dynamic updates** with reactive programming

- **Result:** Fully interactive visualization for data exploration and PM2.5 forecasting.

---

## Project Structure

ML_Project_Mexico/
├─ src/
│ ├─ preprocessing.py # Data cleaning
│ ├─ eda.py # EDA plots
│ ├─ MexicoDataLoader.py # Sequence creation & DataLoaders
│ ├─ model.py # PM_LSTM class (optional)
│ ├─ Model_Training.py # LSTM training
│ └─ app.py # Interactive visualization
├─ data/ # Raw and cleaned datasets
├─ model/ # Saved model and scaler
├─ results/ # EDA and prediction plots
└─ README.md


---

## Dependencies

- Python 3.11+
- pandas, numpy
- matplotlib
- torch (PyTorch)
- Shiny Express for Python
- pickle (for saving scalers)

---

## Dataset

- **Source:** `AirQualityIBEROCDMX.csv` (Mexico City air quality dataset) #https://data.mendeley.com/datasets/gjvrn32zbm/2
- Contains: `PM2.5`, `PM10`, `O3`, `CO`, `Temperature`, `Humidity`  
- Time features: hour, day of week, month  

---

## Notes & Insights

- PM2.5 predictions are **strongly influenced by PM10** because of their high correlation.  
- The LSTM uses **past 24-hour sequences** to predict PM2.5.  
- Additional pollutants and weather features may help slightly improve predictions.  
- Visualizations include **daily trends, correlation heatmaps, and histograms**.  

---

## How to Run

Run all commands from the **project root** (`ML_Project_Mexico/`):

1. **Preprocess the data**: clean the raw CSV and generate the cleaned dataset

python -m src.preprocessing

2. **Run Exploratory Data Analysis (EDA):** generate plots like daily PM2.5 trends, correlation heatmaps, and histograms

python -m src.eda

3. **Data Loader**

python -m src.MexicoDataLoader

3. **Train the LSTM model:** create sequences, train the LSTM, and save the model and scaler

python -m src.Model_Training

4. **Run the interactive Shiny app:** visualize pollutant levels and PM2.5 predictions

python -m shiny run src/app.py