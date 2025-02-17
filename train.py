#!/usr/bin/env python3

import time
import os
import pandas as pd
import numpy as np
import pickle
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import XGBModel
from darts.ad import NormScorer, KMeansScorer, PyODScorer
from darts.ad.anomaly_model.forecasting_am import ForecastingAnomalyModel
import argparse
from pyod.models.knn import KNN
from pyod.models.iforest import IForest

COLUMNS_TO_KEEP = [
    "timestamp", 
    "/agrorob/engine_state.engine_rotation_speed_rpm",
    "/agrorob/robot_state.left_front_wheel_encoder_imp",
    "/agrorob/robot_state.right_front_wheel_encoder_imp",
    "/agrorob/robot_state.left_rear_wheel_encoder_imp",
    "/agrorob/robot_state.right_rear_wheel_encoder_imp",
    "/agrorob/tool_state.fertilizer_tank1_level",
    "/agrorob/tool_state.fertilizer_tank2_level",
    "/agrorob/tool_state.hydraulic_oil_tem_celsius",
    "/agrorob/tool_state.hydraulic_oil_pressure",
    "/agrorob/engine_state.engine_coolant_temp_celsius",
    "/agrorob/engine_state.engine_fuel_level_percent",
    "/agrorob/engine_state.engine_oil_pressure_bar",
]


def main():
    # --- 0. Set up paths ---
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs_path", type=str, help="Path to the folder containing the CSV files")
    args = parser.parse_args()
    CSVS_PATH = args.csvs_path

    # --- 1. Read and clean Data ---
    dfs = []
    for csv in os.listdir(CSVS_PATH):
        if csv.endswith(".csv"):
            df = pd.read_csv(os.path.join(CSVS_PATH, csv))
            # Keep only columns we need
            df = df[COLUMNS_TO_KEEP]
            
            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp and remove duplicates
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

            # Replace empty strings and drop NaNs
            df = df.replace('', np.nan).dropna()

            # Sort columns alphabetically
            df = df.sort_index(axis=1)  
            
            # Append to list of DataFrames
            dfs.append(df)

    # --- 2. Convert DataFrames to TimeSeries ---
    series = []
    for df in dfs:
        ts = TimeSeries.from_dataframe(df, time_col='timestamp', fill_missing_dates=False)
        series.append(ts)
    
    scaler = Scaler()
    scaler.fit(series[0])  # or pick a representative series
    scaled_series = [scaler.transform(s) for s in series]

    # --- 4. Define model + scorers ---
    model = XGBModel(lags=64, random_state=42)  # or RNNModel, etc.
    scorers = [
        # NormScorer(ord=1, component_wise=True),
        KMeansScorer(k=20, window=32, component_wise=True, random_state=42),
    ]

    # You can pass the scaled series to the anomaly model
    anomaly_model = ForecastingAnomalyModel(
        model=model,
        scorer=scorers,
    )

    print("Training the anomaly model on scaled data...")
    anomaly_model.fit(scaled_series, allow_model_training=True, verbose=True)
    print("Anomaly model trained!")

    # --- 5. Save the model + scaler ---
    timestamp_str = time.strftime("%d-%m-%Y_%H-%M-%S")
    model_filename = f"anomaly_model_{timestamp_str}.pkl"

    # It's often helpful to store both the anomaly model and the scaler together
    # so you can easily load them later and apply the same transformations
    model_package = {
        "anomaly_model": anomaly_model,
        "scaler": scaler
    }

    with open(model_filename, 'wb') as f:
        pickle.dump(model_package, f)

    # For quick reference (overwrites "latest" each time)
    with open("latest_anomaly_model.pkl", 'wb') as f:
        pickle.dump(model_package, f)

    print("Anomaly model and scaler saved!")
    print("Done!")


if __name__ == "__main__":
    main()
