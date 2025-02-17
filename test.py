from train import COLUMNS_TO_KEEP

import pandas as pd
from darts.ad import NormScorer, KMeansScorer, QuantileDetector, ThresholdDetector
from darts import TimeSeries
import time
from darts.ad.detectors import IQRDetector

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pickle
import os


TEST_CSV_PATH = "/ros_ws/src/anomaly_detection/data/test/rosbag2_2024_08_28-10_20_12.csv"
MODEL_PATH = "/ros_ws/latest_anomaly_model.pkl"
COLUMN = "/agrorob/engine_state.engine_rotation_speed_rpm"

# Prepare the test data
test_df = pd.read_csv(TEST_CSV_PATH)
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
test_df = test_df[COLUMNS_TO_KEEP]
test_df = test_df.replace('', np.nan)
test_df = test_df.dropna()
# Sort columns alphabetically
test_df = test_df.sort_index(axis=1)
# print(test_df.columns)

# Introduce fake anomalies to "agrorob_engine_state__engine_coolant_temp_celsius"
anomaly_start = 2000
anomaly_end = 2550
test_df.loc[anomaly_start:anomaly_end, COLUMN] = test_df.loc[anomaly_start:anomaly_end, COLUMN] * 2

# Test df to TimeSeries
test_ts = TimeSeries.from_dataframe(test_df, 'timestamp')

# Load the anomaly model
print("Loading the anomaly model...")
model_package = pickle.load(open(MODEL_PATH, 'rb'))

# Extract the scaler and the anomaly model
scaler = model_package['scaler']
anomaly_model = model_package['anomaly_model']

# Scale the test data
test_ts_scaled = scaler.transform(test_ts)

print("Scoring the anomaly model...")
anomaly_scores, predictions = anomaly_model.score(series=test_ts_scaled, return_model_prediction=True)
print("Done!")

# Plot the predictions
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 6))
test_ts_scaled[COLUMN].plot(ax=ax1, label='Actual')
predictions[COLUMN].plot(ax=ax1, label='Predicted')
# Highlight the anomalies
ax1.axvspan(test_ts.time_index[anomaly_start], test_ts.time_index[anomaly_end], color='red', alpha=0.3)
ax1.legend()

USED_SCORER = 2

# Plot the anomaly scores
# anomaly_scores[0][:, -12].plot(ax=ax1, label='Norm Scorer')
anomaly_scores[COLUMN].plot(ax=ax2, label='Scorer')
# ax1.axvspan(test_ts.time_index[anomaly_start], test_ts.time_index[anomaly_end], color='red', alpha=0.3)
ax2.axvspan(test_ts.time_index[anomaly_start], test_ts.time_index[anomaly_end], color='red', alpha=0.3)
ax2.legend()

# Set the x axis to be equal for all plots
ax2.set_xlim(ax1.get_xlim())
ax3.set_xlim(ax1.get_xlim())

# Binary anomaly detection plot using the quantile detector
detector = QuantileDetector(low_quantile=0.01, high_quantile=0.99)
detector2 = IQRDetector(1.5)
# detector = ThresholdDetector(low_threshold=0.4, high_threshold=0.6)
# (detector.fit_detect(anomaly_scores[0])-0).plot(ax=ax, label='Norm Scorer - detected anomalies')
detected_anomalies = detector.fit_detect(anomaly_scores[COLUMN])
detected_anomalies2 = detector2.fit_detect(anomaly_scores[COLUMN])
detected_anomalies = detected_anomalies * detected_anomalies2
detected_anomalies.plot(ax=ax3, label='Scorer - detected anomalies')
# (detector.detect(anomaly_scores[USED_SCORER])[COLUMN]).plot(ax=ax3, label='Scorer - detected anomalies')
ax3.axvspan(test_ts.time_index[anomaly_start], test_ts.time_index[anomaly_end], color='red', alpha=0.3)
ax3.legend()
plt.show()