#!/usr/bin/env python3

import rosbag2_py
import os
from rclpy.serialization import deserialize_message
import pandas as pd
from agrorob_msgs.msg import RemoteState, RobotState, ToolState, Logs, EngineState, FailureState
import matplotlib.pyplot as plt
import argparse

W  = '\033[0m'  # white (normal)
O  = '\033[33m' # orange
B  = '\033[34m' # blue

def flatten_message(msg):
    """Convert a ROS message object into a flat dictionary."""
    return {field: getattr(msg, field) for field in msg.__slots__}


def process_bag(bag_path, topics, save_path, length_threshold):
    bag_name = os.path.basename(bag_path)
    save_name = os.path.join(save_path, bag_name.replace(".db3", ".csv"))
    if not save_name.endswith(".csv"):
        save_name += ".csv"

    # Open the ROS 2 bag
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions()
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        raise ValueError(f"Failed to open bag: {e}")

    # Initialize a dictionary to hold all dataframes
    data_dict = {topic: [] for topic in topics.keys()}

    # Check if the bag is long enough
    bag_duration = reader.get_metadata().duration
    bag_duration = bag_duration.nanoseconds / 1e9
    
    if bag_duration < length_threshold:
        print(O+f"Bag {bag_name} is too short ({bag_duration:.2f}s), skipping"+W)
        return
    else:
        print(B+f"Bag {bag_name} is {bag_duration:.2f}s long"+W)

    # Read the bag
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic in topics:
            msg = deserialize_message(data, topics[topic])
            # print(msg)
            # print('-'*50)
            flat_msg = flatten_message(msg)
            # print(flat_msg)
            flat_msg["timestamp"] = t
            data_dict[topic].append(flat_msg)

    # Combine data from all topics
    combined_data = []
    for topic, rows in data_dict.items():
        if rows:  # If there's data for this topic
            for row in rows:
                # remove '_' from keys
                for key in list(row.keys()):
                    if key.startswith("_"):
                        row[key[1:]] = row.pop(key)
            df = pd.DataFrame(rows)
            df.set_index("timestamp", inplace=True)
            # prefix = f"{topic.replace('/', '_')}_"
            # if prefix.startswith("_"):
                # prefix = prefix[1:]
            df = df.add_prefix(topic+".")
            # print(df.head())
            combined_data.append(df)

    if not combined_data:
        print(f"No data found in {bag_name}")
    # Merge all dataframes on the timestamp
    merged_df = pd.concat(combined_data, axis=1, join="outer")

    # Fill missing values with average of previous and next values
    merged_df.interpolate(method="nearest", inplace=True)

    # Sort by timestamp
    # merged_df.sort_values("timestamp", inplace=True)
    merged_df = merged_df.sort_index()
    print(merged_df.columns)

    # Fill missing boolean values with nearest value
    merged_df.fillna(method="ffill", inplace=True)
    merged_df.fillna(method="bfill", inplace=True)

    # Change boolean columns to integers
    bool_cols = merged_df.select_dtypes(include=["bool"]).columns
    merged_df[bool_cols] = merged_df[bool_cols].astype(int)

    # Modify to have data with frequency of 10Hz
    merged_df.index = pd.to_datetime(merged_df.index)
    merged_df = merged_df.resample("100ms").ffill()

    
    merged_df.to_csv(save_name, index=True)
    print(f"Saved {save_name}")

    # Visualize the data with titles for each subplot, but 
    merged_df.plot(subplots=True, figsize=(20, 60), legend=False)
    

    # titles for each subplot
    for i, ax in enumerate(plt.gcf().axes):
        ax.set_title(merged_df.columns[i])

    # more space between subplots
    plt.tight_layout()

    plt.subplots_adjust(top=0.985, hspace=0.5)

    # Add title
    plt.suptitle(bag_name, fontsize=24, y=0.999)
    plt.savefig(save_name.replace(".csv", ".png"))


if __name__ == "__main__":

    LENGTH_THRESHOLD = 60 # length of the bag in seconds

    # Get the path to the bags and where to save the CSV files as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("bags_path", type=str, help="Path to the folder containing the bags")
    parser.add_argument("save_path", type=str, help="Path to save the CSV files")
    args = parser.parse_args()

    bags_path = args.bags_path
    save_path = args.save_path

    # Ensure save_path exists
    os.makedirs(save_path, exist_ok=True)

    bags = os.listdir(bags_path)
    bags = [os.path.join(bags_path, bag) for bag in bags]

    topics = {
        "/agrorob/remote_state": RemoteState,
        "/agrorob/robot_state": RobotState,
        "/agrorob/tool_state": ToolState,
        "/agrorob/logs": Logs,
        "/agrorob/engine_state": EngineState,
        "/agrorob/failure_state": FailureState,
    }

    print(f"Found {len(bags)} bags")
    remove_bags = []
    for i, bag_path in enumerate(bags):
        print(f"Processing bag {i + 1}/{len(bags)}: {bag_path}")
        try:
            process_bag(bag_path, topics, save_path, LENGTH_THRESHOLD)
        except Exception as e:
            print(f"Failed to process bag: {e}")
            continue
    