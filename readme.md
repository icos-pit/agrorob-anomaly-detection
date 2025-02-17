# Agrorob Anomaly Detection

## Using as a docker container with ROS2

Run the anomaly detection as a docker container:

```bash
docker run -it --rm kamilmlody97/agrorob_anomaly_detection:latest
```

Example data can be played using another container:

```bash
docker run -it --rm \
  --entrypoint "/bin/bash" \
  kamilmlody97/agrorob_anomaly_detection:latest \
  -c "source /opt/ros/jazzy/setup.bash && source /ros_ws/install/setup.bash && ros2 bag play -l anomaly_bag"
```

or using the same container (you need to provide the container id):

```bash
docker exec -it <DOCKER_CONTAINER_ID> /bin/bash -c "source /opt/ros/jazzy/setup.bash && source /ros_ws/install/setup.bash && ros2 bag play -l anomaly_bag"
```

## Using with CSV files

### Installation

To install the requirements, run the following command:

```bash
pip install -r requirements.txt
```

### Training

To extract data from the bag files, run the following command:

```bash
python3 bag_to_df.py <path_to_bag_files> <path_to_save_data>
```

To train the model, run the following command:

```bash
python3 anomaly_detection train.py <path_to_data>
```

### Testing

To test the model, run the following command (modify the test.py file with desired test data paths and parameters):

```bash
python3 anomaly_detection test.py
```