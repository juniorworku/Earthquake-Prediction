# Earthquake Magnitude Prediction

![Earthquake Magnitude Prediction](https://via.placeholder.com/800x200.png?text=Earthquake+Magnitude+Prediction)

## Overview
The **Earthquake Magnitude Prediction** project aims to leverage machine learning techniques to accurately predict the magnitudes of earthquakes based on geophysical parameters. This predictive model utilizes a Histogram Gradient Boosting Regressor to analyze historical earthquake data, providing valuable insights into seismic activity. The project features automated pipelines for daily feature extraction and batch inference, ensuring the model remains updated with the latest data.

## Table of Contents
- [Project Objectives](#project-objectives)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How to Run the Code](#how-to-run-the-code)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Inference Pipelines](#inference-pipelines)
- [User Interface](#user-interface)
- [Batch Inference Monitoring](#batch-inference-monitoring)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Project Objectives
- To develop a robust machine learning model capable of predicting earthquake magnitudes from geophysical data.
- To automate the feature extraction and inference processes, ensuring continuous model improvement.
- To provide an interactive user interface for real-time predictions and analysis of seismic events.

## Features
- **Automated Pipelines:** Daily extraction of features and batch inference using GitHub Actions.
- **Interactive Gradio App:** Users can input parameters and receive magnitude predictions.
- **Batch Monitoring UI:** Visualize recent predictions and assess model performance over time.
- **Error Analysis:** Compare recent prediction errors with historical performance metrics to detect shifts in data patterns.
- **Model Retraining:** Seamless retraining of the model with new data inputs to improve accuracy.

## Technologies Used
- **Programming Language:** Python
- **Machine Learning Framework:** Scikit-learn
- **Web Framework:** Gradio
- **Version Control:** GitKraken
- **Automation:** GitHub Actions
- **Data Source:** US Geological Survey (USGS) Earthquake Catalog
- **Environment Management:** Pip

## Installation
To set up this project on your local machine, follow the steps below:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/juniorworku/earthquake-magnitude-prediction.git
   cd earthquake-magnitude-prediction
   ```

2. **Create a virtual environment:**
   ```sh
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   ```sh
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## How to Run the Code

### Daily Feature Pipeline
The daily feature extraction pipeline can be triggered manually or scheduled via GitHub Actions. To run it manually:

1. **Navigate to the project directory:**
   ```sh
   cd earthquake-magnitude-prediction
   ```

2. **Running the daily-feature-pipeline:**
   ```sh
   ./run-daily-feature-pipeline.sh
   ```
   This script will fetch the latest earthquake data from the USGS Earthquake Catalog, extract relevant features, and store them in the hopsworks machine learning platform.

### Batch Inference Pipeline
The batch inference pipeline continuously monitors the hopsworks machine learning platform for new data inputs and triggers the model to make predictions. To run it manually:

1. **Navigate to the project directory:**
   ```sh
   cd earthquake-magnitude-prediction
   ```

2. **Running the batch-inference-pipeline:**
   ```sh
   ./run-batch-inference-pipeline.sh
   ```
   This script will continuously monitor the hopsworks machine learning platform for new data inputs, extract relevant features, and make predictions using the trained model.

   Note: The batch inference pipeline assumes that the trained model file (`model.pkl`) is located in the `models` directory.

### User Interface
The Gradio user interface allows users to input parameters and receive magnitude predictions. To start the Gradio app:

1. **Navigate to the project directory:**
   ```sh
   cd earthquake-magnitude-prediction/earthquakes-magnitude-prediction
   ```

2. **Running the Gradio app:**
   ```sh
   gradio run app.py --port 7860
   ```
   This script will start the Gradio app on port 7860, allowing users to input parameters and receive magnitude predictions.

   Note: The Gradio app assumes that the trained model file (`model.pkl`) is located in the `models` directory.

## Dataset
The project utilizes historical earthquake data from the USGS Earthquake Catalog to train the machine learning model. The dataset consists of earthquake magnitudes, locations, depths, and timestamps.

1. **Data Source:** https://earthquake.usgs.gov/fdsnws/event/1/query
2. **Data Format:** CSV
3. **Data Fields:** magnitude, latitude, longitude, depth, time

## Feature Extraction
The following features are extracted from the historical earthquake data:

1. **id:** Unique identifier for each event.
2. **time:** Timestamp for sorting.
3. **latitude**, **longitude:** Geographic coordinates of the earthquake.
4. **depth**, **depthError:** Depth parameters where the earthquake initiates.
5. **rms:** Root Mean Square (RMS) amplitude of the seismic signal.
6. **reviwed:** Flag indicating whether the earthquake has been reviewed by a human.
7. **mag:** Target feature representing earthquake magnitude.

## Model Training

### Model Selection
The machine learning model used in this project is a Histogram Gradient Boosting Regressor (HGBR). The model is trained on historical earthquake data using the extracted features and earthquake magnitudes. The model is evaluated using cross-validation to assess its performance.

### Training Strategy
- **Train-Test Split:** 80% training data and 20% testing data.
- **Hyperparameter Tuning:** Conducted a random search with 5-fold cross-validation to optimize model parameters.

#### Best Hyperparameters:
- **learning_rate:** 0.001
- **L2 Regularization:** 0.00001
- **Max Iterations:** 200
- **Max Leaf Nodes:** 51
- **Minimum Samples per Leaf:** 15

#### Performance Evaluation
The model's performance is evaluated using the Mean Squared Error (MSE) metric, with an optimal value achieved on the test set.

## Inference Pipelines

### Gradio UI/Online Inference
The Gradio user interface allows users to input parameters and receive magnitude predictions. The inference pipeline continuously monitors the hopsworks machine learning platform for new data inputs and triggers the model to make predictions. Users will provide:

- Latitude
- Longitude
- Depth
- Depth Error
- RMS
- Reviewed

**Output:** The application returns the predicted magnitude along with a visual representation on a world map, indicating the earthquake's location.

### Batch Inference Monitoring
The batch inference pipeline continuously monitors the hopsworks machine learning platform for new data inputs and triggers the model to make predictions. The predictions are stored in a separate hopsworks table and visualized on a dashboard. The dashboard provides a visual representation of recent predictions and allows users to assess model

- The last five predictions made by the model.
- Comparison of MSE for recent predictions against the training dataset metrics, facilitating the detection of covariate shifts.

## Performance Metrics
The model's performance metrics are evaluated using the Mean Squared Error (MSE) metric. The MSE represents the average squared difference between the predicted and actual earthquake magnitudes. A lower MSE indicates better model performance.

## Contributing
To contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request detailing your changes.


## License
This project is licensed under the MIT License. See the [LICENSE] file for details.

## Acknowledgments
- US Geological Survey (USGS) for providing the historical earthquake data.
- Gradio for enabling user interaction with the model.
- GitHub Actions for automating our workflows.


## Contact
For any questions or concerns, please contact the project maintainers at juniorworku[at]gmail.com.
[LICENSE]: https://github.com/juniorworku/earthquake-magnitude-prediction/blob/main/LICENSE







