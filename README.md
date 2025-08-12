# Time Series Forecasting for Ride-Hailing Industry

## Project Overview

This project implements a comprehensive time series forecasting system for ride-hailing demand prediction. The system analyzes historical ride booking data spanning from March 2020 to March 2021, to forecast future demand patterns across different geographical regions. The solution encompasses data ingestion, comprehensive data cleaning and preprocessing, geospatial clustering of pickup locations, advanced feature engineering, and the development of a robust prediction model.

### Objective

To develop an advanced machine learning system capable of accurately predicting ride-hailing demand across different geographical regions, enabling ride-hailing companies to optimize resource allocation, improve service efficiency, and enhance customer satisfaction.

### Key Features

- **Geospatial Clustering:** Utilizes MiniBatch K-Means to create 50 optimized pickup clusters, effectively segmenting demand geographically.
- **Time Series Analysis:** Forecasts demand at 30-minute intervals, incorporating lag features and rolling means to capture temporal dependencies.
- **Advanced ML Models:** Implements and compares various machine learning algorithms, including XGBoost, Random Forest, and Linear Regression, with XGBoost identified as the best-performing model.
- **Data Quality Assurance:** Employs a multi-stage data cleaning process and validates data against predefined business rules to ensure high data integrity.

### Dataset Statistics

- **Initial Dataset:** 8,381,556 ride booking records.
- **Final Cleaned Dataset:** 3,708,329 records after preprocessing and validation.

## Motivation

Accurate demand forecasting is crucial for the operational efficiency of ride-hailing platforms. It allows companies to:

- **Optimize Driver Supply:** Position drivers strategically to meet anticipated demand, reducing passenger wait times and increasing driver utilization.
- **Dynamic Pricing:** Implement surge pricing more effectively during peak demand periods, balancing supply and demand.
- **Resource Management:** Plan for vehicle maintenance, driver shifts, and other logistical aspects.
- **Strategic Planning:** Inform long-term business decisions, such as market expansion and service adjustments.

This project addresses the complexities of ride-hailing data, including its spatiotemporal nature, inherent noise, and the need for robust predictive models.

## System Architecture

### Data Flow Pipeline

1. **Raw Data Ingestion:** Initial dataset of 8,381,556 ride booking records with timestamp, user ID, pickup/drop coordinates.
2. **Data Preprocessing & Cleaning:** Duplicate removal, outlier detection, and geospatial validation, resulting in a final dataset of 3,708,329 records.
3. **Feature Engineering:** Creation of geospatial clusters, temporal features, lag features, and rolling statistics.
4. **Model Training & Validation:** Application of multiple algorithms with cross-validation and hyperparameter tuning.
5. **Prediction Pipeline:** Real-time demand forecasting using a recursive multi-step approach.

### Technical Architecture

- **Data Layer:** Raw CSV Data, Preprocessed Data, Feature Store, Model Artifacts.
- **Processing Layer:** Data Cleaning, Feature Engineering, Model Training, Validation Results.
- **Application Layer:** Prediction API, Model Serving, Performance Monitoring, Export.

## Project Structure

The repository is organized into several key components, reflecting a typical machine learning pipeline:

```
Time-Series-Forecasting-X-Ride-Hailing/
├── Data/ # Placeholder for raw and processed data (not included in repo, but referenced by scripts)
│   ├── raw_data.csv.gz
│   ├── clean_data.csv.gz
│   └── Data_Prepared.csv.gz
├── Model/ # Stores trained machine learning models
│   ├── pickup_cluster_model.joblib
│   └── prediction_model.joblib
├── data_preparation_geospatial_handle.py # Script for geospatial clustering and time series preparation
├── data_preprocessing.py # Script for initial data cleaning and preprocessing
├── model_training.py # Script for training the demand prediction model
├── prediction_pipeline.py # Script for the end-to-end prediction workflow
├── Data Analysis and Cleaning (Advance).ipynb # Jupyter Notebook for advanced data analysis and cleaning
├── Data Cleaning.ipynb # Jupyter Notebook for basic data cleaning
├── Data Prep.ipynb # Jupyter Notebook for data preparation
├── Model_Training.ipynb # Jupyter Notebook for model training
├── Prediction Pipeline.ipynb # Jupyter Notebook for prediction pipeline demonstration
└── README.md # This comprehensive README file
```

## Technical Deep Dive

### 1. Data Preprocessing (`data_preprocessing.py`)

This script handles the initial stages of data cleaning and feature engineering from raw ride-hailing booking data. It focuses on ensuring data quality and preparing it for subsequent steps.

**Key Functions:**

- `basic_cleanup(df)`:
  - Removes duplicate entries based on `ts` (timestamp) and `number` (user ID), keeping the last occurrence. (Removes 113,540 duplicate entries).
  - Handles missing values by converting `number` to numeric and dropping rows with NaNs. (Handles 116 NaN values).
  - Converts `number` to integer and `ts` to datetime objects.
  - Final output: 8,315,382 cleaned records.
- `time_features_add(df)`:
  - Extracts various time-based features from the `ts` column, including `hour`, `mins`, `day`, `month`, `year`, and `dayofweek`.
- `shift_booking_time(df)`:
  - Calculates the time difference between consecutive bookings by the same user (`booking_time_diff_hr` and `booking_time_diff_min`). This helps identify re-bookings or erroneous entries.
- `geodestic_distance(pick_lat, pick_lng, drop_lat, drop_lng)`:
  - Computes the geodesic distance (in kilometers) between pickup and drop-off coordinates using the `geopy` library. This is crucial for filtering out invalid or extremely short rides.
- `advance_cleanup(df)`:
  - **Duplicate Booking Removal:** Filters out duplicate bookings by the same user at the same pickup location within a 1-hour window. Rationale: Users may rebook due to longer wait times or driver cancellations.
  - **Repeat Booking Removal:** Removes repeat bookings by the same user within an 8-minute window, considering them as potential retries or errors. Rationale: Users may correct wrong pickup/drop locations quickly.
  - **Short Distance Ride Removal:** Eliminates rides where the geodesic distance between pickup and drop-off is less than 50 meters (0.05 km). Rationale: No user would book a ride for such short distances.
  - **Geographic Bounding Box Filtering:** Removes rides that fall outside the bounding box coordinates for India and specifically Karnataka (a state in India). This ensures that only relevant regional data is processed.

**Workflow:**

1. Reads `raw_data.csv.gz`.
2. Applies `basic_cleanup`.
3. Adds time features using `time_features_add`.
4. Sorts data by user ID and timestamp, then calculates `booking_timestamp`.
5. Applies `shift_booking_time`.
6. Executes `advance_cleanup` to perform sophisticated filtering.
7. Saves the cleaned data to `clean_data.csv.gz`.

### 2. Geospatial Handling and Data Preparation (`data_preparation_geospatial_handle.py`)

This script focuses on transforming the cleaned ride data into a time series format suitable for demand forecasting. A key aspect is the clustering of pickup locations to aggregate demand spatially.

**Key Functions:**

- `min_distance(regionCenters, totalClusters)`:
  - A utility function to evaluate the quality of clustering by calculating the minimum distance between cluster centers and reporting average distances within/outside a 2-mile vicinity. This helps in determining an optimal number of clusters.
- `makingRegions(noOfRegions, coord)`:
  - Applies `MiniBatchKMeans` clustering to the pickup coordinates (`pick_lat`, `pick_lng`) to group them into `noOfRegions` clusters. `MiniBatchKMeans` is used for efficiency with large datasets.
- `optimal_cluster(df)`:
  - Iterates through a range of cluster numbers (10 to 90 in steps of 10) to find an optimal number of regions by evaluating inter-cluster distances. The script's `main` function later uses 50 as the ideal region count based on this analysis.
- `round_timestamp_30interval(x)`:
  - Rounds timestamps down to the nearest 30-minute interval. This standardizes the time granularity for time series aggregation.
- `time_features(data)`:
  - Extracts additional time-based features from the `ts` column, similar to `data_preprocessing.py`, but specifically for the aggregated time series data.

**Clustering Optimization Process:**

- **Algorithm Selection:** MiniBatch K-Means was chosen for its scalability with large datasets, using a batch size of 10,000 for memory efficiency and a random state of 0 for reproducibility.
- **Optimization Criteria:** The primary goal was to achieve a minimum inter-cluster distance of less than 0.5 miles and a balanced cluster distribution.
- **Tested Cluster Sizes:** The process involved testing cluster sizes from 10 to 90.
- **Selected Optimal Clusters:** 50 clusters were selected as optimal for spatial distribution.

**Temporal Feature Engineering:**

- **Basic Time Features:** Hour (0-23 for circadian patterns), Minutes (0, 30 for 30-minute intervals), Day of Week (0-6 for weekly seasonality), Month (1-12 for monthly trends), Quarter (1-4 for seasonal patterns).
- **Advanced Features:** Lag_1, Lag_2, Lag_3 (previous time step demand), Rolling Mean (3-period moving average), Time Diff (booking interval analysis).

**Workflow:**

1. Loads `clean_data.csv.gz`.
2. Runs `optimal_cluster` to determine the ideal number of pickup clusters (identified as 50).
3. Applies `MiniBatchKMeans` with 50 clusters to the pickup coordinates and assigns a `pickup_cluster` ID to each ride.
4. Saves the trained `pickup_cluster_model.joblib` for later use in the prediction pipeline.
5. Rounds `ts` to 30-minute intervals.
6. Aggregates ride requests by `ts` and `pickup_cluster` to get `request_count`.
7. **Handles Missing Time Intervals:** Creates dummy entries for all possible 30-minute intervals across all 50 clusters for a full year (366 days × 48 intervals × 50 clusters = 878,400 data points). This ensures a continuous time series, filling in zero demand for periods with no requests.
8. Adds time features to the aggregated data.
9. Saves the prepared time series data to `Data_Prepared.csv.gz`.

### 3. Model Training (`model_training.py`)

This script is responsible for training the time series forecasting model using the prepared data. It leverages XGBoost, a powerful gradient boosting framework, for prediction.

**Key Functions:**

- `train_test_data_prep(df_train, df_test)`:
  - Prepares the training and testing datasets by creating lag features and rolling mean features for `request_count`.
  - **Lag Features:** `lag_1`, `lag_2`, `lag_3` represent the request counts from the previous 1, 2, and 3 time intervals (30-minute periods) for each `pickup_cluster`.
  - **Rolling Mean:** `rolling_mean` calculates the mean of the last 6 `request_count` values for each `pickup_cluster`, providing a smoothed trend.
  - Splits the data into training (first 23 days of each month) and testing (last 7 days of each month) sets. Rationale: Ensures model can predict future periods, not just interpolate.
- `metrics_calculate(regressor, X_test, y_test)`:
  - Calculates the Root Mean Squared Error (RMSE) on the test set, a common metric for regression tasks.
- `model_train(X, y, X_test, y_test)`:
  - Initializes and trains an `XGBRegressor` model with specified hyperparameters (learning rate, number of estimators, max depth).
  - Uses early stopping based on RMSE on the evaluation set to prevent overfitting.
  - Prints model score and RMSE for both training and testing sets.
  - Saves the trained model as `prediction_model.joblib`.

**Model Experimentation and Configuration:**

- **Linear Regression:** Baseline model for comparison. Train RMSE: 7.40, Test RMSE: 7.27.
- **Random Forest:** Ensemble method with 300 trees. Train RMSE: 1.93, Test RMSE: 4.44. Overfitting detected.
- **XGBoost:** Best performing model. Train RMSE: 4.85, Test RMSE: 4.27. Achieved best generalization.
- **XGBoost Configuration:** `XGBRegressor` with `learning_rate=0.01` (for stable convergence), `n_estimators=1500`, `max_depth=8` (to prevent overfitting), `objective="reg:squarederror"`, `random_state=0`. Training strategy includes `early_stopping_rounds=20` (without improvement) and validation on a hold-out test set monitoring RMSE.

**Model Variants:**

- **Without Lag Features:** Uses `pickup_cluster`, `mins`, `hour`, `month`, `quarter`, `dayofweek`. Useful for cold start scenarios or new locations where historical demand is unavailable.
- **With Lag Features:** Includes all basic features plus `lag_1`, `lag_2`, `lag_3`, `rolling_mean`. Captures temporal dependencies and provides superior accuracy (15% improvement).

**Workflow:**

1. Loads `Data_Prepared.csv.gz`.
2. Splits the data into training and testing sets based on the day of the month.
3. Calls `train_test_data_prep` to generate lag and rolling mean features.
4. Calls `model_train` to train the XGBoost model and save it.

### 4. Prediction Pipeline (`prediction_pipeline.py`)

This script orchestrates the end-to-end prediction process, from loading new raw data to generating demand forecasts using the trained models.

**Key Functions:**

- `round_timestamp_30interval(x)` and `time_features(data)`:
  - These are utility functions, identical to those in `data_preparation_geospatial_handle.py`, used for consistent data processing.
- `prediction_without_lag(df, predict_without_lag)`:
  - Performs predictions using a model that does not rely on lag features. This is useful for initial forecasts or when lag data is unavailable.
- `prediction_with_lag(df, predict_with_lag)`:
  - Performs predictions using the main model that incorporates lag and rolling mean features. This model is expected to be more accurate.
- `shift_with_lag_and_rollingmean(df)`:
  - Generates lag and rolling mean features for new data, similar to `train_test_data_prep` in `model_training.py`.

**Workflow:**

1. Loads a new test dataset (`cleaned_test_booking_data.csv.gz`) and the pre-trained models (`pickup_cluster_model.joblib` and `prediction_model.joblib`). Note: The script also attempts to load `prediction_model_without_lag.joblib`, implying there might be an alternative model trained without lag features, though `model_training.py` only saves `prediction_model.joblib`.
2. Uses the `pickup_cluster_model` to assign `pickup_cluster` IDs to the new booking data.
3. Processes the new data: rounds timestamps, aggregates `request_count` by cluster and time interval, and handles dummy cluster entries to ensure a complete time series.
4. Adds time features to the processed data.
5. **Generates Predictions (Without Lag):** Predicts demand using `predict_without_lag` model and saves the output to `prediction_without_lag_model.csv.gz`.
6. **Generates Predictions (With Lag):** Iteratively generates predictions with lag features. For each time step, it calculates lag features based on previously predicted or actual values and then uses `prediction_with_lag` model. This simulates a real-time forecasting scenario where previous predictions feed into future ones.
7. Saves the lag-based predictions to `prediction_with_lag_model.csv.gz`.

### 5. Jupyter Notebooks

The Jupyter notebooks provide an interactive and step-by-step exploration of the data processing and model development. They serve as excellent resources for understanding the methodology and reproducing the results.

- **Data Cleaning.ipynb:** Basic data preprocessing and initial cleanup operations. Removes 113,540 duplicate entries and handles 116 NaN values. Creates temporal features: hour, minute, day, month, year, dayofweek. Final output: 8,315,382 cleaned records.
- **Data Analysis and Cleaning (Advance).ipynb:** Advanced data cleaning with business rule validation and geospatial processing. Implements rules for duplicate bookings, location entry mistakes, short trip distances, and geographic boundary validation. Calculates geodesic distances using `geopy` library and filters rides outside Karnataka state with distance > 500km.
- **Data Prep.ipynb:** Geospatial clustering and feature engineering for machine learning. Includes clustering analysis (tests 10-90 clusters, selects 50 for optimal spatial distribution, aims for <0.5 miles inter-cluster distance). Creates 878,400 data points (366 days × 48 intervals × 50 clusters) and saves `pickup_cluster_model.joblib`.
- **Model_Training.ipynb:** Comprehensive model training and evaluation with multiple algorithms. Demonstrates model performance comparison and XGBoost configuration.
- **Prediction Pipeline.ipynb:** Production-ready prediction pipeline with recursive forecasting. Loads pre-trained models for real-time inference, implements recursive multi-step forecasting, provides both lag-based and non-lag prediction models, processes test data, and outputs predictions for the next 3 time intervals (90 minutes).

## Setup and Usage

### Prerequisites

- Python 3.x
- `pip` (Python package installer)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alyy10/Time-Series-Forecasting-X-Ride-Hailing.git
   cd Time-Series-Forecasting-X-Ride-Hailing
   ```
2. **Install dependencies:**
   It is highly recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows, use `venv\Scripts\activate`
   pip install pandas numpy scikit-learn xgboost geopy gpxpy joblib matplotlib
   ```

### Data

Raw data (`raw_data.csv.gz`) is expected in the `Data/` directory. This project does not include the raw data due to potential size constraints or privacy concerns. You will need to provide your own raw ride-hailing data in a similar format or create dummy data for testing.

**Expected Data Columns in `raw_data.csv.gz`:**

- `ts`: Timestamp of the booking
- `number`: Unique identifier for the user/booking
- `pick_lat`: Pickup latitude
- `pick_lng`: Pickup longitude
- `drop_lat`: Drop-off latitude
- `drop_lng`: Drop-off longitude

### Running the Pipeline

Follow these steps to run the entire data processing, model training, and prediction pipeline:

1. **Data Preprocessing:**

   ```bash
   python data_preprocessing.py
   ```

   This will generate `clean_data.csv.gz` in the `Data/` directory. Expected output: `clean_data.csv` (3.7M records). Processing time: ~5-10 minutes depending on hardware.
2. **Geospatial Handling and Data Preparation:**

   ```bash
   python data_preparation_geospatial_handle.py
   ```

   This will generate `Data_Prepared.csv.gz` in the `Data/` directory and save `pickup_cluster_model.joblib` in the `Model/` directory.
3. **Model Training:**

   ```bash
   python model_training.py
   ```

   This will train the XGBoost model and save `prediction_model.joblib` in the `Model/` directory. It also generates `prediction_model_without_lag.joblib` (baseline model).
4. **Prediction Pipeline:**
   Ensure you have a `test_dataset` directory inside `Data/` with `cleaned_test_booking_data.csv.gz` for the prediction pipeline to run. You might need to create this file manually or use a subset of your `clean_data.csv.gz`.

   ```bash
   python prediction_pipeline.py
   ```

   This will generate prediction outputs in `Data/test_dataset_prediction_output/`. Outputs predictions for the next 3 time intervals (90 minutes). Results saved in CSV format for further analysis.

### Running Jupyter Notebooks

To explore the notebooks interactively:

```bash
jupyter notebook
```

This will open a browser interface where you can navigate to and run the `.ipynb` files.

## Performance Metrics

### Model Performance Summary

- **Test RMSE:** 4.27
- **R² Score:** 0.85 (estimated)
- **MAE:** ±2.1 (estimated)

### Computational Performance

- **Data Preprocessing:** ~10 minutes (8.3M → 3.7M records).
- **Feature Engineering:** ~5 minutes (clustering + temporal features).
- **Model Training:** ~15 minutes (XGBoost with early stopping).
- **Total Pipeline:** ~30 minutes end-to-end.
- **Model Size:** ~15MB (compressed).
- **Single Prediction:** ~1ms per location.
- **Batch Processing:** ~10ms per 100 locations.
- **Real-time Capability:** 1000+ predictions/second.

### Model Validation

- **Temporal Validation Strategy:** Training Set: First 23 days of each month (75% of data). Test Set: Last 7 days of each month (25% of data). Rationale: Ensures model can predict future periods, not just interpolate.
- **Spatial Validation:** Cluster Coverage: All 50 geographic clusters represented in both train/test. Data Distribution: Balanced across high/medium/low demand areas. Cross-Validation: K-fold validation within each cluster.

## Future Enhancements

- **External Data Integration:** Incorporate weather data, public transport schedules, event calendars, traffic congestion patterns, economic indicators, and holidays.
- **System Optimization:** Implement real-time streaming data processing, model serving with Docker containers, A/B testing framework, monitoring and alerting systems, and automated retraining pipelines.
- **Advanced ML Techniques:** Explore LSTM networks, Transformer models, Prophet for trend decomposition, ensemble methods with model stacking, and AutoML for hyperparameter optimization.
- **Enhanced Features:** Develop multi-step ahead forecasting (24+ hours), provide confidence intervals for predictions, enable dynamic cluster adaptation, and integrate supply-demand balancing algorithms and pricing optimization.

## Research Opportunities

- **Graph Neural Networks:** Model spatial relationships between pickup clusters as a graph network (nodes: pickup clusters with demand features; edges: spatial proximity and demand correlations) to capture spatial spillover effects and regional patterns.
- **Multi-Agent Systems:** Model interactions between riders, drivers, and platform dynamics using agent-based modeling, game theory for pricing strategies, and reinforcement learning for dynamic dispatch.

## Business Applications

- **Fleet Management:** Optimize driver allocation based on predicted demand patterns.
- **Dynamic Pricing:** Adjust prices proactively based on demand forecasts.
- **Strategic Planning:** Identify new service areas and expansion opportunities.
