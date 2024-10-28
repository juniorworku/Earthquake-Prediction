import os
import pandas as pd
import hopsworks
import joblib
import datetime
from datetime import datetime
import dataframe_image as dfi
import numpy as np
import time

def inference_earthquakes():
    print("Logging into Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print("Retrieving the best model...")
    model = mr.get_best_model("earthquakes_model", "mse", "min")
    model_mse = model.training_metrics['mse']
    model_dir = model.download()
    model = joblib.load(model_dir + "/earthquakes_model.pkl")

    # Set up feature view
    feature_view = fs.get_feature_view(name="earthquakes", version=1)
    batch_data = feature_view.get_batch_data(read_options={"use_hive": True})

    print("Generating predictions...")
    y_pred = model.predict(batch_data.drop(columns=['id', 'time']))

    offset = 100
    pred_magnitudes = y_pred[y_pred.size - offset:]
    dataset_api = project.get_dataset_api()

    print("Reading feature group for labels...")
    earthquakes_fg = fs.get_feature_group(name='earthquakes', version=1)
    df = earthquakes_fg.read()
    labels = df.iloc[-offset:]['mag']

    print("Preparing data for monitoring feature group insertion...")
    monitor_fg = fs.get_or_create_feature_group(
        name='earthquakes_predictions',
        version=1,
        primary_key=["id"],
        description="Earthquake Magnitude Prediction/Outcome Monitoring"
    )

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    data = {
        'prediction': pred_magnitudes,
        'label': labels.values,
        'datetime': [now] * offset,
        'id': df.iloc[-offset:]['id'].values
    }

    monitor_df = pd.DataFrame(data)
    print("Inserting predictions into monitoring feature group...")
    monitor_fg.insert(monitor_df)

    print("Generating history dataframe...")
    history_df = monitor_fg.read(read_options={"use_hive": True})

    # Combine recent predictions with historical data for analysis
    history_df = pd.concat([history_df, monitor_df])

    # Sort by datetime
    history_df['date_created'] = pd.to_datetime(history_df['datetime'], dayfirst=True)
    history_df = history_df.sort_values(by=['date_created'], ascending=False)

    print("Saving recent predictions image...")
    df_recent = history_df.head(5)
    dfi.export(df_recent.drop(columns=['date_created', 'id']).style.hide(axis='index'), 
                              './df_recent.png', table_conversion='matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)

    print("Saving MSE comparison image...")
    df_mse = pd.DataFrame({'MSE': [np.square(pred_magnitudes - labels.values).mean(), model_mse]},
                          index=['Recent inference', 'Model training'])

    dfi.export(df_mse, './df_mse.png', table_conversion='matplotlib')
    dataset_api.upload("./df_mse.png", "Resources/images", overwrite=True)

if __name__ == "__main__":
    inference_earthquakes()
