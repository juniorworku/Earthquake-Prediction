import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd
import hsfs
import numpy as np
import plotly.graph_objects as go

# Initialize Hopsworks project and load model
project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("earthquakes_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/earthquakes_model.pkl")
print("Model downloaded")

def earthquake_info(mag):
    if mag <= 2.5:
        return 'Usually not felt, but can be recorded by seismograph.'
    elif 2.5 < mag < 5.45:
        return 'Often felt, but only causes minor damage.'
    elif 5.45 <= mag < 6.05:
        return 'Slight damage to buildings and other structures.'
    elif 6.05 <= mag < 6.95:
        return 'May cause a lot of damage in very populated areas.'
    elif 6.95 <= mag < 7.95:
        return 'Major earthquake. Serious damage.'
    else:
        return 'Great earthquake. Can totally destroy communities near the epicenter.'

def earthquake(latitude, longitude, depth, depth_error, rms, reviewed):
    print("Calling function")
    df = pd.DataFrame([[latitude, longitude, depth, depth_error, rms, reviewed]], 
                      columns=['latitude', 'longitude', 'depth', 'deptherror', 'rms', 'reviewed'])
    print("Predicting")
    print(df)

    # Predict magnitude
    res = model.predict(df)[0]  # Accessing the first element if the model output is an array
    print("Predicted Magnitude:", res)
    
    # Map display of earthquake location
    fig = go.Figure(go.Scattermapbox(
        lat=[latitude],
        lon=[longitude],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=6
        )
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=latitude,
                lon=longitude
            ),
        ),
    )
    
    # Generate earthquake description based on predicted magnitude
    description = earthquake_info(res)
    return fig, float(res), description  # Ensure magnitude is returned as a float for gr.Number

demo = gr.Interface(
    fn=earthquake,
    title="Earthquake Magnitude Predictor",
    description="Input Known Earthquake Features to Determine the Magnitude",
    allow_flagging="never",
    inputs=[
        gr.Slider(-90, 90, value=0, label="latitude"),
        gr.Slider(-180, 180, value=0, label="longitude"),
        gr.Slider(-10, 700, value=25, label="depth"),
        gr.Slider(0, 70, value=2, label="depth_error"),
        gr.Slider(0, 4, value=0.25, label="rms", info="The root-mean-square travel time residual."),
        gr.Checkbox(label="Reviewed", info="Has the earthquake been reviewed by a human?")
    ],
    outputs=[
        gr.Plot(label="location"),
        gr.Number(label="magnitude"),
        gr.Textbox(label="description")
    ]
)

demo.launch(share=True)
