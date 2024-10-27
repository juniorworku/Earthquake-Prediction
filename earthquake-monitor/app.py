import gradio as gr
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api

dataset_api.download("Resources/images/df_recent.png")
dataset_api.download("Resources/images/df_mse.png")

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Label("Recent Prediction History")
            input_img = gr.Image("df_recent.png", elem_id="recent-predictions")
        with gr.Column():          
          gr.Label("Recent inference MSE compared to Model Training MSE")
          input_img = gr.Image("df_mse.png", elem_id="mse")        

demo.launch()
