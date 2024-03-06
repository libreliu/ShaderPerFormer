import multiprocessing
import vkDisplay
import threading
import vkExecute
import numpy as np
import gradio as gr
import run
import pandas as pd
# import torch
# import os
# from vkPredict.misc.Directory import getVkPredictRootDir

def test(codeEditor, widthSld, heightSld, timeSld, frameSld):
    data = {
                'column1': ['value1', '1111', 'value3'],
                'column2': ['value4', 'value5', 'value6']
            }

            # convert dictionary to pandas DataFrame
    df = pd.DataFrame(data)
    return df

with gr.Blocks() as mainBlock:
    with gr.Row(elem_id="layout"):
        with gr.Column(scale=3):
            codeEditor = gr.Code(language="typescript", scale=3, elem_id="code_editor", lines=50)

        with gr.Column(scale=2):
            widthSld = gr.Slider(minimum=0, maximum=1920, value=1024, step=1, label="Width", scale=1)
            heightSld = gr.Slider(minimum=0, maximum=1080, value=768, step=1, label="Height", scale=1)
            timeSld = gr.Slider(minimum=0, maximum=10, value=0, step=0.01, label="Time", scale=1)
            frameSld = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label="Frame", scale=1)

            with gr.Row():
                renderBtn = gr.Button("Render&Predict", scale=1)
                testBtn = gr.Button("RenderOneFrame", scale=1)

            renderResult = gr.Image(image_mode="RGBA")
            # your dictionary
            data = {
                'column1': ['value1', 'value2', 'value3'],
                'column2': ['value4', 'value5', 'value6']
            }

            # convert dictionary to pandas DataFrame
            df = pd.DataFrame(data)

            # set the DataFrame as output to gr.Dataframe()
            predictions = gr.Dataframe(df)

            # set the function to be called when the button is clicked
            renderBtn.click(fn=test, inputs=[codeEditor, widthSld, heightSld, timeSld, frameSld], outputs=predictions)

mainBlock.launch()