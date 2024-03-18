import multiprocessing
import threading
import vkExecute
from vkExecute import RenderWindow
import numpy as np
import gradio as gr
import predictor
import torch
import os
import pandas as pd
from vkPredict.misc.Directory import getVkPredictRootDir

COMMON_VERT_SHADER_SRC = '''#version 310 es
    precision highp float;
    precision highp int;
    precision mediump sampler3D;
    layout(location = 0) in vec3 inPosition;
    void main() {gl_Position = vec4(inPosition, 1.0);}
    '''

FRAG_SHADER_SRC_HEAD = '''#version 310 es
    precision highp float;
    precision highp int;
    precision mediump sampler3D;

    layout(location = 0) out vec4 outColor;

    layout (binding=0) uniform PrimaryUBO {
      uniform vec3 iResolution;
      uniform float iTime;
      uniform vec4 iChannelTime;
      uniform vec4 iMouse;
      uniform vec4 iDate;
      uniform vec3 iChannelResolution[4];
      uniform float iSampleRate;
      uniform int iFrame;
      uniform float iTimeDelta;
      uniform float iFrameRate;
    };

    void mainImage(out vec4 c, in vec2 f);
    void main() {mainImage(outColor, gl_FragCoord.xy);}

    '''

MODEL_TYPE = "perfformer-layer9-regression-trace-input-embedding-xformers-memeff"

class ProcessMessager:
    def __init__(self, runner, inQueue, outQueue=None):
        self.runner = runner
        self.inQueue = inQueue
        self.outQueue = outQueue

    def run(self, *args):
        if self.runner == None:
            return
        if self.outQueue != None:
            return self.runner(self.inQueue, self.outQueue, *args)
        else:
            return self.runner(self.inQueue, *args)


# render one frame with vkExecute
def getOneFrameFn(inQueue, outQueue):
    imageRender = vkExecute.ImageOnlyExecutor()
    imageRender.init(True, False)
    
    while True:
        width = inQueue.get()
        height = inQueue.get()
        iTime = inQueue.get()
        iFrame = inQueue.get()

        fragShdrSrc = FRAG_SHADER_SRC_HEAD + inQueue.get()

        vertShaderSpv, vertErrMsg = vkExecute.ShaderProcessor.compileShaderToSPIRV_Vulkan(
            vkExecute.ShaderProcessor.ShaderStages.VERTEX,
            COMMON_VERT_SHADER_SRC,
            "VertShader"
        )

        fragShaderSpv, fragErrMsg = vkExecute.ShaderProcessor.compileShaderToSPIRV_Vulkan(
            vkExecute.ShaderProcessor.ShaderStages.FRAGMENT,
            fragShdrSrc,
            "FragShader"
        )

        # if the shader is valid, render one frame, else return an empty array
        if vertErrMsg == "" and fragErrMsg == "":
            pCfg = vkExecute.ImageOnlyExecutor.PipelineConfig()
            pCfg.targetWidth = width
            pCfg.targetHeight = height
            pCfg.vertexShader = vertShaderSpv
            pCfg.fragmentShader = fragShaderSpv

            imageUniform = vkExecute.ImageOnlyExecutor.ImageUniformBlock()
            imageUniform.iResolution = [width, height, 0]
            imageUniform.iTime = iTime
            imageUniform.iFrame = iFrame

            imageRender.initPipeline(pCfg)
            imageRender.setUniform(imageUniform)
            imageRender.preRender()
            imageRender.render(1)

            img, size = imageRender.getResults()

            imgData = np.array(img)

            outQueue.put(imgData)
        else:
            outQueue.put([])


def getOneFrame(inQueue, outQueue, shaderSrc, width, height, iTime, iFrame):
    inQueue.put(width)
    inQueue.put(height)
    inQueue.put(iTime)
    inQueue.put(iFrame)
    inQueue.put(shaderSrc)

    imgData = outQueue.get()

    return imgData


def predict(inQueue, outQueue, shaderSrc, width, height, iTime, iFrame):
    inQueue.put(shaderSrc)
    inQueue.put(width)
    inQueue.put(height)
    inQueue.put(iTime)
    inQueue.put(iFrame)

    predictions = outQueue.get()
    dataFrame = pd.DataFrame(list(predictions.items()), columns=["Device", "Time (s)"])
    dataFrame['Time (s)'] = dataFrame['Time (s)'].apply(lambda x: f'{x:.6f}')
    return dataFrame

def predictFn():
    while True:
        shaderSrc = predictInQueue.get()
        width = predictInQueue.get()
        height = predictInQueue.get()
        iTime = predictInQueue.get()
        iFrame = predictInQueue.get()

        dataProcessor.setShader(shaderSrc, width, height, iTime, iFrame)
        dataProcessor.process()
        modelInput = dataProcessor.getModelInput()

        # predict
        predictor.predict(modelInput)
        predictions = predictor.getPreds()

        predictOutQueue.put(predictions)

def setShaderSrc(displayInQueue, shaderSrc, width, height):
    # dynamically change the shader and resolution
    displayInQueue.put(width)
    displayInQueue.put(height)
    displayInQueue.put(shaderSrc)

def setShaderSrcFn():
    # dynamically change the shader and resolution
    while True:
        width = displayInQueue.get()
        height = displayInQueue.get()
        shaderSrc = displayInQueue.get()
        renderWindow.setResolution(width, height)
        renderWindow.setShader(shaderSrc)

def webMain(displayInQueue, frameInQueue, frameOutQueue, predictInQueue, predictOutQueue):
    styleCSS = '''
    #layout {
        height: 90vh; 
    }
    #code_editor {
        height: 90vh;
        box-sizing: border-box;
    }
    .ͼ2p .cm-scroller {
        height: 95%;
    }
    '''
    display = ProcessMessager(setShaderSrc, displayInQueue)
    frame = ProcessMessager(getOneFrame, frameInQueue, frameOutQueue)
    predictor = ProcessMessager(predict, predictInQueue, predictOutQueue)

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
                predictions = gr.Dataframe()

                # set the function to be called when the button is clicked
                renderBtn.click(fn=display.run, inputs=[codeEditor, widthSld, heightSld])
                renderBtn.click(fn=predictor.run, inputs=[codeEditor, widthSld, heightSld, timeSld, frameSld], outputs=predictions)
                testBtn.click(fn=frame.run, inputs=[codeEditor, widthSld, heightSld, timeSld, frameSld],
                              outputs=renderResult)

    mainBlock.launch()

def loadModels():
    modelPaths = {}
    modelPaths["3060"] = os.path.join(getVkPredictRootDir(), "Perfformer-NoRope-Trace-Onehot-Base2-Regression-3060-With-Val-fp16-log-time_HfTracedSpvTokenizer-multiple-entrypoint_perfformer-layer9-regression-trace-input-embedding-xformers-memeff/model.safetensors")
    modelPaths["4060"] = os.path.join(getVkPredictRootDir(), "Perfformer-NoRope-Trace-Onehot-Base2-Regression-4060-With-Val-fp16-log-time_HfTracedSpvTokenizer-multiple-entrypoint_perfformer-layer9-regression-trace-input-embedding-xformers-memeff/model.safetensors")
    modelPaths["1660Ti"] = os.path.join(getVkPredictRootDir(), "Perfformer-NoRope-Trace-Onehot-Base2-Regression-1660-With-Val-fp16-log-time_HfTracedSpvTokenizer-multiple-entrypoint_perfformer-layer9-regression-trace-input-embedding-xformers-memeff/model.safetensors")
    modelPaths["6600XT"] = os.path.join(getVkPredictRootDir(), "Perfformer-NoRope-Trace-Onehot-Base2-Regression-6600XT-With-Val-fp16-log-time_HfTracedSpvTokenizer-multiple-entrypoint_perfformer-layer9-regression-trace-input-embedding-xformers-memeff/model.safetensors")
    modelPaths["UHD630"] = os.path.join(getVkPredictRootDir(), "Perfformer-NoRope-Trace-Onehot-Base2-Regression-UHD630-With-Val-fp16-log-time_HfTracedSpvTokenizer-multiple-entrypoint_perfformer-layer9-regression-trace-input-embedding-xformers-memeff/model.safetensors")
    predictor.setModelPathes(modelPathes=modelPaths)


if __name__ == "__main__":
    renderWindow = RenderWindow()
    dataProcessor = predictor.DataProcessor(fragShaderSrcHead=FRAG_SHADER_SRC_HEAD)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = dataProcessor.tokenizer
    predictor = predictor.Predictor(MODEL_TYPE, device, tokenizer)
    loadModels()

    displayInQueue = multiprocessing.Queue()
    
    predictInQueue = multiprocessing.Queue()
    predictOutQueue = multiprocessing.Queue()

    # init the display
    renderWindow.setResolution(1024, 768)
    renderWindow.setUiEnabled(True)
    renderWindow.initMainWnd()
    renderWindow.initVkToy()
    renderWindow.setShader(
        "void mainImage( out vec4 fragColor, in vec2 fragCoord )\n{\n fragColor = vec4(0.0,1.0,1.0,1.0);\n}")
    
    # this needs to be a thread, otherwise the display can not be passed
    setShaderSrcThread = threading.Thread(target=setShaderSrcFn)
    setShaderSrcThread.start()

    # this needs to be a process, otherwise it will be conflict with the display
    frameInQueue = multiprocessing.Queue()
    frameOutQueue = multiprocessing.Queue()
    imgRender = multiprocessing.Process(target=getOneFrameFn, args=(frameInQueue, frameOutQueue))
    imgRender.start()

    predictThread = threading.Thread(target=predictFn)
    predictThread.start()

    # this needs to be a process, otherwise gradio will not work
    grThread = multiprocessing.Process(target=webMain, args=(displayInQueue, frameInQueue, frameOutQueue, predictInQueue, predictOutQueue))
    grThread.start()

    # display must be in the main thread
    renderWindow.renderUnlocked()
    # while True:
    #     pass
