import sys
sys.path.insert(0, sys.path[0]+"/../")
sys.path.insert(0, sys.path[0]+"vkPredict/")
sys.path.insert(0, sys.path[0]+"toyDb/")
import os
import logging
import vkExecute
from vkPredict.toyDb.utils import SubprocessManager
from vkPredict.toyDb.experiments import ImageOnlyRunner
from vkPredict.toyDb.databases.ExperimentDb import packSpvToBytes
from vkPredict.model.ModelBuilder import build_model
from vkPredict.misc.TokenizerBuilder import build_tokenizer
from vkPredict.misc.dataCollator import DataCollatorWithPaddingAndTraceEmbedding
from vkPredict.misc.Directory import getVkPredictRootDir
from vkPredict.misc.normalization import LogNormalizer
from safetensors.torch import load_file, load_model
from typing import List
from dataclasses import dataclass
import torch
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ShaderData:
    fragShaderSrc: str
    width: int
    height: int
    iTime: float
    iFrame: int

class DataProcessor:
    def __init__(self, fragShaderSrcHead="", trace_embedding_dim=768):
        self.traceRunner = SubprocessManager.SubprocessManager(self.traceFn)
        self.tokenizer = build_tokenizer("HfTracedSpvTokenizer-multiple-entrypoint")
        self.dataCollator = DataCollatorWithPaddingAndTraceEmbedding(
            tokenizer=self.tokenizer,
            trace_embedding_method="onehot-base2",
            trace_embedding_dim=trace_embedding_dim,
            padding="longest",
            pad_to_multiple_of=8
        )
        self.shader = None
        self.fragShaderSrcHead = fragShaderSrcHead
        self.modelInput = None

    def setShader(self, fragShaderSrc: str, width: int, height: int, iTime: float, iFrame: int):
        self.shader = ShaderData(fragShaderSrc, width, height, iTime, iFrame)

    def process(self):
        if self.shader is None:
            logger.error(f"Shader not set")
            return None

        traceResult = self.trace()
        if traceResult is None:
            logger.error(f"Trace failed")
            return None

        processedData = self.postProcessTracedData(traceResult)
        self.modelInput = self.dataCollator([processedData])
    
    def getModelInput(self):
        return self.modelInput

    @staticmethod
    def traceFn(
        fragShaderSpv: List[str],
        width: int, height: int,
        iTime: float, iFrame: int,
        traceWithU64: bool
    ):
        cfg = ImageOnlyRunner.ImageOnlyRunnerConfig.defaultTracedConfig(traceWithU64=traceWithU64)
        cfg.width = width
        cfg.height = height
        runner = ImageOnlyRunner.ImageOnlyRunner(cfg)

        runner.loadShaderFromDb(fragShaderSpv)
        runner.uniformBlock = vkExecute.ImageOnlyExecutor.ImageUniformBlock()
        runner.uniformBlock.iResolution = [width, height, 0]
        runner.uniformBlock.iTime = iTime
        runner.uniformBlock.iFrame = iFrame

        result = runner.run()
        return result

    def trace(self):
        fragShaderSrc = self.fragShaderSrcHead + self.shader.fragShaderSrc
        fragShaderSpv, fragErrMsg = vkExecute.ShaderProcessor.compileShaderToSPIRV_Vulkan(
            vkExecute.ShaderProcessor.ShaderStages.FRAGMENT,
            fragShaderSrc,
            "FragShader"
        )
        if fragErrMsg != "":
            logger.error(f"Shader compile error: {fragErrMsg}")
            return None

        result, success, errorMessages = self.traceRunner.spawnExec(
            fragShaderSpv,
            self.shader.width, self.shader.height,
            self.shader.iTime, self.shader.iFrame,
            True
        )

        if not success:
            logger.error(f"Trace error: {errorMessages}")
            return None
        
        return {
            "fragSpv": packSpvToBytes(fragShaderSpv),
            "traceFragSpv": packSpvToBytes(result.traceFragShdrSpv),
            "bbIdxMap": result.id2TraceIdxMap,
            "bbTraceCounters": result.traceData,
            "timeMean": 0
        }

    def postProcessTracedData(self, elem):
        encoded_inputs = self.tokenizer(
            spvBinaryRepr=elem['fragSpv'],
            id2TraceIdxMap=elem['bbIdxMap'],
            traceCounters=elem['bbTraceCounters']
        )

        encoded_inputs['labels'] = elem['timeMean']

        return encoded_inputs


class Predictor:
    def __init__(self, targetModel, device, tokenizer):
        self.model = build_model(
            targetModel, "mse", 4096,
            tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        )
        self.model.to(device)
        self.model.eval()
        self.modelPathes = {}
        self.preds = {}
    
    def setModelPathes(self, modelPathes: dict):
        self.modelPathes = modelPathes

    def _loadModel(self, ModelPath):
        if ModelPath.endswith(".safetensors"):
            load_model(self.model, ModelPath)
        elif ModelPath.endswith(".pt") or ModelPath.endswith(".pth") or ModelPath.endswith(".bin"):
            self.model.load_state_dict(torch.load(ModelPath))
    
    def _predictOne(self, modelInput):
        with torch.no_grad():
            for key in modelInput:
                modelInput[key] = modelInput[key].to(self.model.device)
            
            return self.model(**modelInput)
    
    def predict(self, modelInput):
        for key in self.modelPathes:
            self._loadModel(self.modelPathes[key])
            self.preds[key] = self._predictOne(modelInput)['logits'].squeeze().cpu().numpy()
    
    def getPreds(self):
        for key in self.preds:
            self.preds[key] = LogNormalizer().invNormalize(self.preds[key])
        return self.preds
        


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # test data
    fragShaderSrc = '''#version 310 es
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

    float Circle( vec2 uv, vec2 p, float r, float blur )
    {
        float d = length(uv - p);
    float c = smoothstep(r, r-blur, d);
    return c;
    }

    float Hash( float h )
    {
    return h = fract(cos(h) * 5422.2465);
    }

    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        vec2 uv = fragCoord.xy / iResolution.xy;
    float c = 0.0;
        
    uv -= .5;
    uv.x *= iResolution.x / iResolution.y;
    float sizer = 1.0;
    float steper = .1;
    for(float i = -sizer; i<sizer; i+=steper)
        for(float j = -sizer; j<sizer; j+=steper)
        {	
        float timer = .5;
        float resetTimer = 7.0;
        if(c<=1.0){
            c += Circle(uv, vec2(i, j),sin(Hash(i))*cos(Hash(j))*(mod(iTime*timer, resetTimer)), sin(Hash(j)));
        }
        else if(c>=1.0)
        {
            c -= Circle(uv, vec2(i, j),sin(Hash(i))*cos(Hash(j))*(mod(iTime*timer, resetTimer)), sin(Hash(j)));     
        }
        }
    fragColor = vec4(vec3(c),1.0);
    }
    '''
    width = 1024
    height = 768
    iTime = 0.0
    iFrame = 0

    # init data processor
    dataProcessor = DataProcessor()
    dataProcessor.setShader(fragShaderSrc, width, height, iTime, iFrame)
    dataProcessor.process()
    modelInput = dataProcessor.getModelInput()

    # init predictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = dataProcessor.tokenizer
    predictor = Predictor("perfformer-layer9-regression-trace-input-embedding-xformers-memeff",device, tokenizer)

    # load model
    modelPaths = {}
    modelPaths["3060"] = os.path.join(getVkPredictRootDir(), "Perfformer-NoRope-Trace-Onehot-Base2-Regression-3060-With-Val-fp16-log-time_HfTracedSpvTokenizer-multiple-entrypoint_perfformer-layer9-regression-trace-input-embedding-xformers-memeff/model.safetensors")
    modelPaths["4060"] = os.path.join(getVkPredictRootDir(), "Perfformer-NoRope-Trace-Onehot-Base2-Regression-4060-With-Val-fp16-log-time_HfTracedSpvTokenizer-multiple-entrypoint_perfformer-layer9-regression-trace-input-embedding-xformers-memeff/model.safetensors")
    modelPaths["1660"] = os.path.join(getVkPredictRootDir(), "Perfformer-NoRope-Trace-Onehot-Base2-Regression-1660-With-Val-fp16-log-time_HfTracedSpvTokenizer-multiple-entrypoint_perfformer-layer9-regression-trace-input-embedding-xformers-memeff/model.safetensors")
    modelPaths["6600XT"] = os.path.join(getVkPredictRootDir(), "Perfformer-NoRope-Trace-Onehot-Base2-Regression-6600XT-With-Val-fp16-log-time_HfTracedSpvTokenizer-multiple-entrypoint_perfformer-layer9-regression-trace-input-embedding-xformers-memeff/model.safetensors")
    modelPaths["UHD630"] = os.path.join(getVkPredictRootDir(), "Perfformer-NoRope-Trace-Onehot-Base2-Regression-UHD630-With-Val-fp16-log-time_HfTracedSpvTokenizer-multiple-entrypoint_perfformer-layer9-regression-trace-input-embedding-xformers-memeff/model.safetensors")
    predictor.setModelPathes(modelPathes=modelPaths)

    # predict
    predictor.predict(modelInput)
    preds = predictor.getPreds()
    print(preds)