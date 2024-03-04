# ShaderPerFormer

This is the repository of codes in our [ACM SIGGRAPH i3D'2024](https://i3dsymposium.org/2024/) paper "ShaderPerFormer: Platform-independent Context-aware Shader Performance Predictor".

![Pipeline Overview](image/PipelineOverview.png)

<!--
Consider citing our work with the following BibTeX command:
```
TODO
```
-->

## Contents

### Codes

`code/` contains the code for our paper.

### Datasets

`dataset/` contains the dataset collected for our paper.

### Pretrained Models

`pretrained/` contains pretrained model for our paper.

## Usage HOWTO

1. Install `vkExecute`, and necessary PyPI packages
2. Retrieve all the shaders snapshoted from Shadertoy website **OR** use our snapshot around Feb. 2023 ([link](); Extract them into `code/toyDb/shaders`)
   1. Register account on [Shadertoy.com](https://www.shadertoy.com) and apply for an api key
   2. Put the key into `code/toyDb/apiKey.txt`
   3. Run `toyDb/getShaders.py`
3. Run measurements
4. Export measurement to `intermediate` file(s) for use in vkPredict
5. Train the ShaderPerformer (and baseline models, if you want)
6. Test on trained models; optionally use **vkToy** to compare your shaders to the rest

## Limitations

Currently only image pass only (=single pass) and uniform block input only (i.e. no texture read / write) inputs are supported.

## Licenses

- Works by us: Licensed under MIT license.
- Third-party: Licenses may vary.
- Shadertoy: Shadertoy shaders have their respective licenses, for detail please refer to their own licenses.