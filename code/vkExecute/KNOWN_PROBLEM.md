(venv) [libreliu@libreliu-GCL-Arch toyDb]$ python manage.py trace --shader-id ttsyzf
2023-10-30 22:55:20,527 -                       databases.ShaderDb - INFO - Loaded 27911 offline shaders.
2023-10-30 22:55:20,540 -               cliFuncs.runImageOnlyTrace - INFO - Scheduling for 1 traces to be generated
  0%|                                                                                                                                                                                                                                                                | 0/1 [00:00<?, ?it/s]
2023-10-30 22:55:21,697 -               cliFuncs.runImageOnlyTrace - INFO - Use a timeout of 120 sec per experiment
  0%|                                                                                                                                                                                                                                                                | 0/1 [00:00<?, ?it/s]
  0%|                                                                                                                                                                                                                                                                | 0/1 [00:00<?, ?it/sSubprocess worker 0 started                                                                                                                                                                                                             | 0/1 [00:00<?, ?it/s, id=ttsyzf, success=0, fail=0]
INFO: Going to use NVIDIA GeForce RTX 3060 | file[context_vulkan.cpp] line[1024]
VUID-VkShaderModuleCreateInfo-pCode-01379(ERROR / SPEC): msgNum: 706474367 - Validation Error: [ VUID-VkShaderModuleCreateInfo-pCode-01379 ] | MessageID = 0x2a1bf17f | SPIR-V module not valid: OpPhi must appear within a non-entry block before all non-OpPhi instructions (except for OpLine, which can be mixed with OpPhi).
  %775 = OpPhi %bool %767 %753 %774 %768
 The Vulkan spec states: If pCode is a pointer to GLSL code, it must be valid GLSL code written to the GL_KHR_vulkan_glsl GLSL extension specification (https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkShaderModuleCreateInfo-pCode-01379)
    Objects: 0
INFO: Allocating new descriptor pool for thread[140501580330816] and name['Descriptor Cache #1] | file[avk.cpp] line[3556]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.40s/it, id=ttsyzf, success=0, fail=0]
Subprocess worker 0 received exiting command, exit█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.40s/it, id=ttsyzf, success=0, fail=0]
(venv) [libreliu@libreliu-GCL-Arch toyDb]$ 
[0] 0:python*                              