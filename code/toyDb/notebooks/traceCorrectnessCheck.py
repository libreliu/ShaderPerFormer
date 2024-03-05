# %% [markdown]
# This is to check the correctness of trace generation functionality


# %%
import os, sys
sys.path.append(os.path.join(os.path.abspath(""), '../'))

from databases import ExperimentDb
import vkExecute

from tqdm import tqdm

ExperimentDb.init_from_default_db()

stopAtFirstError = False
allShader = ExperimentDb.ImageOnlyShader.select()
spvProc = vkExecute.SpvProcessor()

origFail = set()
traceU32Fail = set()
traceU64Fail = set()
allSuccess = set()
numAllShaders = len(allShader)

for shader in tqdm(allShader):
    shaderSpv = ExperimentDb.packBytesToSpv(shader.fragment_spv)
    spvProc.loadSpv(shaderSpv)
    
    resOrig = spvProc.validate()
    if not resOrig[0]:
        tqdm.write(f"{shader.shader_id} failed in validating original, skip")
        print(f"Reason: \n{resOrig[1]}")
        origFail.add(shader.shader_id)
        continue

    # u32Ver
    spvProc.instrumentBasicBlockTrace(False)
    resU32 = spvProc.validate()
    if not resU32[0]:
        tqdm.write(f"{shader.shader_id} failed in validating u32 traced version")
        print(f"Reason: \n{resU32[1]}")
        traceU32Fail.add(shader.shader_id)

        if stopAtFirstError:
            break
        
    # u64Ver
    shaderSpv = ExperimentDb.packBytesToSpv(shader.fragment_spv)
    spvProc.loadSpv(shaderSpv)
    spvProc.instrumentBasicBlockTrace(True)
    resU64 = spvProc.validate()
    if not resU64[0]:
        tqdm.write(f"{shader.shader_id} failed in validating u64 traced version")
        print(f"Reason: \n{resU64[1]}")
        traceU64Fail.add(shader.shader_id)

        if stopAtFirstError:
            break
    
    if resU32[0] and resU64[0]:
        allSuccess.add(shader.shader_id)


# %%
print(f"origFail: {origFail}")
print(f"traceU32Fail: {traceU32Fail}")
print(f"traceU64Fail: {traceU64Fail}")
