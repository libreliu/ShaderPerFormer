from databases.ExperimentDb import db, Environment
from experiments.ImageOnlyRunner import initImageOnlyExecutor

import platform
import os, subprocess, re
import logging
import vkExecute

logger = logging.getLogger(__name__)

# helper functions

# https://stackoverflow.com/questions/4842448/getting-processor-information-in-python
def getCpuModel():
    """Linux and Windows supported"""
    cpu_model = None
    if platform.system() == "Windows":
        cpu_model = platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        cpu_model = subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                cpu_model = re.sub(".*model name.*:", "", line, 1)
    if cpu_model is None:
        logger.warning("Unable to detect CPU model, return empty string as walkaround")
        cpu_model = ""

    return cpu_model

def getGpuDescriptions():
    executor = vkExecute.ImageOnlyExecutor()
    # TODO: relax - but this might require careful handling of trace code interop (if we have)
    initImageOnlyExecutor(executor, True, True)

    # e.g. 'AMD Radeon Graphics (RADV GFX1100)'
    gpu = executor.getDeviceName()

    # e.g. 'radv Mesa 23.2.1-arch1.2'
    gpuDriver = executor.getDriverDescription()

    return gpu, gpuDriver

@db.atomic()
def getOrAddEnvironment(comment: str):
    uname_result = platform.uname()
    node = uname_result.node
    os = f"{uname_result.system} {uname_result.release}"
    cpu = getCpuModel()
    gpu, gpu_driver = getGpuDescriptions()
    
    inst, is_created = Environment.get_or_create(
        os=os, cpu=cpu, gpu=gpu, gpu_driver=gpu_driver,
        node=node, comment=comment
    )
    return inst

def getEnvironment(comment: str):
    uname_result = platform.uname()
    node = uname_result.node
    os = f"{uname_result.system} {uname_result.release}"
    cpu = getCpuModel()
    gpu, gpu_driver = getGpuDescriptions()
    
    inst = Environment.get_or_none(
        os=os, cpu=cpu, gpu=gpu, gpu_driver=gpu_driver,
        node=node, comment=comment
    )
    return inst