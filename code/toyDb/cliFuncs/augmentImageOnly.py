import os, logging
import json
import tqdm
import sys
import vkExecute
import datetime

from databases.ShaderDb import ShaderDB, ShaderProxy
import databases.ExperimentDb as ExperimentDb
from databases.ExperimentDb import (
  getAugmentedImageOnlyExperiments,
  getAugmentedButNotRunExperiments,
  ImageOnlyTrace,
  ImageOnlyShader,
  ImageOnlyExperiment,
  packSpvToBytes,
  packBytesToSpv,
)
from utils import SubprocessManager
import experiments.ImageOnlyRunner as ImageOnlyRunner
from utils.ExperimentDbHelpers import getEnvironment
from utils.AugmentationManager import (AugManager, AugManagerConfig)

logger = logging.getLogger(__name__)

def runShader(
  fragShdrSpv: list[str],
  uniformData: bytes,
  width: int=ExperimentDb.CANONICAL_WIDTH,
  height: int=ExperimentDb.CANONICAL_HEIGHT,
  numCycles: int=ExperimentDb.CANONICAL_NUM_CYCLES,
  numTrials: int=ExperimentDb.CANONICAL_NUM_TRIALS
) -> list[ImageOnlyRunner.ImageOnlyRunnerResult]:
  
  cfg = ImageOnlyRunner.ImageOnlyRunnerConfig\
    .defaultNonTracedConfig(width=width, height=height, numCycles=numCycles, numTrails=numTrials)

  runner = ImageOnlyRunner.ImageOnlyRunner(cfg)
  runner.loadShaderFromDb(fragShdrSpv)
  runner.uniformBlock = vkExecute.ImageOnlyExecutor.ImageUniformBlock()
  runner.uniformBlock.importFromBytes(uniformData)
  result = runner.run()

  return result

def cliAugment(db, args):
  this_env = getEnvironment(args.env_comment)
  if not this_env:
    raise Exception("No existing environment found!")

  exprs = getAugmentedImageOnlyExperiments(
    None if args.aug_parent_flag == "" else AugManager.augFlagToType(args.aug_parent_flag),
    None if args.shader_id == "" else args.shader_id,
    None if args.environment_id == -1 else args.environment_id,
    1000 if args.limit == -1 else args.limit
  )

  if args.shader_id != "" and len(exprs) > 1:
    logger.warning(f"Got multiple matching candidate, choose first")
    exprs = [exprs[0]]

  logger.info(
    f"Scheduling for {len(exprs)} traces to be generated")

  imageDir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../image-augment-results/")
  if args.save_images:
    os.makedirs(imageDir, exist_ok=True)

  pbar = tqdm.tqdm(exprs)
  successful_attempts = 0
  failed_attempts = 0

  if args.no_subprocess:
    logger.info("Using no subprocess mode - Test code will run in main thread")
    spm = SubprocessManager.SubprocessManagerMock(runShader)
  else:
    timeout = None
    if not args.disable_timeout:
      timeout = args.timeout
      logger.info(f"Use a timeout of {timeout} sec per experiment")
    else:
      logger.info(f"Timeout is disabled in the experiment")

    spm = SubprocessManager.SubprocessManager(runShader, 0, True, timeout)

  augConfig = AugManagerConfig(this_env.id)

  for expr in pbar:
    shaderID = expr.shader.shader_id
    resourceId = expr.resource
    num_cycles = expr.num_cycles
    num_trials = expr.num_trials

    pbar.set_postfix({
      "id": shaderID, "success": successful_attempts, "fail": failed_attempts
    })

    excludedPasses = []
    for child in expr.children:
      excludedPasses.append(child.augmentation)

    augManager = AugManager(augConfig)
    augManager.initShaderAncestor(expr.id)
    if args.random_times > 0:
      augManager.initAugListWithRandomPass(args.random_times, excludedPasses)
    else:
      augManager.initAugListFromFlags([args.augment_flag])
    augManager.augment()
    augManager.evalDistance()
    shaders = augManager.exportRunAugSpv()

    results = []
    errors = []
    for shader in shaders:
      result, success, errorMessage = spm.spawnExec(shader, resourceId.uniform_block)

      if not success:
        assert(errorMessage is not None)

        errorReason = ExperimentDb.ErrorType.OTHER
        if "ShaderResultInconsistencyException" in errorMessage:
          errorReason = ExperimentDb.ErrorType.INCONSISTENT_RUN
        elif "ShaderCompilationException" in errorMessage:
          errorReason = ExperimentDb.ErrorType.SHADER_COMPILATION_FAILED
        elif errorMessage == SubprocessManager.ERROR_REASON_CRASH_OR_TOO_LONG_TO_RESPOND:
          errorReason = ExperimentDb.ErrorType.TIMEOUT_OR_CRASH

        logger.error(f"[{shaderID}] Failed to run; Reason: {errorMessage}")
        errors.append(errorReason)
        results.append(None)
        failed_attempts += 1
        continue
      else:
        results.append(result)
        errors.append(ExperimentDb.ErrorType.NONE)
        successful_attempts += 1

    parentNode = expr
    for augNode in augManager.augNodeList:
      augType = augNode.augType
      imgHash = None
      shdrInst = None
      error = ExperimentDb.ErrorType.AUGMENTED_BUT_SAME
      renderResult = None
      resultTime = ""
      
      if augNode.disFromLast > 0:
        renderResult = results.pop(0)
        error = errors.pop(0)
        if renderResult is not None:
          resultTime = json.dumps(renderResult.trialTime.tolist())
          imgHash = ImageOnlyRunner.ImageOnlyRunner.getImageHash(renderResult.imgData)
          if args.save_images:
            ImageOnlyRunner.ImageOnlyRunner.saveImage(
              os.path.join(imageDir, f"./{shaderID}.png"),
              renderResult.imgData
            )
          # save the shader
          shdrInst, _ = ImageOnlyShader.get_or_create(
            shader_id=shaderID, fragment_spv = packSpvToBytes(renderResult.fragShdrSpv)
          )
          shdrInst.save()

      exprNew = ImageOnlyExperiment.create(
        time=datetime.datetime.now(),
        environment=this_env,
        augmentation=augType,
        width=augConfig.width,
        height=augConfig.height,
        shader_shadertoy_id=shaderID,
        shader=shdrInst,
        resource=resourceId,
        trace=None,
        parent=parentNode,
        measurement=ExperimentDb.MeasurementType.DEFAULT,
        image_hash=imgHash,
        num_cycles=num_cycles,
        num_trials=num_trials,
        errors=error,
        results=resultTime
      )

      exprNew.save()
      parentNode = exprNew

  spm.endExec()




def cliAugmentRun(db, args):

  exprs = getAugmentedButNotRunExperiments(
    None if args.augment_flag == "" else AugManager.augFlagToType(args.augment_flag),
    None if args.shader_id == "" else args.shader_id,
    None if args.environment_id == -1 else args.environment_id,
  )

  if args.shader_id != "" and len(exprs) > 1:
    logger.warning(f"Got multiple matching candidate, choose first")
    exprs = [exprs[0]]

  logger.info(
    f"Scheduling for {len(exprs)} traces to be generated")

  imageDir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../image-augment-results/")
  if args.save_images:
    os.makedirs(imageDir, exist_ok=True)

  pbar = tqdm.tqdm(exprs)
  successful_attempts = 0
  failed_attempts = 0

  if args.no_subprocess:
    logger.info("Using no subprocess mode - Test code will run in main thread")
    spm = SubprocessManager.SubprocessManagerMock(runShader)
  else:
    timeout = None
    if not args.disable_timeout:
      timeout = args.timeout
      logger.info(f"Use a timeout of {timeout} sec per experiment")
    else:
      logger.info(f"Timeout is disabled in the experiment")

    spm = SubprocessManager.SubprocessManager(runShader, 0, True, timeout)

  for expr in pbar:
    shaderID = expr.shader.shader_id
    resourceId = expr.resource

    pbar.set_postfix({
      "id": shaderID, "success": successful_attempts, "fail": failed_attempts
    })

    result, success, errorMessage = spm.spawnExec(
      packBytesToSpv(expr.shader.fragment_spv),
      resourceId.uniform_block,
      expr.width, expr.height,
      expr.num_cycles, expr.num_trials
    )

    if not success:
      assert(errorMessage is not None)

      errorReason = ExperimentDb.ErrorType.OTHER
      if "ShaderResultInconsistencyException" in errorMessage:
        errorReason = ExperimentDb.ErrorType.INCONSISTENT_RUN
      elif "ShaderCompilationException" in errorMessage:
        errorReason = ExperimentDb.ErrorType.SHADER_COMPILATION_FAILED
      elif errorMessage == SubprocessManager.ERROR_REASON_CRASH_OR_TOO_LONG_TO_RESPOND:
        errorReason = ExperimentDb.ErrorType.TIMEOUT_OR_CRASH

      logger.error(f"[{shaderID}] Failed to run; Reason: {errorMessage}")
      failed_attempts += 1
      continue
    else:
      errorReason = ExperimentDb.ErrorType.NONE
      successful_attempts += 1
  
    if result is not None:
      resultTime = json.dumps(result.trialTime.tolist())
      imgHash = ImageOnlyRunner.ImageOnlyRunner.getImageHash(result.imgData)
      expr.results = resultTime
      expr.image_hash = imgHash
      if args.save_images:
        ImageOnlyRunner.ImageOnlyRunner.saveImage(
          os.path.join(imageDir, f"./{shaderID}.png"),
          result.imgData
        )
      
    expr.errors = errorReason
    
    expr.save()

  spm.endExec()

def register(parser):
  parser.add_argument(
    "--shader-id",
    type=str,
    default="",
    help="Only filter selected if specified"
  )
  parser.add_argument(
    "--environment-id",
    type=int,
    default=-1,
    help="Specify detailed environment to filter; -1 to disable"
  )
  parser.add_argument(
    "--verbose",
    action='store_true'
  )
  parser.add_argument(
    '--save-images',
    action='store_true',
    help="save images of each run"
  )
  parser.add_argument(
    '--no-subprocess',
    action='store_true',
    help="No running things in child process"
  )
  parser.add_argument(
    '--disable-timeout',
    action='store_true',
    help="No timeout while carrying measurement; NOTE: of no use when --no-subprocess"
  )
  parser.add_argument(
    '--timeout',
    type=float,
    default=120,
    help="Timeout in seconds for one trace experiment run"
  )
  parser.add_argument(
    '--trace-with-u32',
    action='store_true',
    help="Trace with U32 instead of U64. " \
      "U64 tracing are less likely to overflow but requires Core 1.2 shaderBufferInt64Atomics, " \
      "which might be missing on some Apple, Intel and other mobile platform GPUs."
  )
  parser.add_argument(
    '--augment-flag',
    type=str,
    default="",
    help="Flag to augment the shader with"
  )
  parser.add_argument(
    '--random-times',
    type=int,
    default=1,
    help="Number of random times to augment the shader"
  )