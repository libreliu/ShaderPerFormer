import os, logging, sys
import datetime
import json

from databases.ShaderDb import ShaderDB, ShaderProxy
import databases.ExperimentDb as ExperimentDb
import utils.SubprocessManager
from experiments.ImageOnlyRunner import (
  ImageOnlyRunner,
  ImageOnlyRunnerConfig
)
from utils.ExperimentDbHelpers import getOrAddEnvironment
from typing import Any, Dict

import tqdm

logger = logging.getLogger(__name__)

def testOneShader(
    shdrProxy: 'ShaderProxy',
    width: int, height: int,
    numCycles: int, numTrials: int,
    onlyCompile: bool=False,
    extraUniformKwargs: Dict[str, Any]=None):
  """Meant to be run in separate process"""
  
  cfg = ImageOnlyRunnerConfig.defaultNonTracedConfig()
  cfg.width = width
  cfg.height = height
  cfg.numCycles = numCycles
  cfg.numTrails = numTrials
  runner = ImageOnlyRunner(cfg)

  runner.loadShader(shdrProxy)
  if extraUniformKwargs is not None:
    uniformData = runner.fillUniform(**extraUniformKwargs)
  else:
    uniformData = runner.fillUniform()
  
  if onlyCompile:
    result = runner.exportShader()
  else:
    result = runner.run()

  return {
    "runnerResult": result,
    "uniformData": uniformData
  }

def cliRun(db, args):
  runnable_shaders = []
  shaderDB = ShaderDB(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../shaders"))
  shaderDB.scan_local()

  if args.shader_id != "":
    if args.shader_id in shaderDB.offlineShaders:
      runnable_shaders.append(args.shader_id)
    else:
      raise Exception(
        f"Unknown shader ID {args.shader_id}, did you try synchronizing the offline shaders?")
  else:
    runnable_shaders = [
      i for i in shaderDB.filter_attribute(["is_imageonly"])]

  # we'll delete the first, so we need trialcount >= 1
  assert (args.num_trials > 1)

  imageDir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../image-results/")
  if args.save_images:
    os.makedirs(imageDir, exist_ok=True)

  this_env = getOrAddEnvironment(args.comment)

  logger.info(
    f"Scheduling for {len(runnable_shaders)} shaders")

  shaderDB.load_all()

  successful_attempts = 0
  failed_attempts = 0

  if args.no_subprocess:
    logger.info("Using no subprocess mode - Test code will run in main thread")
    spm = utils.SubprocessManager.SubprocessManagerMock(testOneShader)
  else:
    timeout = None
    if not args.disable_timeout:
      timeout = args.timeout_shader_base * args.num_cycles * args.num_trials + args.timeout_extra
      logger.info(f"Use a timeout of {timeout} sec per experiment")
    else:
      logger.info(f"Timeout is disabled in the experiment")

    spm = utils.SubprocessManager.SubprocessManager(testOneShader, 0, True, timeout)

  pbar = tqdm.tqdm(runnable_shaders, file=sys.stdout)
  for shaderID in pbar:
    pbar.set_postfix(
      {"id": shaderID, "success": successful_attempts, "fail": failed_attempts})
    shdrProxy = shaderDB.offlineShaders[shaderID]

    res, success, errorMessage = spm.spawnExec(
      shdrProxy.get_slim_version(),
      args.run_width, args.run_height, args.num_cycles, args.num_trials, args.only_compile,
      {
        "iTime": args.iTime,
        "iFrame": args.iFrame
      }
    )

    if not success:
      assert(errorMessage is not None)

      errorReason = ExperimentDb.ErrorType.OTHER
      if "ShaderResultInconsistencyException" in errorMessage:
        errorReason = ExperimentDb.ErrorType.INCONSISTENT_RUN
      elif "ShaderCompilationException" in errorMessage:
        errorReason = ExperimentDb.ErrorType.SHADER_COMPILATION_FAILED
      elif errorMessage == utils.SubprocessManager.ERROR_REASON_CRASH_OR_TOO_LONG_TO_RESPOND:
        errorReason = ExperimentDb.ErrorType.TIMEOUT_OR_CRASH

      # Log the error message
      logger.error(f"[{shaderID}] Failed to measure performance; Reason: {errorReason}")
      logger.debug(f"[{shaderID}] Failed to measure performance; Detailed reason: {errorMessage}")

      # Log the failed run
      expr = ExperimentDb.ImageOnlyExperiment.create(
        time=datetime.datetime.now(),
        environment=this_env,
        augmentation=ExperimentDb.AugmentationType.NONE,
        width=args.run_width,
        height=args.run_height,
        shader_shadertoy_id=shaderID,
        shader=None,
        resource=None,
        trace=None,
        measurement=ExperimentDb.MeasurementType.DEFAULT,
        image_hash=None,
        num_cycles=args.num_cycles,
        num_trials=args.num_trials,
        errors=errorReason,
        results=""
      )
      expr.save()

      failed_attempts += 1
      continue

    if args.only_compile:
      uniformInst, _ = ExperimentDb.ImageOnlyResource.get_or_create(
        uniform_block=res["uniformData"]
      )
      uniformInst.save()

      shdrInst, _ = ExperimentDb.ImageOnlyShader.get_or_create(
        shader_id=shaderID, fragment_spv=ExperimentDb.packSpvToBytes(res["runnerResult"])
      )
      shdrInst.save()

      expr = ExperimentDb.ImageOnlyExperiment.create(
        time=datetime.datetime.now(),
        environment=this_env,
        augmentation=ExperimentDb.AugmentationType.NONE,
        width=args.run_width,
        height=args.run_height,
        shader_shadertoy_id=shaderID,
        shader=shdrInst,
        resource=uniformInst,
        trace=None,
        measurement=ExperimentDb.MeasurementType.DEFAULT,
        image_hash=None,
        num_cycles=args.num_cycles,
        num_trials=args.num_trials,
        errors=ExperimentDb.ErrorType.NONE,
        results=""
      )
      # logger.info(f"Experiment: {expr}")

      expr.save()

      successful_attempts += 1
    else:
      imgHash = ImageOnlyRunner.getImageHash(res["runnerResult"].imgData)
      if args.save_images:
        ImageOnlyRunner.saveImage(
          os.path.join(imageDir, f"./{shaderID}.png"),
          res["runnerResult"].imgData
        )

      # get or create resource instance
      uniformInst, _ = ExperimentDb.ImageOnlyResource.get_or_create(
        uniform_block=res["uniformData"]
      )
      uniformInst.save()

      # get or create shader instance
      shdrInst, _ = ExperimentDb.ImageOnlyShader.get_or_create(
        shader_id=shaderID, fragment_spv=ExperimentDb.packSpvToBytes(res["runnerResult"].fragShdrSpv)
      )
      shdrInst.save()

      expr = ExperimentDb.ImageOnlyExperiment.create(
        time=datetime.datetime.now(),
        environment=this_env,
        augmentation=ExperimentDb.AugmentationType.NONE,
        width=args.run_width,
        height=args.run_height,
        shader_shadertoy_id=shaderID,
        shader=shdrInst,
        resource=uniformInst,
        trace=None,
        measurement=ExperimentDb.MeasurementType.DEFAULT,
        image_hash=imgHash,
        num_cycles=args.num_cycles,
        num_trials=args.num_trials,
        errors=ExperimentDb.ErrorType.NONE,
        results=json.dumps(res["runnerResult"].trialTime.tolist())
      )
      # logger.info(f"Experiment: {expr}")

      expr.save()

      successful_attempts += 1

  spm.endExec()

  shaderDB.unload_all()

def register(parser):
  parser.add_argument(
    "--shader-id",
    type=str,
    default="",
    help="Specific shader ID to run"
  )
  parser.add_argument(
    '--save-images',
    action='store_true',
    help="save images of each run"
  )
  parser.add_argument(
    '--run-width',
    type=int,
    default=ExperimentDb.CANONICAL_WIDTH
  )
  parser.add_argument(
    '--run-height',
    type=int,
    default=ExperimentDb.CANONICAL_HEIGHT
  )
  parser.add_argument(
    '--comment',
    type=str,
    default="",
    help='comment of the environment'
  )
  parser.add_argument(
    '--num-cycles',
    type=int,
    default=ExperimentDb.CANONICAL_NUM_CYCLES
  )
  parser.add_argument(
    '--num-trials',
    type=int,
    default=ExperimentDb.CANONICAL_NUM_TRIALS
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
    '--timeout-shader-base',
    type=float,
    # Defaults to 0.33 FPS - that's 3sec, enough for most of the GPU
    # to raise a TDR when no detailed preemption enabled
    default=3,
    help="Max allowed time in seconds for one shader run"
      "(will be multiplied by numCycles & numTrials)"
  )
  parser.add_argument(
    '--timeout-extra',
    type=float,
    default=10,
    help="Extra seconds for compilation, running and other costs"
  )
  parser.add_argument(
    '--iTime',
    type=float,
    default=1.0,
    help="The specific iTime in Uniform"
  )
  parser.add_argument(
    '--iFrame',
    type=int,
    default=1,
    help="The specific iFrame in Uniform"
  )
  parser.add_argument(
    '--only-compile',
    action='store_true',
    help="Only compile the shader, no running"
  )