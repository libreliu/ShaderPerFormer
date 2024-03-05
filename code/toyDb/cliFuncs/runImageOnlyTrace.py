import os, logging
import json
import tqdm
import sys
import vkExecute

from databases.ShaderDb import ShaderDB, ShaderProxy
from databases.ExperimentDb import (
  getCanonicalImageOnlyExperiments,
  getSuccessfulImageOnlyExperiments,
  getAugmentedExprsForTrace,
  ImageOnlyTrace,
  packSpvToBytes,
  packBytesToSpv
)
from utils import SubprocessManager
import experiments.ImageOnlyRunner as ImageOnlyRunner
from typing import List

logger = logging.getLogger(__name__)

def traceOneShader(
  fragShdrSpv: List[str],
  width: int, height: int,
  uniformData: bytes,
  traceWithU64: bool
):
  
  cfg = ImageOnlyRunner.ImageOnlyRunnerConfig.defaultTracedConfig(traceWithU64=traceWithU64)
  cfg.width = width
  cfg.height = height
  runner = ImageOnlyRunner.ImageOnlyRunner(cfg)

  runner.loadShaderFromDb(fragShdrSpv)
  runner.uniformBlock = vkExecute.ImageOnlyExecutor.ImageUniformBlock()
  runner.uniformBlock.importFromBytes(uniformData)
  
  result = runner.run()
  return result

def cliTrace(db, args):

  if args.all_successful:
    getFn = getSuccessfulImageOnlyExperiments
  elif args.augment_trace:
    getFn = getAugmentedExprsForTrace
  else:
    getFn = getCanonicalImageOnlyExperiments

  candidateExprs = getFn(
    None if args.shader_id == "" else args.shader_id,
    True,
    None if args.environment_id == -1 else args.environment_id
  )

  if args.shader_id != "" and len(candidateExprs) > 1:
    logger.warning(f"Got multiple matching candidate, choose first")
    candidateExprs = [candidateExprs[0]]

  logger.info(
    f"Scheduling for {len(candidateExprs)} traces to be generated")

  imageDir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../image-trace-results/")
  if args.save_images:
    os.makedirs(imageDir, exist_ok=True)

  pbar = tqdm.tqdm(candidateExprs)
  successful_attempts = 0
  failed_attempts = 0

  if args.no_subprocess:
    logger.info("Using no subprocess mode - Test code will run in main thread")
    spm = SubprocessManager.SubprocessManagerMock(traceOneShader)
  else:
    timeout = None
    if not args.disable_timeout:
      timeout = args.timeout
      logger.info(f"Use a timeout of {timeout} sec per experiment")
    else:
      logger.info(f"Timeout is disabled in the experiment")

    spm = SubprocessManager.SubprocessManager(traceOneShader, 0, True, timeout)

  # pbar = tqdm.tqdm(candidateExprs, file=sys.stdout)
  for expr in pbar:
    shaderID = expr.shader.shader_id

    pbar.set_postfix({
      "id": shaderID, "success": successful_attempts, "fail": failed_attempts
    })

    result, success, errorMessage = spm.spawnExec(
      packBytesToSpv(expr.shader.fragment_spv),
      expr.width, expr.height, expr.resource.uniform_block,
      not args.trace_with_u32
    )

    if not success:
      assert(errorMessage is not None)
      logger.error(f"[{shaderID}] Failed to run trace; Reason: {errorMessage}")

      failed_attempts += 1
      continue

    imgHash = ImageOnlyRunner.ImageOnlyRunner.getImageHash(result.imgData)
    if args.save_images:
      ImageOnlyRunner.ImageOnlyRunner.saveImage(
        os.path.join(imageDir, f"./{shaderID}-{expr.width}-{expr.height}.png"),
        result.imgData
      )
    if imgHash != expr.image_hash:
      logger.warning(f"[{shaderID}] have different hash between trace run and original run")

    expr.trace = ImageOnlyTrace.create(
      # NOTE: dict[int, int] will be represented as dict[str, int] under json
      bb_idx_map=json.dumps(result.id2TraceIdxMap),
      bb_trace_counters=json.dumps(result.traceData),
      traced_fragment_spv=packSpvToBytes(result.traceFragShdrSpv)
    )

    expr.save()
    successful_attempts += 1

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
    '--all-successful',
    action='store_true',
    help="Trace all successful experiment runs instead of only canonical ones"
  )
  parser.add_argument(
    '--augment-trace',
    action='store_true',
    help="Trace all augmented but not run experiments"
  )