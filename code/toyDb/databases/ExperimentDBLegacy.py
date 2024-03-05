import platform
import argparse, subprocess, os, re
import logging
from . import ShaderDB
import tqdm
import numpy as np
import PIL.Image
import hashlib
import datetime
from .experiments import ImageOnly
from .experiments.SubprocessManager import SubprocessManager
from .utils.spv.analyzeSpv import SpvInstrClasses, SpvTextContext, OpsToClass
import io, json
from .experiments import SpvContext
from .experiments import SpvInstStat

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)



def testOneShader(
    shaderID, errorDir, imageDir, save_images,
    shdrProxy, w, h, num_cycles, num_trials):
    """Meant to be run in separate process"""

    success = True
    try:
        trialFps, imgData, fragShdrSpv = ImageOnly.testShader(
            shdrProxy, w, h, num_cycles, num_trials, verbose=False
        )
    except ImageOnly.ShaderCompilationException as e:
        with open(os.path.join(errorDir, f"{shaderID}-shader-error.log"), 'w') as f:
            f.write(e.glslangError)
        success = False
        
    except ImageOnly.ShaderResultInconsistencyException as e:
        with open(os.path.join(errorDir, f"inconsistent-shader-error.log"), 'a') as f:
            f.write(f"ShaderResultInconsistencyException with shader id = {shaderID}\n")
        success = False
    
    if not success:
        return {
            "success": False
        }

    fragment_spv_disasm = ImageOnly.disassemble(fragShdrSpv)

    if save_images:
        save_image(
            os.path.join(imageDir, f"./{shaderID}-1.png"),
            imgData
        )

    imgHash = get_image_hash(imgData)

    # delete the first one due to power feature
    res_mean = np.mean(trialFps[1:])
    res_stdev = np.std(trialFps[1:])

    return {
        "success": success,
        "imgHash": imgHash,
        "fragment_spv_disasm": fragment_spv_disasm,
        "res_mean": res_mean,
        "res_stdev": res_stdev
    }

def cli_run(db, args):
    runnable_shaders = []
    shaderDB = ShaderDB.ShaderDB(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./shaders"))
    shaderDB.scan_local()

    if args.shader_id != "":
        if args.shader_id in shaderDB.offlineShaders:
            runnable_shaders.append(args.shader_id)
        else:
            raise Exception(f"Unknown shader ID {args.shader_id}, did you try synchronizing the offline shaders?")
    else:
        runnable_shaders = [i for i in shaderDB.filter_attribute(["is_imageonly"])]

    assert(args.run_width is not None)
    assert(args.run_height is not None)

    # we'll delete the first, so we need trialcount >= 1
    assert(args.num_trials > 1)

    imageDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./results/")
    if args.save_images:
        os.makedirs(imageDir, exist_ok=True)

    errorDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./errors/")
    os.makedirs(errorDir, exist_ok=True)

    this_env = get_or_add_environment(args.comment)

    num_wh_groups = len(args.run_width.split(';'))
    if num_wh_groups != len(args.run_height.split(';')):
        raise Exception("Width and height have different groups")
    
    w_groups = args.run_width.split(';')
    h_groups = args.run_height.split(';')
    wh_groups = []
    for i in range(num_wh_groups):
        wh_groups.append((int(w_groups[i]), int(h_groups[i])))

    logger.info(f"Scheduling for {len(runnable_shaders)} shaders, wh_groups={wh_groups}")

    shaderDB.load_all()

    successful_attempts = 0
    failed_attempts = 0

    spm = SubprocessManager(testOneShader, 0, True)


    for w, h in wh_groups:
        pbar = tqdm.tqdm(runnable_shaders)
        for shaderID in pbar:
            pbar.set_postfix({"id": shaderID, "success": successful_attempts, "fail": failed_attempts})
            shdrProxy = shaderDB.offlineShaders[shaderID]

            res, success = spm.spawnExec(
                shaderID, errorDir, imageDir, args.save_images,
                shdrProxy.get_slim_version(), w, h, args.num_cycles, args.num_trials
            )

            if (not success) or res["success"] != True:
                failed_attempts += 1

                if not success:
                    with open(os.path.join(errorDir, f"crash-shader.log"), 'a') as f:
                        f.write(f"crash with shader id = {shaderID}\n")

                continue

            # get or create shader instance
            shdrInst, _ = ImageOnlyShader.get_or_create(
                shader_id=shaderID, fragment_spv=res["fragment_spv_disasm"]
            )
            shdrInst.save()

            # TODO: get or create uniform block

            expr = ImageOnlyExperiment.create(
                time=datetime.datetime.now(),
                environment=this_env,
                augmentation=0,
                width=w,
                height=h,
                shader=shdrInst,
                resource=None,
                measurement=0,
                image_hash=res["imgHash"],
                num_cycles=args.num_cycles,
                num_trials=args.num_trials,
                result_mean=res["res_mean"],
                result_stdev=res["res_stdev"]
            )
            # logger.info(f"Experiment: {expr}")
            
            expr.save()

            successful_attempts += 1

    spm.endExec()

    shaderDB.unload_all()

def analyze_one_spv(spvText):
    """peewee instances"""
    res = {
        "num_instr": 0,
        "num_instr_by_class": {k: 0 for k in SpvInstrClasses}
    }

    def analyzeCallback(opcode, result, ops):
        res["num_instr"] += 1
        res["num_instr_by_class"][OpsToClass[opcode]] += 1

    ctx = SpvTextContext(analyzeCallback)
    ctx.parse(spvText)
    return res

def analyze_one_spv_traced(expr):
    """expr: one record of ImageOnlyExperiments
    Returns dict, same as analyze_one_spv
    """
    
    fragInlinedStream = pack_bytes_to_bytes_stream(expr.trace.frag_inlined_spv)
    bbIdx2TraceIdx = {
        int(k): v for k, v in json.loads(expr.trace.bb_idx_map).items()
    }
    bbTraceCounters = json.loads(expr.trace.bb_trace_counters)
    
    parser = SpvContext.BinaryParser()
    binCtx = parser.parse(
        SpvContext.Grammar(SpvContext.SpvGrammarPath),
        fragInlinedStream
    )

    result = SpvInstStat.statWithTrace(
        binCtx, bbIdx2TraceIdx, bbTraceCounters
    )

    return {
        "num_instr": result.numInsts,
        "num_instr_by_class": result.numInstsByClass
    }


def cli_analyze_spv(args, canonicalShaders):
    numSpvInstr = []
    numSpvMemory = []
    numSpvArithmetic = []
    numSpvControlFlow = []
    numSpvConstantCreation = []

    pbar = tqdm.tqdm(canonicalShaders)
    pbar.set_description("Analyzing Shader SPIR-V")
    for shdr in pbar:
        res = analyze_one_spv(shdr.fragment_spv)
        numSpvInstr.append(res["num_instr"])
        numSpvMemory.append(res["num_instr_by_class"]["Memory"])
        numSpvArithmetic.append(res["num_instr_by_class"]["Arithmetic"])
        numSpvControlFlow.append(res["num_instr_by_class"]["Control-Flow"])
        numSpvConstantCreation.append(res["num_instr_by_class"]["Constant-Creation"])

    def do_combine_hist_plot(data: np.ndarray, suptitle, xlabel):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle(suptitle)
        
        max_95 = np.percentile(data, 95)
        ax1.hist(data, bins='auto', range=(0, max_95))
        # ax1.set_xlabel('# SPIR-V instructions')
        ax1.set_ylabel('# Shaders')
        
        ax2.hist(data, bins='auto')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('# Shaders')

        plt.show()

    do_combine_hist_plot(
        np.array(numSpvInstr),
        "SPIR-V Total Instructions (Total & 95 pencentile)",
        "# SPIR-V instructions"
    )
    do_combine_hist_plot(
        np.array(numSpvMemory),
        "SPIR-V Memory Instructions (Total & 95 pencentile)",
        "# SPIR-V instructions"
    )
    do_combine_hist_plot(
        np.array(numSpvArithmetic),
        "SPIR-V Arithmetic Instructions (Total & 95 pencentile)",
        "# SPIR-V instructions"
    )
    do_combine_hist_plot(
        np.array(numSpvControlFlow),
        "SPIR-V Control Flow Instructions (Total & 95 pencentile)",
        "# SPIR-V instructions"
    )
    do_combine_hist_plot(
        np.array(numSpvConstantCreation),
        "SPIR-V Constant Creation Instructions (Total & 95 pencentile)",
        "# SPIR-V instructions"
    )

def cli_analyze_votality(canonicalExprs):
    relativeVotality = []
    for expr in canonicalExprs:
        relativeVotality.append(
            expr.result_stdev / expr.result_mean
        )

    relativeVotality = np.array(relativeVotality)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("Experiment Relative Votality (stdev / mean), Total & 95 pencentile")
    
    max_95 = np.percentile(relativeVotality, 95)
    ax1.hist(relativeVotality, bins='auto', range=(0, max_95))
    # ax1.set_xlabel('# SPIR-V instructions')
    ax1.set_ylabel('# Shaders')
    
    ax2.hist(relativeVotality, bins='auto')
    ax2.set_xlabel("Relative votality")
    ax2.set_ylabel('# Shaders')

    plt.show()

def cli_analyze(db, args):
    if args.shader_id != "":
        canonicalExprs = ImageOnlyExperiment.select().where(
                            ImageOnlyShader.shader_id == args.shader_id,
                            ImageOnlyExperiment.num_cycles == 50,
                            ImageOnlyExperiment.num_trials == 10
                        ).join(ImageOnlyShader)
        logger.info(f"Selected {canonicalExprs}")
    else:
        # Get canonical set
        canonicalExprs = ImageOnlyExperiment.select().where(
            ImageOnlyExperiment.num_cycles == 50,
            ImageOnlyExperiment.num_trials == 10
        ).join(ImageOnlyShader)

    canonicalShaders = [expr.shader for expr in canonicalExprs]

    # Analyze 1: spir-v
    if args.analyze_spv:
        cli_analyze_spv(args, canonicalShaders)

    # Analyze 2: Shader measurement votality
    if args.analyze_votality:
        cli_analyze_votality(canonicalExprs)

def cli_fit(db, args):
    if not args.use_trace_data:
        if args.shader_id != "":
            canonicalExprs = ImageOnlyExperiment.select().join(ImageOnlyShader) \
                            .where(
                                ImageOnlyShader.shader_id == args.shader_id,
                                ImageOnlyExperiment.num_cycles == 50,
                                ImageOnlyExperiment.num_trials == 10
                            )
            logger.info(f"Selected {canonicalExprs}")
        else:
            # Get canonical set
            canonicalExprs = ImageOnlyExperiment.select().where(
                ImageOnlyExperiment.num_cycles == 50,
                ImageOnlyExperiment.num_trials == 10
            ).join(ImageOnlyShader)
    else:
        if args.shader_id != "":
            canonicalExprs = ImageOnlyExperiment.select().join(ImageOnlyShader) \
                            .where(
                                ImageOnlyShader.shader_id == args.shader_id,
                                ImageOnlyExperiment.num_cycles == 50,
                                ImageOnlyExperiment.num_trials == 10,
                                ImageOnlyExperiment.trace_id.is_null(False)
                            )
            logger.info(f"Selected {canonicalExprs}")
        else:
            # Get canonical set
            canonicalExprs = ImageOnlyExperiment.select().where(
                ImageOnlyExperiment.num_cycles == 50,
                ImageOnlyExperiment.num_trials == 10,
                ImageOnlyExperiment.trace_id.is_null(False)
            ).join(ImageOnlyShader)

    logger.info(f"Selected {len(canonicalExprs)} candidates")

    num_input_feature = 2
    input_feature = np.ndarray((len(canonicalExprs), num_input_feature))
    target = np.ndarray((len(canonicalExprs),))

    pbar = tqdm.tqdm(enumerate(canonicalExprs))
    pbar.set_description("Analyzing Experiments")
    for idx, expr in pbar:
        if args.use_trace_data:
            res = analyze_one_spv_traced(expr)
        else:
            res = analyze_one_spv(expr.shader.fragment_spv)
        input_feature[idx, 0] = res["num_instr_by_class"]["Arithmetic"]
        input_feature[idx, 1] = res["num_instr_by_class"]["Memory"]
        if args.reciprocal:
            target[idx] = 1.0 / expr.result_mean
        else:
            target[idx] = expr.result_mean
    
        if args.verbose:
            print(res)

    if True:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        ax.scatter(input_feature[:, 0], input_feature[:, 1], target, marker='^')
        ax.set_xlabel("# Arithmetic")
        ax.set_ylabel("# Memory")

        if args.reciprocal:
            ax.set_zlabel("Mean frame time (sec)")
        else:
            ax.set_zlabel("Mean FPS")
        
        plt.show()

    # # linear fit
    # A = np.vstack([input_feature, np.ones(len(input_feature))]).T

    # weight, residual, rank, s = np.linalg.lstsq(A, target, rcond=None)
    # logger.info(f"weight={weight}, residual={residual}")

    # if True:
    #     _ = plt.plot(input_feature, target, 'o', label='Original data')
    #     _ = plt.plot(input_feature, weight[0] * input_feature + weight[1], 'r', label='Fitted line')
    #     _ = plt.legend()
    #     plt.show()

def traceOneShader(
    shaderID, errorDir, expectedImageHash,
    shdrProxy, w, h, verbose):
    """Meant to be run in separate process"""

    try:
        results = ImageOnly.traceShader(
            shdrProxy, w, h
        )
    except ImageOnly.ShaderCompilationException as e:
        with open(os.path.join(errorDir, f"{shaderID}-trace-error.log"), 'w') as f:
            f.write(e.glslangError)
        success = False
    
    # checking for image hash
    imgHash = get_image_hash(results["imgData"])
    if imgHash != expectedImageHash:
        logger.warning(f"Hash mismatch for shader trace {shaderID}, dump photo")

        # TODO: implement me
        save_image(
            os.path.join(errorDir, f"./{shaderID}-mismatch.png"),
            results["imgData"]
        )

    print(results["traceData"])

    if verbose:
        show_traced_shader(results["fragInlinedSpv"])
    
    return {
        "fps": results["fps"], "imgHash": imgHash, 
        "traceData": results["traceData"],
        "id2TraceIdxMap": results["id2TraceIdxMap"],
        "fragInlinedSpv": results["fragInlinedSpv"],
        "traceFragShdrSpv": results["traceFragShdrSpv"]
    }
    

# TODO: complete me
def show_traced_shader(
        fragInlinedSpv
    ):
    fragInlinedStream = pack_spv_to_bytes_stream(fragInlinedSpv)

    parser = SpvContext.BinaryParser()
    binCtx = parser.parse(
        SpvContext.Grammar(SpvContext.SpvGrammarPath),
        fragInlinedStream
    )
    
    print(binCtx.toTextRepr())

def cli_trace(db, args):
    shaderDB = ShaderDB.ShaderDB(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./shaders"))
    shaderDB.scan_local()

    # TODO: add option to override traced
    if args.shader_id is not None:
        # find candidate runs
        candidateExprs = ImageOnlyExperiment.select().where(
            ImageOnlyShader.shader_id == args.shader_id,
            ImageOnlyExperiment.num_cycles == 50,
            ImageOnlyExperiment.num_trials == 10
        ).join(ImageOnlyShader)

        logger.info(f"Selected {len(candidateExprs)} exprs to gen trace")
        
        if len(candidateExprs) > 1:
            logger.warn(f"Got multiple matching candidate, choose first")
            candidateExprs = [candidateExprs[0]]
    else:
        candidateExprs = ImageOnlyExperiment.select().where(
            ImageOnlyExperiment.num_cycles == 50,
            ImageOnlyExperiment.num_trials == 10,
            ImageOnlyExperiment.trace.is_null(True)
        ).join(ImageOnlyShader)

        logger.info(f"Selected {len(candidateExprs)} exprs to gen trace")
    
    shaderDB.load_all()

    imageDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./trace-results/")
    os.makedirs(imageDir, exist_ok=True)

    errorDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./trace-errors/")
    os.makedirs(errorDir, exist_ok=True)

    pbar = tqdm.tqdm(candidateExprs)
    successful_attempts = 0
    failed_attempts = 0

    spm = SubprocessManager(traceOneShader, 0, True)

    for expr in pbar:
        shaderID = expr.shader.shader_id
        shdrProxy = shaderDB.offlineShaders[shaderID]

        pbar.set_postfix({
            "id": shaderID, "success": successful_attempts, "fail": failed_attempts
        })

        width = expr.width
        height = expr.height
        shader = expr.shader
        origImgHash = expr.image_hash

        result, success = spm.spawnExec(
            shaderID, errorDir, origImgHash, shdrProxy, width, height, args.verbose
        )

        if not success:
            failed_attempts += 1
            continue

        if expr.trace is not None:
            logger.warning(f"experiment {expr.id} on {shaderID} has already recoreded with a trace, ignore")

            failed_attempts += 1
            continue
        
        fragInlinedSpvBinaryRepr = bytes(map(lambda x: ord(x), result["fragInlinedSpv"]))

        expr.trace = ImageOnlyTrace.create(
            bb_idx_map=json.dumps(result["id2TraceIdxMap"]),
            bb_trace_counters=json.dumps(result["traceData"]),
            frag_inlined_spv=fragInlinedSpvBinaryRepr
        )

        expr.save()
        successful_attempts += 1

    spm.endExec()

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)40s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(prog='ExperimentDB')
    parser.add_argument('--database-file', type=str, default=None, help="Database file path")

    subparsers = parser.add_subparsers(dest='command')

    create_sp = subparsers.add_parser("create")

    run_sp = subparsers.add_parser("run")
    run_sp.add_argument(
        "--shader-id",
        type=str,
        default="",
        help="Shader ID to run"
    )
    run_sp.add_argument(
        '--execution-policy',
        type=str,
        default='timeframe-replay',
        help="Choose between {timeframe-replay, first-frame-only}"
    )
    run_sp.add_argument(
        '--save-images',
        action='store_true',
        help="save images of each run"
    )
    run_sp.add_argument(
        '--run-width',
        type=str,
        help='width of the render output, use ; to separate between groups'
    )
    run_sp.add_argument(
        '--run-height',
        type=str,
        help='height of the render output, use ; to separate between groups'
    )
    run_sp.add_argument(
        '--comment',
        type=str,
        default="",
        help='comment of the environment'
    )
    run_sp.add_argument(
        '--num-cycles',
        type=int,
        default=50
    )
    run_sp.add_argument(
        '--num-trials',
        type=int,
        default=10
    )

    trace_sp = subparsers.add_parser("trace")
    trace_sp.add_argument(
        "--shader-id",
        default=None,
        help="Only filter selected if specified"
    )
    trace_sp.add_argument(
        "--verbose",
        action='store_true'
    )

    inspect_sp = subparsers.add_parser("inspect")
    inspect_sp.add_argument("shader_id", help="the shader to be inspected")

    analyze_sp = subparsers.add_parser("analyze")
    # todo: implement me
    analyze_sp.add_argument(
        '--shader-id',
        type=str,
        default="",
        help="specific shader ID to analyze"
    )
    analyze_sp.add_argument(
        '--analyze-spv',
        action='store_true',
        help="run spir-v analyze"
    )
    analyze_sp.add_argument(
        '--analyze-votality',
        action='store_true'
    )

    fit_sp = subparsers.add_parser("fit")
    fit_sp.add_argument(
        '--shader-id',
        type=str,
        default="",
        help="specific shader ID to analyze"
    )
    fit_sp.add_argument(
        '--use-trace-data',
        action='store_true'
    )
    fit_sp.add_argument(
        "--verbose",
        action='store_true'
    )
    fit_sp.add_argument(
        "--reciprocal",
        action='store_true'
    )


    args = parser.parse_args()

    if args.database_file is not None:
        db.init(args.database_file)
    else:
        init_from_default_db()

    if args.command == 'create':
        cli_create(db, args)
    elif args.command == 'run':
        cli_run(db, args)
    elif args.command == 'trace':
        cli_trace(db, args)
    elif args.command == 'analyze':
        cli_analyze(db, args)
    elif args.command == 'fit':
        cli_fit(db, args)
    else:
        logger.error(f"Unexpected command {args.command}, abort.")