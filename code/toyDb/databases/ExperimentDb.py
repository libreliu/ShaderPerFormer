import peewee as pw
import os
import io
import logging
import struct

from enum import IntEnum
from typing import List, Any, Dict

logger = logging.getLogger(__name__)

# Some definitions on canonical runs
CANONICAL_NUM_CYCLES = 30
CANONICAL_NUM_TRIALS = 10
CANONICAL_WIDTH = 1024
CANONICAL_HEIGHT = 768


db = pw.SqliteDatabase(None)
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../dataset/experiments.db"
)

def init_from_default_db():
    db.init(DEFAULT_DB_PATH, pragmas={'foreign_keys': 1})

def init_from_in_memory_db():
    """This is useful for debugging"""
    db.init(":memory:", pragmas={'foreign_keys': 1})

class BaseModel(pw.Model):
    class Meta:
        database = db

class Environment(BaseModel):
    # hostname
    node = pw.CharField()
    os = pw.CharField()
    cpu = pw.CharField()
    gpu = pw.TextField()
    gpu_driver = pw.TextField()
    comment = pw.TextField()

class AugmentationType(IntEnum):
    NONE = 0

    STRIP_DEBUG=10000
    STRIP_REFLECT=10001
    STRIP_NONSEMANTIC=10002
    SET_SPEC_CONST_DEFAULT_VALUE=10003
    IF_CONVERSION=10004
    FREEZE_SPEC_CONST=10005
    INLINE_ENTRY_POINTS_EXHAUSTIVE=10006
    INLINE_ENTRY_POINTS_OPAQUE=10007
    COMBINE_ACCESS_CHAINS=10008
    CONVERT_LOCAL_ACCESS_CHAINS=10009
    REPLACE_DESC_ARRAY_ACCESS_USING_VAR_INDEX=10010
    SPREAD_VOLATILE_SEMANTICS=10011
    DESCRIPTOR_SCALAR_REPLACEMENT=10012
    ELIMINATE_DEAD_CODE_AGGRESSIVE=10013
    ELIMINATE_INSERT_EXTRACT=10014
    ELIMINATE_LOCAL_SINGLE_BLOCK=10015
    ELIMINATE_LOCAL_SINGLE_STORE=10016
    MERGE_BLOCKS=10017
    MERGE_RETURN=10018
    ELIMINATE_DEAD_BRANCHES=10019
    ELIMINATE_DEAD_FUNCTIONS=10020
    ELIMINATE_LOCAL_MULTI_STORE=10021
    ELIMINATE_DEAD_CONST=10022
    ELIMINATE_DEAD_INSERTS=10023
    ELIMINATE_DEAD_VARIABLES=10024
    ELIMINATE_DEAD_MEMBERS=10025
    FOLD_SPEC_CONST_OP_COMPOSITE=10026
    LOOP_UNSWITCH=10027
    SCALAR_REPLACEMENT=10028
    STRENGTH_REDUCTION=10029
    UNIFY_CONST=10030
    FLATTEN_DECORATIONS=10031
    COMPACT_IDS=10032
    CFG_CLEANUP=10033
    LOCAL_REDUNDANCY_ELIMINATION=10034
    LOOP_INVARIANT_CODE_MOTION=10035
    REDUCE_LOAD_SIZE=10036
    REDUNDANCY_ELIMINATION=10037
    PRIVATE_TO_LOCAL=10038
    REMOVE_DUPLICATES=10039
    WORKAROUND_1209=10040
    REPLACE_INVALID_OPCODE=10041
    CONVERT_RELAXED_TO_HALF=10042
    RELAX_FLOAT_OPS=10043
    INST_DEBUG_PRINTF=10044
    SIMPLIFY_INSTRUCTIONS=10045
    SSA_REWRITE=10046
    COPY_PROPAGATE_ARRAYS=10047
    LOOP_FISSION=10048
    LOOP_FUSION=10049
    LOOP_UNROLL=10050
    UPGRADE_MEMORY_MODEL=10051
    VECTOR_DCE=10052
    LOOP_UNROLL_PARTIAL=10053
    LOOP_PEELING=10054  # not ?????
    LOOP_PEELING_THRESHOLD=10055
    CCP=10056
    CODE_SINK=10057
    FIX_STORAGE_CLASS=10058
    LEGALIZE_HLSL=10059
    REMOVE_UNUSED_INTERFACE_VARIABLES=10060
    GRAPHICS_ROBUST_ACCESS=10061
    WRAP_OPKILL=10062
    AMD_EXT_TO_KHR=10063
    INTERPOLATE_FIXUP=10064
    REMOVE_DONT_INLINE=10065
    ELIMINATE_DEAD_INPUT_COMPONENTS=10066
    FIX_FUNC_CALL_PARAM=10067
    CONVERT_TO_SAMPLED_IMAGE=10068
    
    O = 20000
    OS = 20001
    INST_BINDLESS_CHECK=20002
    INST_DESC_IDX_CHECK=20003
    INST_BUFF_OOB_CHECK=20004
    INST_BUFF_ADDR_CHECK=20005


class MeasurementType(IntEnum):
    DEFAULT = 0

class ErrorType(IntEnum):
    NONE = 0
    SHADER_COMPILATION_FAILED = 1
    INCONSISTENT_RUN = 2
    TIMEOUT_OR_CRASH = 3
    AUGMENTED_BUT_SAME = 4
    AUGMENTED_BUT_NOT_RUN = 5
    OTHER = 100

# Additional index used (should migrate manually if not)
# CREATE INDEX "imageonlyexperiment_shader_shadertoy_id" ON "imageonlyexperiment" ("shader_shadertoy_id")
# CREATE INDEX "imageonlyshader_shader_id" ON "imageonlyshader" ("shader_id")

class ImageOnlyShader(BaseModel):
    shader_id = pw.CharField(index=True)
    # Store as spirv blob for more accurate representation
    fragment_spv = pw.BlobField()

class ImageOnlyResource(BaseModel):
    """Uses ctype to pack"""
    uniform_block = pw.BlobField()

# NOTE: SQLite has limited ALTER TABLE support
# so the most conventient way is to dump data as sql
# with alias support, and then import into the new database.
class ImageOnlyTrace(BaseModel):
    # JSON serialized basic block idx -> trace idx map
    bb_idx_map = pw.TextField()
    # JSON serialized basic block trace counters
    bb_trace_counters = pw.TextField()
    # the instrumented one
    traced_fragment_spv = pw.BlobField()

class ImageOnlyExperiment(BaseModel):
    time = pw.DateTimeField()
    environment = pw.ForeignKeyField(Environment, backref='experiments')
    augmentation = pw.IntegerField()
    augmentation_annotation = pw.CharField(null=True)
    width = pw.IntegerField()
    height = pw.IntegerField()

    # source
    shader_shadertoy_id = pw.CharField(index=True)
    shader = pw.ForeignKeyField(ImageOnlyShader, backref='experiments', null=True)
    resource = pw.ForeignKeyField(ImageOnlyResource, backref='experiments', null=True)
    trace = pw.ForeignKeyField(ImageOnlyTrace, backref='experiments', null=True)
    parent = pw.ForeignKeyField('self', backref='children', null=True)

    # result
    measurement = pw.IntegerField()
    # help="SHA-256 sum of raw image content"
    image_hash = pw.CharField(null=True)
    num_cycles = pw.IntegerField()
    num_trials = pw.IntegerField()

    errors = pw.IntegerField()

    # mean, stdev
    # JSON serialized time for each trial
    results = pw.TextField()

    # Exported calculations - TODO
    # result_mean = pw.FloatField()
    # result_stdev = pw.FloatField()


def packSpvToBytesStream(spvArray: List[str]):
    """spvArray: List[str], corresponding to std::vector<char>"""
    dataBytes = bytes(map(lambda x: ord(x), spvArray))
    return io.BytesIO(dataBytes)

def packSpvToBytes(spvArray: List[str]):
    return bytes(map(lambda x: ord(x), spvArray))

# Do the reverse of the above
def packBytesToSpv(inputSpv: bytes):
    return [i for i in map(lambda x: chr(x), inputSpv)]

def packBytesToBytesStream(dataBytes: bytes):
    return io.BytesIO(dataBytes)

def getCanonicalImageOnlyExperiments(shaderId=None, nonTraced=False, environmentId=None):
    args = [
        ImageOnlyExperiment.num_cycles == CANONICAL_NUM_CYCLES,
        ImageOnlyExperiment.num_trials == CANONICAL_NUM_TRIALS,
        ImageOnlyExperiment.width == CANONICAL_WIDTH,
        ImageOnlyExperiment.height == CANONICAL_HEIGHT,
        ImageOnlyExperiment.errors == ErrorType.NONE
    ]

    if shaderId is not None:
        args.append(ImageOnlyShader.shader_id == shaderId)
    
    if nonTraced:
        args.append(ImageOnlyExperiment.trace.is_null(True))

    if environmentId is not None:
        args.append(ImageOnlyExperiment.environment == environmentId)

    return ImageOnlyExperiment.select().where(*args).join(ImageOnlyShader)

def getSuccessfulImageOnlyExperiments(shaderId=None, nonTraced=False, environmentId=None):
    args = [
        ImageOnlyExperiment.errors == ErrorType.NONE
    ]

    if shaderId is not None:
        args.append(ImageOnlyShader.shader_id == shaderId)
    
    if nonTraced:
        args.append(ImageOnlyExperiment.trace.is_null(True))

    if environmentId is not None:
        args.append(ImageOnlyExperiment.environment == environmentId)

    return ImageOnlyExperiment.select().where(*args).join(ImageOnlyShader)


def unpackUniformResource(uniform: 'bytes') -> 'Dict[str, Any]':
    """
    struct ImageUniformBlock {
        vec3 iResolution;
        float iTime;
        // NOTE: this is float iChannelTime[4] inside shader block
        // doing this is for the ease of serialization
        vec4 iChannelTime;
        vec4 iMouse;
        vec4 iDate;
        float iSampleRate;
        std::array<vec4, 4> iChannelResolution;    // for padding purpose
        int iFrame;
        float iTimeDelta;
        float iFrameRate;
    };
    """
    format_string = '<3f f 4f 4f 4f f 16f i f f'
    assert(struct.calcsize(format_string) == 144)

    unpacked = struct.unpack(format_string, uniform)
    # print(unpacked)

    return {
        "iResolution": unpacked[:3],
        "iTime": unpacked[3],
        "iChannelTime": unpacked[4:8],
        "iMouse": unpacked[8:12],
        "iDate": unpacked[12:16],
        "iSampleRate": unpacked[16],
        "iChannelResolution": (
            unpacked[17:21],
            unpacked[21:25],
            unpacked[25:29],
            unpacked[29:33]
        ),
        "iFrame": unpacked[33],
        "iTimeDelta": unpacked[34],
        "iFrameRate": unpacked[35]
    }


def getAugmentedExprsForTrace(shaderId=None, nonTraced=False, environmentId=None):
    args = [
        ImageOnlyExperiment.errors == ErrorType.AUGMENTED_BUT_NOT_RUN
    ]

    if shaderId is not None:
        args.append(ImageOnlyShader.shader_id == shaderId)
    if nonTraced:
        args.append(ImageOnlyExperiment.trace.is_null(True))
    if environmentId is not None:
        args.append(ImageOnlyExperiment.environment == environmentId)

    return ImageOnlyExperiment.select().where(*args).join(ImageOnlyShader)


def getAugmentedImageOnlyExperiments(augmentationType=None, shaderId=None, environmentId=None, limit=1000):
    args = [
        ImageOnlyExperiment.errors == ErrorType.NONE,
    ]

    if augmentationType is not None:
        args.append(ImageOnlyExperiment.augmentation == augmentationType)
    if shaderId is not None:
        args.append(ImageOnlyShader.shader_id == shaderId)
    if environmentId is not None:
        args.append(ImageOnlyExperiment.environment == environmentId)

    return ImageOnlyExperiment.select().where(*args).join(ImageOnlyShader).order_by(db.random()).limit(limit)

def getAugmentedButNotRunExperiments(augmentationType=None, shaderId=None, environmentId=None):
    args = [
        ImageOnlyExperiment.errors == ErrorType.AUGMENTED_BUT_NOT_RUN,
    ]

    if augmentationType is not None:
        args.append(ImageOnlyExperiment.augmentation == augmentationType)
    if shaderId is not None:
        args.append(ImageOnlyShader.shader_id == shaderId)
    if environmentId is not None:
        args.append(ImageOnlyExperiment.environment == environmentId)

    return ImageOnlyExperiment.select().where(*args).join(ImageOnlyShader)