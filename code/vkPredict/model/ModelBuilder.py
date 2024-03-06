import transformers
from .custom_roberta import (
    RobertaForSequenceClassificationFast,
    RobertaForSequenceRegressionMAPE,
    RobertaRegressionModelFast,
    RobertaTracedMultFast,
    RobertaTracedMultWithAffineTransformFast,
    RobertaForMaskedLMFast,
    RobertaFastConfig
)
from .configuration_perfformer import (
    PerfformerConfig
)
from .modeling_perfformer import (
    PerfformerForMaskedLM,
    PerfformerForRegression,
    PerfformerForMultiTaskLearning,
    PerfformerForMultiArchLearning,
    PerfformerForMoCo
)

import os
import logging

logger = logging.getLogger(__name__)

def build_model(modelName, lossName, maxSeqLength, bosTokenId, eosTokenId, padTokenId, loadDir=None):
    """Possible configurations:
    
    roberta-base-regression-vocab40000
    roberta-base-regression-mape-vocab40000
    """
    model = None

    useLoadDir = loadDir is not None and os.path.exists(loadDir)

    if modelName == "roberta-base-regression-vocab40000":
        assert(lossName == "mse")
        config = transformers.RobertaConfig.from_pretrained("roberta-base")
        config.num_labels = 1
        config.problem_type = "regression"
        config.vocab_size = 40000
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId

        if useLoadDir:
            # only load the weights
            model = transformers.RobertaForSequenceClassification.from_pretrained(loadDir, config=config)
        else:
            model = transformers.RobertaForSequenceClassification(config)

    elif modelName == "roberta-base-layer9-regression-vocab40000":
        config = transformers.RobertaConfig.from_pretrained("roberta-base")
        assert(lossName == "mse")

        config.num_labels = 1
        config.problem_type = "regression"
        config.vocab_size = 40000
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9

        if useLoadDir:
            # only load the weights
            model = transformers.RobertaForSequenceClassification.from_pretrained(loadDir, config=config)
        else:
            model = transformers.RobertaForSequenceClassification(config)
    elif modelName == "roberta-base-layer9-maskedlm-vocab40000":
        config = transformers.RobertaConfig.from_pretrained("roberta-base")
        assert(lossName == "mse")

        config.vocab_size = 40000
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9

        if useLoadDir:
            model = transformers.RobertaForMaskedLM.from_pretrained(loadDir, config=config)
        else:
            model = transformers.RobertaForMaskedLM(config)
    elif modelName.startswith("roberta-fast-base-layer9-maskedlm-vocab40000"):
        config = transformers.RobertaConfig.from_pretrained("roberta-base")
        assert(lossName == "mse")

        config.vocab_size = 40000
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.attention_type = modelName[len("roberta-fast-base-layer9-maskedlm-vocab40000-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            model = RobertaForMaskedLMFast.from_pretrained(loadDir, config=config)
        else:
            model = RobertaForMaskedLMFast(config)
    elif modelName.startswith("roberta-fast-base-layer9-regression-vocab40000"):
        config = RobertaFastConfig.from_pretrained("roberta-base")
        assert(lossName == "mse" or lossName == "mape")

        config.num_labels = 1
        config.problem_type = "regression-" + lossName
        config.vocab_size = 40000
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.attention_type = modelName[len("roberta-fast-base-layer9-regression-vocab40000-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = RobertaForSequenceClassificationFast.from_pretrained(loadDir, config=config)
        else:
            model = RobertaForSequenceClassificationFast(config)

    elif modelName.startswith("roberta-fast-base-layer9-traced-regression-vocab40000"):
        config = RobertaFastConfig.from_pretrained("roberta-base")
        assert(lossName in ("mse", "mape"))

        config.num_labels = 1
        config.problem_type = "regression-" + lossName
        config.vocab_size = 40000
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.attention_type = modelName[len("roberta-fast-base-layer9-traced-regression-vocab40000-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = RobertaTracedMultFast.from_pretrained(loadDir, config=config)
        else:
            model = RobertaTracedMultFast(config)
    elif modelName.startswith("roberta-fast-base-layer9-traced-with-affine-regression-vocab40000"):
        config = RobertaFastConfig.from_pretrained("roberta-base")
        assert(lossName in ("mse", "mape"))

        config.num_labels = 1
        config.problem_type = "regression-" + lossName
        config.vocab_size = 40000
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.attention_type = modelName[len("roberta-fast-base-layer9-traced-with-affine-regression-vocab40000-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = RobertaTracedMultWithAffineTransformFast.from_pretrained(loadDir, config=config)
        else:
            model = RobertaTracedMultWithAffineTransformFast(config)
    elif modelName.startswith("roberta-fast-base-layer9-regression-large-head-vocab40000"):
        config = RobertaFastConfig.from_pretrained("roberta-base")
        assert(lossName in ("mse", "mape"))

        config.num_labels = 1
        config.problem_type = "regression-" + lossName
        config.vocab_size = 40000
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.attention_type = modelName[len("roberta-fast-base-layer9-regression-large-head-vocab40000-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = RobertaRegressionModelFast.from_pretrained(loadDir, config=config)
        else:
            model = RobertaRegressionModelFast(config)
    elif modelName.startswith("perfformer-layer9-maskedlm"):
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.attention_type = modelName[len("perfformer-layer9-maskedlm-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForMaskedLM.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForMaskedLM(config)
    elif modelName.startswith("perfformer-layer9-regression-seq-sum-reduction-trace-input-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.trace_label_embedding_type = 'input'
        config.problem_type = "regression-" + lossName
        config.attention_type = modelName[len("perfformer-layer9-regression-seq-sum-reduction-trace-input-embedding-"):]
        config.regression_head_configuration = "seq-sum-reduction"

        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForRegression.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForRegression(config)
    elif modelName.startswith("perfformer-layer9-regression-trace-input-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.trace_label_embedding_type = 'input'
        config.problem_type = "regression-" + lossName
        config.attention_type = modelName[len("perfformer-layer9-regression-trace-input-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForRegression.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForRegression(config)
    elif modelName.startswith("perfformer-layer6-regression-trace-input-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 6
        config.trace_label_embedding_type = 'input'
        config.problem_type = "regression-" + lossName
        config.attention_type = modelName[len("perfformer-layer6-regression-trace-input-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForRegression.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForRegression(config)
    elif modelName.startswith("perfformer-layer3-regression-trace-input-embedding"):

        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 3
        config.trace_label_embedding_type = 'input'
        config.problem_type = "regression-" + lossName
        config.attention_type = modelName[len("perfformer-layer3-regression-trace-input-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForRegression.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForRegression(config)
    elif modelName.startswith("perfformer-layer12-regression-trace-input-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 12
        config.trace_label_embedding_type = 'input'
        config.problem_type = "regression-" + lossName
        config.attention_type = modelName[len("perfformer-layer12-regression-trace-input-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForRegression.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForRegression(config)
    elif modelName.startswith("perfformer-layer9-regression-trace-binary-learnable-37-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.trace_label_embedding_type = 'binary-learnable'
        config.problem_type = "regression-" + lossName
        config.trace_label_binary_embedding_max_length = 37
        config.attention_type = modelName[len("perfformer-layer9-regression-trace-binary-learnable-37-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForRegression.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForRegression(config)
    elif modelName.startswith("perfformer-layer9-regression-trace-binary-learnable-33-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.trace_label_embedding_type = 'binary-learnable'
        config.problem_type = "regression-" + lossName
        config.trace_label_binary_embedding_max_length = 33
        config.attention_type = modelName[len("perfformer-layer9-regression-trace-binary-learnable-33-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForRegression.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForRegression(config)
    elif modelName.startswith("perfformer-layer9-regression-trace-binary-learnable-30-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.trace_label_embedding_type = 'binary-learnable'
        config.problem_type = "regression-" + lossName
        config.trace_label_binary_embedding_max_length = 30
        config.attention_type = modelName[len("perfformer-layer9-regression-trace-binary-learnable-30-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForRegression.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForRegression(config)
    elif modelName.startswith("perfformer-layer9-regression"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.trace_label_embedding_type = 'none'
        config.problem_type = "regression-" + lossName
        config.attention_type = modelName[len("perfformer-layer9-regression-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForRegression.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForRegression(config)

    elif modelName.startswith("perfformer-layer9-mal-2-arch-trace-input-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.mal_num_lm_heads = 2
        config.trace_label_embedding_type = 'input'
        config.problem_type = "regression-" + lossName
        config.attention_type = modelName[len("perfformer-layer9-mal-2-arch-trace-input-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForMultiArchLearning.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForMultiArchLearning(config)

    elif modelName.startswith("perfformer-layer9-mal-3-arch-trace-input-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.mal_num_lm_heads = 3
        config.trace_label_embedding_type = 'input'
        config.problem_type = "regression-" + lossName
        config.attention_type = modelName[len("perfformer-layer9-mal-3-arch-trace-input-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForMultiArchLearning.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForMultiArchLearning(config)
    elif modelName.startswith("perfformer-layer9-mal-4-arch-trace-input-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.mal_num_lm_heads = 4
        config.trace_label_embedding_type = 'input'
        config.problem_type = "regression-" + lossName
        config.attention_type = modelName[len("perfformer-layer9-mal-4-arch-trace-input-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForMultiArchLearning.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForMultiArchLearning(config)

    elif modelName.startswith("perfformer-layer9-mtl-mlm-0.3-reg-0.7-trace-input-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.trace_label_embedding_type = 'input'
        config.problem_type = "regression-" + lossName
        config.mtl_mlm_loss_weight = 0.3
        config.mtl_regression_loss_weight = 0.7
        config.attention_type = modelName[len("perfformer-layer9-mtl-mlm-0.3-reg-0.7-trace-input-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForMultiTaskLearning.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForMultiTaskLearning(config)

    elif modelName.startswith("perfformer-layer9-mtl-mlm-0.7-reg-0.3-trace-input-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.num_hidden_layers = 9
        config.trace_label_embedding_type = 'input'
        config.problem_type = "regression-" + lossName
        config.mtl_mlm_loss_weight = 0.7
        config.mtl_regression_loss_weight = 0.3
        config.attention_type = modelName[len("perfformer-layer9-mtl-mlm-0.7-reg-0.3-trace-input-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForMultiTaskLearning.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForMultiTaskLearning(config)

    # = REAL ROPE = !!
    elif modelName.startswith("perfformer-rope-layer9-regression-trace-input-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.position_embedding_type = 'rope'
        config.num_hidden_layers = 9
        config.trace_label_embedding_type = 'input'
        config.problem_type = "regression-" + lossName
        config.attention_type = modelName[len("perfformer-rope-layer9-regression-trace-input-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForRegression.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForRegression(config)


    elif modelName.startswith("perfformer-rope-layer9-mtl-mlm-0.3-reg-0.7-trace-input-embedding"):
        
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.pad_token_id = padTokenId
        config.position_embedding_type = 'rope'
        config.num_hidden_layers = 9
        config.trace_label_embedding_type = 'input'
        config.problem_type = "regression-" + lossName
        config.mtl_mlm_loss_weight = 0.3
        config.mtl_regression_loss_weight = 0.7
        config.attention_type = modelName[len("perfformer-rope-layer9-mtl-mlm-0.3-reg-0.7-trace-input-embedding-"):]
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForMultiTaskLearning.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForMultiTaskLearning(config)
    

    elif modelName.startswith("perfformer-rope-layer9-moco-m-0.999-T-0.07-trace-input-emmbedding"):
        config = PerfformerConfig()
        config.position_embedding_type = 'rope'
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.num_hidden_layers = 9
        config.trace_label_embedding_type = 'input'
        # config.problem_type = "regression-" + lossName
        config.attention_type = modelName[len("perfformer-rope-layer9-moco-m-0.999-T-0.07-trace-input-emmbedding-"):]
        config.moco_K = 4096
        config.moco_m = 0.999
        config.moco_T = 0.07
        config.moco_symmetric = False
        config.moco_shuffle_batch = True
        config.moco_use_bn = False
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForMoCo.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForMoCo(config)
    

    elif modelName.startswith("perfformer-layer9-moco-m-0.999-T-0.07-trace-input-emmbedding"):
        config = PerfformerConfig()
        config.max_position_embeddings = maxSeqLength
        config.bos_token_id = bosTokenId
        config.eos_token_id = eosTokenId
        config.num_hidden_layers = 9
        config.trace_label_embedding_type = 'input'
        # config.problem_type = "regression-" + lossName
        config.attention_type = modelName[len("perfformer-layer9-moco-m-0.999-T-0.07-trace-input-emmbedding-"):]
        config.moco_K = 4096
        config.moco_m = 0.999
        config.moco_T = 0.07
        config.moco_symmetric = False
        config.moco_shuffle_batch = True
        config.moco_use_bn = False
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0
        logger.info(f"Use attention type {config.attention_type}")

        if useLoadDir:
            # only load the weights
            model = PerfformerForMoCo.from_pretrained(loadDir, config=config)
        else:
            model = PerfformerForMoCo(config)
    
    
    else:
        assert(False)

    return model