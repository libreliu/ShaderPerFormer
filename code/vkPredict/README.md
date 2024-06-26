# vkPredict

## Prerequisite

- pip install numpy torch torchinfo transformers peewee pillow matplotlib joblib tokenizers requests

## Commands

### Training

Example command used to train our model is as below:

```bash
# This command trains the model `Perfformer-NoRope-Trace-Onehot-Base2-Regression-3060-ConstTrace-Val-TimeFiltered-fp16-log-time`
# using the dataset provided in ${GEMINI_DATA_IN1}/intermediates/FragPerfSnapshotTracedFinalDataset-RTX3060-Val-TimeFiltered.dat

python train.py --dataset-root-dir-override ${GEMINI_DATA_IN1}/ \
                --output-dir-prefix ${GEMINI_DATA_OUT}/Perfformer-NoRope-Trace-Onehot-Base2-Regression-3060-ConstTrace-Val-TimeFiltered-fp16-log-time \
                --num-epochs 50 --per-device-batch-size 2 --learning-rate 3e-5 --gradient-accumulation-steps 20 \
                --dataset FragPerfSnapshotTracedFinalDataset-RTX3060-Val-TimeFiltered --label-normalizer log-normalizer-time \
                --trace-normalizer dummy-time --post-processor TraceDatasetAsConstTracedPostProcessor \
                --model perfformer-layer9-regression-trace-input-embedding-xformers-memeff --pad-to 8 \
                --collator-trace-embedding onehot-base2 --tokenizer HfTracedSpvTokenizer-multiple-entrypoint \
                --use-fp16 --load-best-model-at-end train
```

We use the xformers' implementation of memory efficient attention in our code. Be sure to read on potential GPU limitations if you choose to train on platforms other than what we use (which is NVIDIA Geforce RTX 3090).

### Misc.

Some testcases are provided and can be run with `python -m unittest discover -s test`.

Also, there are some baseline methods (but they have been migrated to Jupyter notebooks in the end, so these are left as a reminder. There's no guarantee that these will work.)

```bash
python compete.py traced-linear-regression --num-features 26 --exclude-first --save-path tracedLinearRegression-26feature-exclude-first.json
python compete.py traced-per-inst-linear-regression  --load-path tracedPerInstLinearRegression.json
python misc.py dataset snapshot-traced --train-ratio 0.5 --dest-file TestTraceDataset.dat
python -mpickle TestTraceDataset.dat
python -m unittest discover -s test -k testTrace.test_shift_params
```

## Tips

1. How to pad the sequence: https://www.codefull.net/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/
