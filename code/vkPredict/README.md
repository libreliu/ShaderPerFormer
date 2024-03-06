# vkPredict

## Prerequisite

- pip install numpy torch torchinfo transformers peewee pillow matplotlib joblib tokenizers requests

## Lock frequency

### NVIDIA

NVIDIA [Ref](https://developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate/):
- Query supported: `nvidia-smi --query-supported-clocks=timestamp,gpu_name,gpu_uuid,memory,graphics --format=csv`
- Enter persistence mode:
  - `sudo nvidia-smi -pm 1`
- Set:
  - Core CLK: `sudo nvidia-smi --lock-gpu-clocks=<core_clock_rate>`
  - Memory CLK: `sudo nvidia-smi --lock-memory-clocks=<memory_clock_rate>`
- Reset:
  - `sudo nvidia-smi --reset-gpu-clocks`
  - `sudo nvidia-smi --reset-memory-clocks`

### AMD

https://wiki.archlinux.org/title/AMDGPU

https://github.com/sibradzic/amdgpu-clocks

### fairseq

> NOTE: fairseq is no longer used. I personally thinks it powerful but hard to use.
> Especially considering the amount of work needed to make it useful for my customized
> output and loss to be working. 
>
> See `test-fairseq` branch for a still-work-in-progress sum task implementation to gain
> potential fairseq knowledge.

## Commands

- Run testcases: `python -m unittest discover -s test`

python compete.py traced-linear-regression --num-features 26 --exclude-first --save-path tracedLinearRegression-26feature-exclude-first.json

python compete.py traced-per-inst-linear-regression  --load-path tracedPerInstLinearRegression.json

python misc.py dataset snapshot-traced --train-ratio 0.5 --dest-file TestTraceDataset.dat

python -mpickle TestTraceDataset.dat

python -m unittest discover -s test -k testTrace.test_shift_params

## Tips
1. How to pad the sequence: https://www.codefull.net/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/


## Dataset notes

1. "Raw shadertoy": 14072
2. `FragPerfSnapShotDataset.json`: (truncuated to 4096 token) 11282, 80.17% of the original, train 9025, test 2257
3. `FragPerfSnapshotDataset1024.json` (truncuated to 1024 token) 3769, 26.79% of original, train 3015, test 754
4. `FragPerfSnapshotTracedDataset.dat`: (not including some error-ed shaders) 14017, 99.61% of the original, train 11213, test 2804
5. `FragPerfSnapshotTracedDataset4096-Correct.dat`: (not including some too-long inlined shaders) 8900, 63.494% of the original, train: 7120, test: 1780


```
(venv) PS C:\Projects\NGPP\vkPredict> python misc.py dataset --snapshot --snapshot-max-tokenized-length 1024 --snapshot-dest-file FragPerfSnapShotDataset1024.json
Filter Processing: 14072it [22:41, 10.34it/s]
2023-08-13 09:48:38,184 -       dataset.FragmentPerformanceDataset - INFO - Raw: 14072; Filtered: 3769 (26.784% of the original;using maxLen=1024)
2023-08-13 09:48:38,185 -             misc.snapshotFragPerfDataset - INFO - Total: 3769, train: 3015, test: 754
Training sample serialization: 100%|████████████████████████████████████████████████████████████████| 3015/3015 [00:00<00:00, 215233.46it/s]
Testing sample serialization: 100%|███████████████████████████████████████████████████████████████████| 754/754 [00:00<00:00, 188020.52it/s]
2023-08-13 09:48:38,483 -             misc.snapshotFragPerfDataset - INFO - Snapshot written to FragPerfSnapShotDataset1024.json
(venv) PS C:\Projects\NGPP\vkPredict>
```

失败 trace

id=4lGSDw

### Trace with inline failed samples

这里发现一个问题，就是 inline 的话绝大多数的程序都超级长。

应该还是要改成 multiple entrypoint 的形式。

### Correct one

之前有 bug

```
(venv) [libreliu@Legion-R7k vkPredict]$ python misc.py dataset snapshot-traced --dest-file FragPerfSnapshotTracedDataset4096-Correct.dat --max-tokenized-length 4096
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 14017/14017 [25:31<00:00,  9.15it/s]
Failed samples (total 229): [22, 35, 87, 128, 136, 176, 207, 232, 272, 283, 407, 452, 487, 540, 562, 588, 596, 642, 667, 676, 701, 803, 843, 892, 946, 981, 997, 1131, 1246, 1280, 1297, 1403, 1407, 1621, 1834, 1876, 1885, 1945, 1987, 2040, 2112, 2350, 2393, 2612, 2633, 2702, 2729, 2730, 2829, 2845, 2904, 2922, 3031, 3146, 3180, 3236, 3460, 3463, 3501, 3530, 3570, 3571, 3619, 3625, 3713, 3719, 3741, 3780, 3787, 3838, 3951, 3990, 4016, 4035, 4054, 4057, 4150, 4175, 4273, 4280, 4314, 4347, 4374, 4496, 4693, 4813, 4868, 5163, 5178, 5217, 5224, 5308, 5375, 5440, 5465, 5532, 5534, 5614, 5628, 5758, 5785, 5864, 6008, 6080, 6096, 6114, 6142, 6258, 6450, 6519, 6546, 6654, 6771, 6772, 6875, 6886, 7057, 7084, 7094, 7109, 7121, 7171, 7222, 7265, 7277, 7317, 7453, 7507, 7531, 7613, 7906, 7954, 7986, 8065, 8183, 8336, 8344, 8450, 8590, 8622, 8761, 8805, 8816, 8873, 8909, 8982, 9090, 9092, 9098, 9124, 9126, 9188, 9216, 9361, 9414, 9633, 9644, 9782, 9811, 9859, 9861, 9894, 9899, 9939, 9959, 9962, 10008, 10034, 10132, 10159, 10169, 10233, 10408, 10412, 10430, 10463, 10526, 10594, 10623, 10679, 10684, 10699, 10712, 10752, 10845, 10851, 10883, 10893, 11002, 11026, 11070, 11080, 11190, 11227, 11283, 11327, 11338, 11395, 11409, 11523, 11531, 11632, 11645, 11852, 11918, 12049, 12060, 12106, 12127, 12132, 12191, 12196, 12242, 12252, 12278, 12396, 12401, 12476, 12542, 12895, 12971, 12982, 13005, 13013, 13154, 13496, 13501, 13591, 13955]
2023-10-16 17:40:03,821 -        misc.snapshotFragPerfTraceDataset - INFO - Total: 14017, filtered: 8900, (63.494% of the original) train: 7120, test: 1780
Training sample serialization: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7120/7120 [00:00<00:00, 20115.66it/s]
Testing sample serialization: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1780/1780 [00:00<00:00, 18785.11it/s]
2023-10-16 17:40:04,451 -        misc.snapshotFragPerfTraceDataset - INFO - Snapshot written to FragPerfSnapshotTracedDataset4096-Correct.dat
(venv) [libreliu@Legion-R7k vkPredict]$ 
```

### Windows shader directory

use `fsutil file setCaseSensitiveInfo <path> enable` to enable the case sensitivity of shader directory
