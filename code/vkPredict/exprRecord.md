## 实验情况

本仓库主要包含下面的实验：

1. 基线
   1. 线性回归
      1. 无 Trace 线性回归 - (1, 3, 26, 所有指令类别)
      2. 有 Trace 线性回归 - (1, 3, 26, 所有指令类别)
   2. 无 Trace Transformer，使用 `roberta-base-layer9`
2. bpe Tokenizer vs spv Tokenizer
   > TODO: implement me
3. 探究预训练方法对于预测任务的作用
   1. 探究 MLM 对于预测任务的作用
      - MLM 权重分别在 0.3, 0.5, 0.7 的情况下，导出预训练模型，然后在下游的
        - 含 Trace
        - 不含 Trace

        网络中进行测试
   2. 探究对比学习对于预测任务的作用
4. 探究 Trace 对于预测任务的作用
5. 探究骨干网络结构对于预测任务的作用
   1. Roberta-Base vs Roberta-Base-layer9
   2. Roberta-Large?
6. 多机数据集挖掘
   1. 分组进行采样，选择 1000 个有代表性的 shader，在各组抽样来做这些“其它实验”
      - 否则全跑的话时间太长了
   2. 探究不同机器上同一个 Shader 的性能比例
   3. 探究 Shader 性能稳定性
      1. numCycles = 1, 2, 5, 8, 10, 20, 30
   4. 自适应 Shader 时间测量策略研究
   5. 快速 Finetune 能力研究

### 基线

#### 线性回归



#### 无 Trace Transformer


### 预训练

#### MLM 探索



## 数据归一化策略

在各组神经网络实验中，数据归一化质量对 Transformer 本身的学习有一定影响。

确保在每组实验中，都能应用不同的数据归一化方法，并且对学习的结果进行对比。

时间归一化策略：
1. Scale
2. logPlus

Trace 归一化策略：
- 除以大数？
- 分组归一化？

> 只需要在无 Trace 场景和有 Trace 场景各探索最好的，然后之后 MLM 等场景直接用这个最好的就行了。

## Metric

在除量化阶过小的 Shader 之外的 Shader 运行时间均近似服从正态分布（虽然 $ \sigma $ 随 $ \mu $ 增长而有一定增长）。

最小的运行时间在 $ 10^{-5} $ 级别，最大的运行时间在 $ 10^{-1} $ 级别，故分组进行预测准确率统计。

分为如下几组：
- `time >= 1e-1 sec` (也就是 `fps <= 10`)
- `1e-2 <= time < 1e-1` (也就是 `10 <= fps <= 100`)
- `1e-3 <= time < 1e-2` (也就是 `100 <= fps <= 1000`)
- `1e-4 <= time < 1e-3` (也就是 `1000 <= fps <= 10000`)
- `time < 1e-4` (也就是 `10000 < fps`)

每组内回报关于时间这一指标的 MSE，MAE 和 MAPE。

> 时间更有参考价值，正态分布的和是正态分布，但是 fps 和 fps 的和都不是正态分布，这在预测器之后的使用上创造了很大的障碍。

训练集和测试集 metric 同时记录在案。

## 数据集情况

### Old world

1. "Raw shadertoy": 14072
2. `FragPerfSnapShotDataset.json`: (truncuated to 4096 token) 11282, 80.17% of the original, train 9025, test 2257
3. `FragPerfSnapshotDataset1024.json` (truncuated to 1024 token) 3769, 26.79% of original, train 3015, test 754
4. `FragPerfSnapshotTracedDataset.dat`: (not including some error-ed shaders) 14017, 99.61% of the original, train 11213, test 2804
5. `FragPerfSnapshotTracedDataset4096-Correct.dat`: (not including some too-long inlined shaders) 8900, 63.494% of the original, train: 7120, test: 1780
6. `FragTokenizedDataset.json`

### New world

