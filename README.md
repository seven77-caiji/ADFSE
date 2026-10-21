# ADFSE 实验入口

基于模糊结构编码的自监督异常检测方法（ADFSE），论文标题为 **Self-Supervised Anomaly Detection with Fuzzy Structure Encoding**。本目录是论文的可复现提交包：数据、代码、预训练权重与配置。

## 复现（只加载预训练权重）

```bash
python reproduce.py
```

按 `config.yml` 的 24 个数据集 × 五个种子（42–46）加载 `Weights/<数据集>/seed_<种子>/model.pth`，重新计算 AUROC / AUCPR，打印各数据集及宏平均的「均值 ± 标准差」。不训练、不重新搜索簇数。

```bash
python reproduce.py --dataset Wine
python reproduce.py --dataset Wine Lymphography --seeds 42-46
```

`--dataset` 可给多个名字，`--seeds` 支持 `42-46` 或 `42,43`。

## 从头训练

```bash
python main.py
```

按同一套公共超参数重训全部 24 个数据集 × 五个种子，权重写回 `Weights/<数据集>/seed_<种子>/model.pth`。

公共参数：隐藏层 `[128, 128]`、潜在维度 16、学习率 0.01、batch size 128、50 epochs、结构编码重构损失权重 1.0、模糊指数 1.3、weight decay 0；所有数据集统一簇数 `c=16`。FCM 最大迭代次数 4000，容差 0.0001。

代码中的 `FCMPosition`、`PositionAE` 和 `position_*` 仍指上述模糊结构编码；这些内部名称保留用于兼容现有配置与预训练权重。

每个种子随机划分正常样本：一半用于训练，剩余正常样本与全部异常样本用于测试；归一化和 FCM 只拟合正常训练集。异常分数为特征重构 MSE（分数越高越异常），AUCPR 用 average precision，宏平均给各数据集相同权重。

## 文件结构

```text
ADFSE_submit/
  main.py          # 公共超参数、五种子，从头训练
  reproduce.py     # 只加载预训练权重并复现分数
  config.yml       # 数据集、种子、公共参数、统一簇数 c
  utils.py
  requirements.txt
  Model/           # 网络、FCM、训练与评分
  Data/            # 论文实验保留的 24 个数据文件
  Weights/<数据集>/seed_<种子>/model.pth
```

使用已有研究 Python 环境（当前为 `D:\anaconda3\python.exe`）；依赖版本见 `requirements.txt`，设备默认为有 CUDA 则用 CUDA，否则 CPU。
