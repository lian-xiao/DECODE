# 虚拟筛选任务框架

本框架专门用于药物分子的虚拟筛选任务，基于预训练的MultiModalMOAPredictor模型生成聚合特征进行分类。

## 核心特性

### 🧬 基于MOA模型的特征生成
- 加载预训练的MultiModalMOAPredictor模型
- 使用"both_missing"场景：仅从药物特征生成RNA和表型特征
- 通过多个剂量值取平均来处理缺失的剂量信息
- 冻结所有预训练编码器，仅训练新的分类器

### 🎯 多剂量特征平均
- 使用多个常见剂量值（如[0.1, 0.3, 1.0, 3.0, 10.0]）
- 为每个剂量生成MOA特征，然后取平均
- 提供更稳健的特征表示，减少剂量选择的影响

### 🔧 灵活的模型配置
- 支持加载预训练MOA模型或使用虚拟编码器
- 可选择性使用Molformer或其他药物编码器
- 完全可配置的网络结构和训练参数

## 框架结构

```
virtual_screening/
├── __init__.py              # 模块初始化
├── models.py               # 模型定义
├── data.py                 # 数据模块
├── utils.py                # 工具函数
├── train_virtual_screening.py  # 训练脚本
└── config_virtual_screening.yaml  # 配置文件
```

## 模型说明

### 1. MolformerModule
- 基于Molformer的药物分子表征提取模块
- 使用预训练的Molformer模型（如 `ibm/MoLFormer-XL-both-10pct`）
- 仅使用SMILES字符串进行分类

### 2. VirtualScreeningModule  
- 基于预训练MultiModalMOAPredictor模型的虚拟筛选模块
- 使用MOA模型的编码器生成聚合特征（药物+模拟的RNA+模拟的表型）
- 通过多个常见剂量值（如[0.1, 0.3, 1.0, 3.0, 10.0]）取平均来处理缺失的剂量信息
- 冻结所有预训练编码器，仅训练新的分类器
- 支持both_missing场景：仅使用药物信息生成RNA和表型特征

## 数据格式

### 训练数据
CSV文件，必须包含以下列：
- `canonical_smiles` 或 `smiles`: SMILES字符串
- `label`: 二分类标签 (0/1 或 active/inactive)

### 外部验证数据
格式同训练数据，用于最终性能评估

## 重要使用说明

### 🔑 MOA模型路径配置
使用虚拟筛选模型时，**强烈建议**设置MOA模型路径：

```bash
# 设置MOA模型路径
python run_virtual_screening.py \
    --moa_model_path "results/multimodal/final_model.ckpt" \
    --train_data "preprocessed_data/Virtual_screening/EP4/ChEMBL-EP4_processed_ac.csv"
```

### 📊 特征生成原理
1. **药物特征提取**: 使用Molformer从SMILES提取768维特征
2. **多剂量MOA特征生成**: 
   - 对每个剂量值（默认：[0.1, 0.3, 1.0, 3.0, 10.0]）
   - 使用MOA模型的"both_missing"场景生成融合特征
   - 对所有剂量的特征取平均
3. **分类**: 通过新的分类器进行最终预测

### ⚙️ 训练策略
- **冻结编码器**: 所有预训练的编码器权重被冻结
- **仅训练分类器**: 只有新添加的分类器参数可训练
- **快速收敛**: 由于大部分参数固定，训练速度较快

## 使用方法

### 0. 快速测试框架
```bash
# 测试框架是否正常工作
python virtual_screening/test_framework.py

# 运行使用示例
python virtual_screening/example_usage.py
```

### 1. 快速开始（推荐）
```bash
# 使用默认配置训练两个模型并比较
python run_virtual_screening.py

# 指定数据路径
python run_virtual_screening.py \
    --train_data "preprocessed_data/Virtual_screening/EP4/ChEMBL-EP4_processed_ac.csv" \
    --external_val_data "preprocessed_data/Virtual_screening/EP4/ExtVal_EP4_processed_ac.csv" \
    --output_dir "results/ep4_virtual_screening"
```

### 2. 仅训练Molformer基线
```bash
python run_virtual_screening.py --mode molformer_only
```

### 3. 仅训练虚拟筛选模型（需要预训练MOA模型）
```bash
python run_virtual_screening.py \
    --mode vs_only \
    --moa_model_path "path/to/your/moa_model.ckpt"
```

### 4. 直接使用训练脚本
```bash
python virtual_screening/train_virtual_screening.py \
    --train_data "preprocessed_data/Virtual_screening/EP4/ChEMBL-EP4_processed_ac.csv" \
    --external_val_data "preprocessed_data/Virtual_screening/EP4/ExtVal_EP4_processed_ac.csv" \
    --moa_model_path "results/multimodal/final_model.ckpt" \
    --output_dir "results/virtual_screening"
```

## 配置文件

可以通过修改 `virtual_screening/config_virtual_screening.yaml` 来调整：

```yaml
data:
  batch_size: 16
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

molformer:
  model_name: "ibm/MoLFormer-XL-both-10pct"
  learning_rate: 1.0e-4
  
virtual_screening:
  moa_model_path: "path/to/your/moa_model.ckpt"  # 重要：设置MOA模型路径
  hidden_dim: 256
  freeze_encoders: true  # 冻结所有编码器
  dose_values: [0.1, 0.3, 1.0, 3.0, 10.0]  # 多剂量平均

training:
  max_epochs: 100
  patience: 10
```

## 输出结果

训练完成后，结果保存在指定的输出目录中：

```
results/virtual_screening/
├── config.yaml                    # 训练配置
├── model_comparison.yaml          # 模型比较结果
├── molformer_baseline/            # Molformer基线模型
│   ├── checkpoints/               # 模型检查点
│   ├── tensorboard/               # TensorBoard日志
│   ├── external_predictions.csv   # 外部验证预测结果
│   └── final_model.ckpt          # 最终模型
└── virtual_screening/             # 虚拟筛选模型
    ├── checkpoints/
    ├── tensorboard/
    ├── external_predictions.csv
    └── final_model.ckpt
```

## 结果分析

### 1. 查看训练日志
```bash
tensorboard --logdir results/virtual_screening/molformer_baseline/tensorboard
tensorboard --logdir results/virtual_screening/virtual_screening/tensorboard
```

### 2. 比较模型性能
查看 `model_comparison.yaml` 文件了解两个模型的测试性能

### 3. 外部验证结果
查看 `external_predictions.csv` 文件查看在外部验证集上的预测结果

## 依赖安装

```bash
# 核心依赖
pip install torch pytorch-lightning transformers
pip install pandas scikit-learn pyyaml

# Molformer相关（可选，会自动下载）
pip install tokenizers

# 分子处理（可选，用于SMILES验证）
pip install rdkit-pypi
```

## 注意事项

1. **Molformer模型下载**: 首次运行时会自动下载Molformer模型，需要网络连接
2. **GPU使用**: 框架会自动检测并使用可用的GPU
3. **内存使用**: Molformer模型较大，建议至少8GB内存
4. **数据预处理**: 确保SMILES字符串格式正确
5. **MOA模型权重**: 如果有预训练的MOA模型，建议使用以获得更好的性能

## 扩展使用

- 可以修改 `models.py` 中的模型结构
- 可以在 `data.py` 中添加更多数据预处理功能
- 可以在 `utils.py` 中添加更多分子描述符计算功能