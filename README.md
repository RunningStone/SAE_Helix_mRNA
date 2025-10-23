# SAE Helix mRNA

**使用稀疏自编码器 (Sparse Auto-Encoder) 探索 Helix-mRNA 模型的生物学可解释性**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 目录

- [项目简介](#项目简介)
- [核心原理](#核心原理)
- [项目结构](#项目结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [详细用法](#详细用法)
- [测试](#测试)
- [参考文献](#参考文献)

---

## 🎯 项目简介

本项目实现了基于**稀疏自编码器 (SAE)** 的可解释性分析框架，用于探索 **Helix-mRNA** 基础模型的内部表示。通过学习过完备的稀疏特征字典，我们可以：

- 🔍 **发现可解释的特征**：从模型激活中提取生物学上有意义的特征
- 📊 **分析层级表示**：理解不同层如何编码 RNA 序列信息
- 🧬 **生物学洞察**：将学到的特征与已知的生物学概念关联

### 核心特性

✅ **完全非侵入式**：使用 PyTorch Hook 机制，无需修改原始模型代码  
✅ **模块化设计**：SAE 模型、训练器、Pipeline 完全解耦  
✅ **多层分析**：支持同时分析模型的多个层  
✅ **灵活配置**：可自定义过完备比率、稀疏惩罚系数等超参数  
✅ **完整测试**：包含单元测试和集成测试

---

## 🧠 核心原理

### 稀疏自编码器 (SAE)

SAE 通过学习过完备的稀疏表示来发现数据中的潜在特征：

**问题形式化**：从观测到的激活向量集合 `{x_i}` 中恢复特征字典 `{f_k}`，使得每个激活可由少数特征的线性组合重构。

**模型架构**：

```
编码: c = ReLU(Wx + b)        # 稀疏激活
解码: x̂ = W^T c                # 特征重组
```

**损失函数**：

```
L(x) = ||x - x̂||² + α||c||₁
       └─重构损失─┘   └─稀疏惩罚─┘
```

**关键设计**：
- **过完备性**：`d_hidden > d_in`（典型值：4倍）
- **稀疏约束**：L1 惩罚 + ReLU + 负偏置
- **权重绑定**：解码器使用编码器的转置
- **行归一化**：`||f_i||₂ = 1`，防止范数放大

### 工作流程

```
RNA 序列
    ↓
[Helix mRNA 模型]
    ↓
多层激活提取 (Hook)
    ↓
{layer_0: activations, layer_1: activations, ...}
    ↓
训练 SAE (每层独立)
    ↓
特征字典 {f₀, f₁, ..., f_n}
    ↓
可解释性分析
```

---

## 📁 项目结构

```
SAE_Helix_mRNA/
├── src/
│   ├── model/
│   │   ├── sparse_autoencoder.py      # SAE 核心模型
│   │   ├── activation_extractor.py    # 激活值提取器 (Hook)
│   │   ├── sae_trainer.py             # SAE 训练器
│   │   └── __init__.py
│   ├── pipeline/
│   │   ├── sae_pipeline.py            # 完整分析 Pipeline
│   │   └── __init__.py
│   └── __init__.py
├── test/
│   ├── test_sparse_autoencoder.py     # SAE 模型测试
│   ├── test_activation_extractor.py   # 激活提取测试
│   ├── test_sae_trainer.py            # 训练器测试
│   └── test_helix_integration.py      # Helix mRNA 集成测试
├── examples/
│   └── full_pipeline_example.py       # 完整示例
├── outputs/                            # 输出目录（自动创建）
├── README.md
└── .gitignore
```

---

## 🚀 安装

### 1. 克隆仓库

```bash
cd /home/pan/Experiments/EXPs/2025_10_FM_explainability/SAE_Helix_mRNA
```

### 2. 安装依赖

```bash
# 基础依赖
pip install torch numpy tqdm

# Helix mRNA 模型
pip install helical

# 可选：用于可视化
pip install matplotlib seaborn
```

### 3. 验证安装

```bash
python test/test_sparse_autoencoder.py
```

---

## ⚡ 快速开始

### 最简示例

```python
from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig
from src.pipeline import SAEAnalysisPipeline
import torch

# 1. 初始化 Helix mRNA 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
helix_config = HelixmRNAConfig(batch_size=16, device=device)
helix_model = HelixmRNA(configurer=helix_config)

# 2. 准备 RNA 序列
sequences = ["EACUEGGG" * 20] * 1000
dataset = helix_model.process_data(sequences)

# 3. 创建 SAE Pipeline
pipeline = SAEAnalysisPipeline(
    helix_model=helix_model,
    expansion_factor=4,
    l1_coefficient=1e-3,
    device=device
)

# 4. 运行完整分析
results = pipeline.run_full_analysis(
    dataset=dataset,
    layer_filter=lambda name, m: 'mixer' in name.lower(),
    num_epochs=100,
    save_dir='./outputs/my_analysis'
)

# 5. 查看结果
for layer_name, analysis in results['feature_analyses'].items():
    print(f"{layer_name}: {analysis['n_features']} features")
```

---

## 📖 详细用法

### 1. 提取激活值

```python
from src.model import ActivationExtractor

# 创建提取器
extractor = ActivationExtractor(helix_model.model)

# 注册要提取的层
extractor.register_hooks(
    layer_filter=lambda name, m: 'mixer' in name  # 只提取 mixer 层
)

# 运行模型
embeddings = helix_model.get_embeddings(dataset)

# 获取激活
activations = extractor.get_activations()
extractor.remove_hooks()
```

### 2. 训练 SAE

```python
from src.model import SparseAutoencoder, SAEConfig, SAETrainer

# 配置 SAE
config = SAEConfig(
    d_in=512,              # 输入维度
    expansion_factor=4,    # 过完备比率
    l1_coefficient=1e-3,   # L1 系数
    learning_rate=1e-3
)

# 创建模型
sae = SparseAutoencoder(config)

# 训练
trainer = SAETrainer(sae, config, device='cuda')
history = trainer.train(
    activations=your_activations,
    num_epochs=100,
    batch_size=256,
    save_dir='./outputs/sae_checkpoints'
)
```

### 3. 分析特征

```python
# 获取特征字典
feature_dict = sae.get_feature_dictionary()  # (d_hidden, d_in)

# 编码新的激活
with torch.no_grad():
    features = sae.encode(new_activations)
    
# 计算稀疏性统计
stats = sae.get_sparsity_stats(features)
print(f"L0 范数: {stats['l0_norm']:.1f}")
print(f"稀疏度: {stats['sparsity']:.2%}")
```

### 4. 多层分析

```python
from src.model import MultiLayerSAETrainer

# 准备多层激活
layer_activations = {
    'layer_0': torch.randn(10000, 512),
    'layer_1': torch.randn(10000, 512),
    'layer_2': torch.randn(10000, 512),
}

# 训练所有层
trainer = MultiLayerSAETrainer(
    layer_activations=layer_activations,
    expansion_factor=4,
    l1_coefficient=1e-3
)

all_histories = trainer.train_all(
    num_epochs=100,
    save_dir='./outputs/multi_layer'
)

# 获取特定层的 SAE
sae_layer0 = trainer.get_sae('layer_0')
```

---

## 🧪 测试

### 运行所有测试

```bash
# SAE 模型测试
python test/test_sparse_autoencoder.py

# 激活提取测试
python test/test_activation_extractor.py

# 训练器测试
python test/test_sae_trainer.py

# Helix mRNA 集成测试（需要安装 helical）
python test/test_helix_integration.py
```

### 运行示例

```bash
python examples/full_pipeline_example.py
```

---

## 🔧 配置参数

### SAEConfig 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `d_in` | int | - | 输入维度（激活向量维度） |
| `expansion_factor` | int | 4 | 过完备比率 R（隐藏层 = R × 输入层） |
| `l1_coefficient` | float | 1e-3 | L1 稀疏惩罚系数 α |
| `learning_rate` | float | 1e-3 | 学习率 |
| `normalize_decoder` | bool | True | 是否归一化解码器权重 |
| `tied_weights` | bool | True | 是否使用权重绑定 |

### 训练参数建议

- **expansion_factor**: 2-8（越大特征越多，但训练越慢）
- **l1_coefficient**: 1e-4 到 1e-2（越大越稀疏）
- **num_epochs**: 50-200（取决于数据量）
- **batch_size**: 128-512（取决于 GPU 内存）

---

## 📊 输出说明

### 训练输出

```
outputs/
└── my_analysis/
    ├── layer_name/
    │   ├── best_model.pt              # 最佳模型检查点
    │   ├── final_model.pt             # 最终模型
    │   └── training_history.json      # 训练历史
    └── analysis_results.pkl           # 完整分析结果
```

### 分析结果

`analysis_results.pkl` 包含：

```python
{
    'layer_activations_shapes': {...},  # 每层激活的形状
    'training_histories': {...},        # 训练历史
    'feature_analyses': {
        'layer_name': {
            'n_features': int,                    # 特征数量
            'feature_dim': int,                   # 特征维度
            'feature_activation_freq': ndarray,   # 激活频率
            'feature_mean_activation': ndarray,   # 平均激活
            'top_k_features': ndarray,            # Top-K 特征索引
            'feature_dictionary': ndarray,        # 特征字典矩阵
        }
    }
}
```

---

## 📚 参考文献

1. **Towards Monosemanticity: Decomposing Language Models With Dictionary Learning**  
   Anthropic, 2023  
   https://arxiv.org/abs/2309.08600

2. **Helix-mRNA: A Foundation Model for mRNA Sequence Analysis**  
   Helical AI

3. **Sparse Autoencoders Find Highly Interpretable Features in Language Models**  
   Cunningham et al., 2023

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License

---

## 💡 提示

### 常见问题

**Q: 如何选择合适的 expansion_factor？**  
A: 从 4 开始尝试。如果特征不够丰富，增加到 8；如果训练太慢，减少到 2。

**Q: L1 系数应该设置多少？**  
A: 从 1e-3 开始，观察 L0 范数。目标是激活 5-10% 的特征。

**Q: 训练需要多少数据？**  
A: 建议至少 10,000 个激活向量。数据越多，学到的特征越稳定。

**Q: 如何解释学到的特征？**  
A: 查看特征字典向量，分析哪些输入维度权重最大。可以与已知的生物学概念关联。

### 性能优化

- 使用 GPU 加速训练
- 对于大规模数据，使用 `max_samples` 限制样本数
- 启用混合精度训练（需要 PyTorch >= 1.6）

---

**Happy Exploring! 🚀**
