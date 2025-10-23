"""
测试 SAE 训练流程 - 完整的端到端测试

测试流程:
1. 加载 Helix mRNA 模型并输出架构
2. 生成若干 mRNA 序列
3. 使用 ActivationExtractor 提取激活值
4. 随机选择一层训练 SAE
"""
import sys
from pathlib import Path
print(f"[root]: {Path(__file__).parent.parent}")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import random
import numpy as np

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

try:
    from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig
    HELICAL_AVAILABLE = True
except ImportError:
    HELICAL_AVAILABLE = False
    raise ImportError("helical 库未安装，跳过测试")

from src.model.activation_extractor import ActivationExtractor
from src.model.sparse_autoencoder import SparseAutoencoder, SAEConfig
from src.model.sae_lightning import SAETrainer
from src.model.multi_sae_lightning import MultiLayerSAETrainer


def print_model_architecture(model):
    """打印模型架构"""
    print("\n" + "="*80)
    print("Helix mRNA 模型架构")
    print("="*80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    print(f"\n模型结构:")
    print("-"*80)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只打印叶子节点
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                print(f"{name:60s} | {type(module).__name__:30s} | {param_count:,} params")
    
    print("="*80)


def generate_mrna_sequences(n_sequences=100, seq_length=50):
    """
    生成随机 mRNA 序列
    
    Parameters:
    -----------
    n_sequences : int
        序列数量
    seq_length : int
        每个序列的长度（碱基数）
    
    Returns:
    --------
    sequences : List[str]
        mRNA 序列列表
    """
    bases = ['A', 'C', 'G', 'U']
    sequences = []
    
    for _ in range(n_sequences):
        seq = ''.join(random.choices(bases, k=seq_length))
        sequences.append(seq)
    
    return sequences


def extract_all_layer_activations(helix_model, dataset, device='cuda'):
    """
    提取所有层的激活值
    
    Parameters:
    -----------
    helix_model : HelixmRNA
        Helix mRNA 模型
    dataset : Dataset
        处理后的数据集
    device : str
        设备
    
    Returns:
    --------
    layer_activations : Dict[str, torch.Tensor]
        每层的激活值字典
    """
    from torch.utils.data import DataLoader
    
    # 创建激活提取器
    extractor = ActivationExtractor(helix_model.model)
    
    # 注册 hooks - 只提取 mixer 模块本身，不包括其子模块
    # 例如: layers.0.mixer, layers.2.mixer, layers.4.mixer, layers.6.mixer
    extractor.register_hooks(
        layer_filter=lambda name, m: (
            'mixer' in name.lower() and 
            name.endswith('mixer')  # 确保是 mixer 本身，不是 mixer.xxx
        )
    )
    
    print(f"\n已注册 {len(extractor.hooks)} 个 hooks 用于激活提取")
    
    # 创建 DataLoader
    config = helix_model.configurer.config
    dataloader = DataLoader(
        dataset,
        collate_fn=helix_model._collate_fn,
        batch_size=config["batch_size"],
        shuffle=False,
    )
    
    # 收集所有激活值
    all_layer_activations = {}
    
    helix_model.model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            extractor.clear_activations()
            
            input_ids = batch["input_ids"].to(device)
            special_tokens_mask = batch["special_tokens_mask"].to(device)
            attention_mask = 1 - special_tokens_mask
            
            # 前向传播
            _ = helix_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 获取当前 batch 的激活值
            batch_activations = extractor.get_activations()
            
            # 累积激活值
            for layer_name, activation in batch_activations.items():
                # activation shape: (batch_size, seq_len, hidden_dim)
                # 展平为 (batch_size * seq_len, hidden_dim)
                batch_size, seq_len, hidden_dim = activation.shape
                flattened = activation.reshape(-1, hidden_dim).cpu()
                
                if layer_name not in all_layer_activations:
                    all_layer_activations[layer_name] = []
                
                all_layer_activations[layer_name].append(flattened)
    
    # 合并所有 batch
    layer_activations = {}
    for layer_name, activation_list in all_layer_activations.items():
        layer_activations[layer_name] = torch.cat(activation_list, dim=0)
    
    extractor.remove_hooks()
    
    return layer_activations


def test_sae_training_pipeline():
    """测试完整的 SAE 训练流程"""
    if not HELICAL_AVAILABLE:
        print("跳过: helical 库未安装")
        return
    
    print("\n" + "="*80)
    print("测试 SAE 训练流程")
    print("="*80)
    
    # ========== 步骤 1: 初始化 Helix mRNA 模型 ==========
    print("\n步骤 1: 初始化 Helix mRNA 模型")
    print("-"*80)
    
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"使用设备: {device}")
    
    config = HelixmRNAConfig(
        batch_size=8,
        max_length=150,
        device=device
    )
    helix_model = HelixmRNA(configurer=config)
    
    # 冻结模型参数，不需要梯度
    for param in helix_model.model.parameters():
        param.requires_grad = False
    
    print(f"已冻结 Helix mRNA 模型的所有参数")
    
    # 打印模型架构
    print_model_architecture(helix_model.model)
    
    # ========== 步骤 2: 生成 mRNA 序列 ==========
    print("\n步骤 2: 生成 mRNA 序列")
    print("-"*80)
    
    n_sequences = 20
    seq_length = 50
    sequences = generate_mrna_sequences(n_sequences=n_sequences, seq_length=seq_length)
    
    print(f"生成了 {len(sequences)} 个序列")
    print(f"序列长度: {seq_length} 碱基")
    print(f"示例序列: {sequences[0][:50]}...")
    
    # 处理数据
    dataset = helix_model.process_data(sequences)
    print(f"数据集大小: {len(dataset)}")
    
    # ========== 步骤 3: 提取激活值 ==========
    print("\n步骤 3: 提取所有层的激活值")
    print("-"*80)
    
    layer_activations = extract_all_layer_activations(helix_model, dataset, device)
    
    print(f"\n提取的层数: {len(layer_activations)}")
    print("\n各层激活值形状:")
    for layer_name, activation in layer_activations.items():
        print(f"  {layer_name:60s} | Shape: {activation.shape}")
    
    # ========== 步骤 4: 为所有层配置并训练 Multi-SAE ==========
    print("\n步骤 4: 为所有层配置并训练 Multi-SAE")
    print("-"*80)
    
    # 显示所有层的信息
    print(f"\n将为以下 {len(layer_activations)} 层训练 SAE:")
    for layer_name, activation in layer_activations.items():
        print(f"  {layer_name:40s} | Shape: {activation.shape} | d_in: {activation.shape[1]}")
    
    # ========== 步骤 5: 创建 Multi-Layer SAE Trainer ==========
    print("\n步骤 5: 创建 Multi-Layer SAE Trainer")
    print("-"*80)
    
    # Multi-SAE 配置
    expansion_factor = 4
    l1_coefficient = 1e-3
    learning_rate = 1e-3
    
    print(f"\nMulti-SAE 配置:")
    print(f"  扩展因子: {expansion_factor}")
    print(f"  L1 系数: {l1_coefficient}")
    print(f"  学习率: {learning_rate}")
    
    # 创建 Multi-Layer SAE Trainer
    multi_trainer = MultiLayerSAETrainer(
        layer_activations=layer_activations,
        expansion_factor=expansion_factor,
        l1_coefficient=l1_coefficient,
        learning_rate=learning_rate,
        accelerator='cpu',  # 使用 CPU
        devices=1
    )
    
    # 显示每层的 SAE 配置
    print(f"\n各层 SAE 配置:")
    for layer_name, config in multi_trainer.configs.items():
        print(f"  {layer_name:40s} | d_in: {config.d_in:4d} | d_hidden: {config.d_hidden:5d}")
    
    # 训练
    save_dir = Path('./test_outputs/multi_sae_checkpoints')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始训练所有层的 SAE...")
    print(f"保存目录: {save_dir}")
    
    histories = multi_trainer.train_all(
        num_epochs=50,
        batch_size=256,
        validation_split=0.1,
        log_interval=10,
        save_dir=save_dir
    )
    
    # ========== 步骤 6: 评估所有层的训练结果 ==========
    print("\n步骤 6: 评估所有层的训练结果")
    print("-"*80)
    
    # 获取所有训练好的 SAE
    all_saes = multi_trainer.get_all_saes()
    
    # 评估每一层
    evaluation_results = {}
    
    with torch.no_grad():
        for layer_name, sae in all_saes.items():
            print(f"\n评估层: {layer_name}")
            print("-" * 60)
            
            # 获取该层的激活值
            layer_activation = layer_activations[layer_name]
            test_samples = layer_activation[:100]  # 取前 100 个样本
            
            sae.eval()
            x_reconstructed, features, loss_dict = sae(test_samples, return_loss=True)
            
            # 计算指标
            mse = torch.mean((test_samples - x_reconstructed) ** 2).item()
            config = multi_trainer.configs[layer_name]
            sparsity = 1.0 - (loss_dict['l0_norm'].item() / config.d_hidden)
            
            print(f"  重建损失: {loss_dict['reconstruction_loss'].item():.6f}")
            print(f"  L0 范数 (平均激活特征数): {loss_dict['l0_norm'].item():.2f}")
            print(f"  稀疏度: {sparsity:.2%}")
            print(f"  MSE: {mse:.6f}")
            
            # 特征统计
            sparsity_stats = sae.get_sparsity_stats(features)
            print(f"  最大激活值: {sparsity_stats['max_activation']:.4f}")
            print(f"  平均激活值: {sparsity_stats['mean_activation']:.4f}")
            
            # 保存评估结果
            evaluation_results[layer_name] = {
                'reconstruction_loss': loss_dict['reconstruction_loss'].item(),
                'l0_norm': loss_dict['l0_norm'].item(),
                'sparsity': sparsity,
                'mse': mse,
                'sparsity_stats': sparsity_stats
            }
    
    print("\n" + "="*80)
    print("✓ Multi-SAE 训练流程测试完成!")
    print("="*80)
    
    # 打印汇总
    print(f"\n训练汇总:")
    print(f"  训练的层数: {len(all_saes)}")
    print(f"  保存目录: {save_dir}")
    print(f"\n各层性能:")
    for layer_name, results in evaluation_results.items():
        print(f"  {layer_name:40s} | MSE: {results['mse']:.6f} | 稀疏度: {results['sparsity']:.2%}")
    
    return {
        'layer_activations': layer_activations,
        'all_saes': all_saes,
        'evaluation_results': evaluation_results,
        'histories': histories,
        'save_dir': save_dir
    }


if __name__ == "__main__":
    result = test_sae_training_pipeline()
    
    if result:
        print(f"\n✓ 所有训练完成!")
        print(f"训练的层数: {len(result['all_saes'])}")
        print(f"模型保存在: {result['save_dir']}")
