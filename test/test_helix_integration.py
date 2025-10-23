"""
测试与 Helix mRNA 模型的集成

注意：需要安装 helical 库
pip install helical
"""

import sys
from pathlib import Path
print(f"[root]: {Path(__file__).parent.parent}")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

try:
    from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig
    HELICAL_AVAILABLE = True
except ImportError:
    HELICAL_AVAILABLE = False
    print("警告: helical 库未安装，跳过集成测试")

from src.model import ActivationExtractor
from src.pipeline import SAEAnalysisPipeline


def test_helix_activation_extraction():
    """测试从 Helix mRNA 提取激活"""
    if not HELICAL_AVAILABLE:
        print("跳过: helical 库未安装")
        return
    
    print("\n" + "="*80)
    print("测试 Helix mRNA 激活提取")
    print("="*80)
    
    # 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = HelixmRNAConfig(batch_size=4, max_length=100, device=device)
    helix_model = HelixmRNA(configurer=config)
    
    # 准备数据（小规模测试）
    sequences = ["EACUEGGG" * 10] * 20
    dataset = helix_model.process_data(sequences)
    
    # 创建激活提取器
    extractor = ActivationExtractor(helix_model.model)
    
    # 注册 hooks - 提取所有包含 'mixer' 的层
    extractor.register_hooks(
        layer_filter=lambda name, m: 'mixer' in name.lower()
    )
    
    print(f"\n已注册 {len(extractor.hooks)} 个 hooks")
    
    # 运行模型提取激活
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        collate_fn=helix_model._collate_fn,
        batch_size=config.config["batch_size"],
        shuffle=False,
    )
    
    helix_model.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            extractor.clear_activations()
            
            input_ids = batch["input_ids"].to(device)
            special_tokens_mask = batch["special_tokens_mask"].to(device)
            attention_mask = 1 - special_tokens_mask
            
            _ = helix_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            activations = extractor.get_activations()
            
            print(f"\n提取的激活层:")
            for name, act in activations.items():
                print(f"  {name}: {act.shape}")
            
            break  # 只测试一个 batch
    
    extractor.remove_hooks()
    
    print("\n✓ Helix mRNA 激活提取测试通过")


def test_sae_pipeline_integration():
    """测试完整 SAE Pipeline"""
    if not HELICAL_AVAILABLE:
        print("跳过: helical 库未安装")
        return
    
    print("\n" + "="*80)
    print("测试 SAE Pipeline 集成")
    print("="*80)
    
    # 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = HelixmRNAConfig(batch_size=4, max_length=100, device=device)
    helix_model = HelixmRNA(configurer=config)
    
    # 准备数据（小规模测试）
    sequences = ["EACUEGGG" * 10] * 50
    dataset = helix_model.process_data(sequences)
    
    # 创建 pipeline
    pipeline = SAEAnalysisPipeline(
        helix_model=helix_model,
        expansion_factor=2,  # 小一点的扩展因子用于测试
        l1_coefficient=1e-3,
        device=device
    )
    
    # 提取激活（只提取一层用于快速测试）
    print("\n步骤 1: 提取激活值...")
    layer_activations = pipeline.extract_activations(
        dataset=dataset,
        layer_filter=lambda name, m: 'backbone.layers.0' in name and 'mixer' in name,
        max_samples=50
    )
    
    print(f"\n提取的层:")
    for name, act in layer_activations.items():
        print(f"  {name}: {act.shape}")
    
    # 训练 SAE（少量 epoch 用于测试）
    print("\n步骤 2: 训练 SAE...")
    training_histories = pipeline.train_saes(
        num_epochs=10,
        batch_size=32,
        save_dir=None
    )
    
    # 分析特征
    print("\n步骤 3: 分析特征...")
    for layer_name in layer_activations.keys():
        analysis = pipeline.analyze_features(layer_name, top_k=5)
        print(f"\n层 {layer_name} 分析:")
        print(f"  特征数: {analysis['n_features']}")
        print(f"  特征维度: {analysis['feature_dim']}")
    
    print("\n✓ SAE Pipeline 集成测试通过")


def test_list_helix_layers():
    """列出 Helix mRNA 模型的所有层"""
    if not HELICAL_AVAILABLE:
        print("跳过: helical 库未安装")
        return
    
    print("\n" + "="*80)
    print("Helix mRNA 模型层结构")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = HelixmRNAConfig(batch_size=4, device=device)
    helix_model = HelixmRNA(configurer=config)
    
    print("\n所有层:")
    for name, module in helix_model.model.named_modules():
        if len(list(module.children())) == 0:  # 只显示叶子节点
            print(f"  {name}: {type(module).__name__}")
    
    print("\n包含 'mixer' 的层:")
    for name, module in helix_model.model.named_modules():
        if 'mixer' in name.lower():
            print(f"  {name}: {type(module).__name__}")


if __name__ == '__main__':
    print("="*80)
    print("测试 Helix mRNA 集成")
    print("="*80)
    
    if HELICAL_AVAILABLE:
        test_list_helix_layers()
        test_helix_activation_extraction()
        test_sae_pipeline_integration()
        
        print("\n" + "="*80)
        print("所有集成测试通过! ✓")
        print("="*80)
    else:
        print("\n请先安装 helical 库:")
        print("  pip install helical")
