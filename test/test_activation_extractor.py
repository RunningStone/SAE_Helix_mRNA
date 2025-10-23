"""
测试激活值提取器
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from src.model import ActivationExtractor, GradientExtractor


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 5)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x


def test_activation_extractor_basic():
    """测试基本激活提取"""
    model = SimpleModel()
    extractor = ActivationExtractor(model)
    
    # 注册所有 Linear 层
    extractor.register_hooks(
        layer_filter=lambda name, m: isinstance(m, nn.Linear)
    )
    
    # 前向传播
    x = torch.randn(8, 10)
    with torch.no_grad():
        output = model(x)
    
    # 获取激活
    activations = extractor.get_activations()
    
    assert 'layer1' in activations
    assert 'layer2' in activations
    assert 'layer3' in activations
    
    assert activations['layer1'].shape == (8, 20)
    assert activations['layer2'].shape == (8, 30)
    assert activations['layer3'].shape == (8, 5)
    
    extractor.remove_hooks()
    
    print("✓ 基本激活提取测试通过")
    print(f"  提取了 {len(activations)} 层的激活")


def test_activation_extractor_layer_names():
    """测试指定层名称提取"""
    model = SimpleModel()
    extractor = ActivationExtractor(model)
    
    # 只提取 layer1 和 layer3
    extractor.register_hooks(layer_names=['layer1', 'layer3'])
    
    x = torch.randn(8, 10)
    with torch.no_grad():
        output = model(x)
    
    activations = extractor.get_activations()
    
    assert 'layer1' in activations
    assert 'layer2' not in activations
    assert 'layer3' in activations
    
    extractor.remove_hooks()
    
    print("✓ 指定层名称提取测试通过")


def test_activation_extractor_context_manager():
    """测试 context manager"""
    model = SimpleModel()
    x = torch.randn(8, 10)
    
    with ActivationExtractor(model) as extractor:
        extractor.register_hooks(layer_types=[nn.Linear])
        
        with torch.no_grad():
            output = model(x)
        
        activations = extractor.get_activations()
        assert len(activations) == 3
    
    # hooks 应该已经被移除
    assert len(extractor.hooks) == 0
    
    print("✓ Context manager 测试通过")


def test_gradient_extractor():
    """测试梯度提取"""
    model = SimpleModel()
    extractor = GradientExtractor(model)
    
    extractor.register_hooks(layer_types=[nn.Linear])
    
    # 前向传播
    x = torch.randn(8, 10)
    output = model(x)
    
    # 反向传播
    loss = output.sum()
    loss.backward()
    
    # 获取梯度
    gradients = extractor.get_gradients()
    
    assert len(gradients) > 0
    
    extractor.remove_hooks()
    
    print("✓ 梯度提取测试通过")
    print(f"  提取了 {len(gradients)} 层的梯度")


def test_layer_info():
    """测试层信息"""
    model = SimpleModel()
    extractor = ActivationExtractor(model)
    
    extractor.register_hooks(layer_types=[nn.Linear])
    
    x = torch.randn(8, 10)
    with torch.no_grad():
        output = model(x)
    
    layer_info = extractor.get_layer_info()
    
    assert 'layer1' in layer_info
    assert layer_info['layer1'].module_type == 'Linear'
    assert layer_info['layer1'].output_shape == (8, 20)
    
    extractor.remove_hooks()
    
    print("✓ 层信息测试通过")


if __name__ == '__main__':
    print("="*80)
    print("测试激活值提取器")
    print("="*80 + "\n")
    
    test_activation_extractor_basic()
    test_activation_extractor_layer_names()
    test_activation_extractor_context_manager()
    test_gradient_extractor()
    test_layer_info()
    
    print("\n" + "="*80)
    print("所有测试通过! ✓")
    print("="*80)
