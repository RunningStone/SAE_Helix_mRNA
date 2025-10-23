"""
激活值提取器 - 从 Helix mRNA 模型提取所有层的激活值

使用 PyTorch Hook 机制，无需修改原始模型代码
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Tuple
from collections import OrderedDict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LayerInfo:
    """层信息"""
    name: str
    module_type: str
    output_shape: Tuple[int, ...]
    num_activations: int  # 激活向量数量（通常是 batch_size * seq_len）


class ActivationExtractor:
    """
    激活值提取器 - 使用 PyTorch Hook 机制
    
    Example:
    --------
    >>> from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig
    >>> 
    >>> # 初始化模型
    >>> model = HelixmRNA(configurer=HelixmRNAConfig())
    >>> 
    >>> # 创建提取器
    >>> extractor = ActivationExtractor(model.model)
    >>> 
    >>> # 注册要提取的层
    >>> extractor.register_hooks(
    >>>     layer_filter=lambda name, m: 'layer' in name
    >>> )
    >>> 
    >>> # 运行模型
    >>> embeddings = model.get_embeddings(dataset)
    >>> 
    >>> # 获取激活值
    >>> activations = extractor.get_activations()
    >>> 
    >>> # 清理
    >>> extractor.remove_hooks()
    """
    
    def __init__(self, model: nn.Module):
        """
        Parameters:
        -----------
        model : nn.Module
            要提取激活值的模型
        """
        self.model = model
        self.activations: Dict[str, torch.Tensor] = OrderedDict()
        self.hooks: List = []
        self.layer_info: Dict[str, LayerInfo] = {}
        
    def _create_hook(self, name: str) -> Callable:
        """创建 hook 函数来捕获激活值"""
        def hook(module, input, output):
            # 处理不同类型的输出
            if isinstance(output, tuple):
                activation = output[0]
            elif isinstance(output, dict):
                # 某些模型返回字典
                activation = output.get('last_hidden_state', output.get('hidden_states', None))
                if activation is None:
                    activation = list(output.values())[0]
            else:
                activation = output
            
            # 保存激活值（detach 以节省内存）
            if isinstance(activation, torch.Tensor):
                self.activations[name] = activation.detach()
                
                # 记录层信息
                if name not in self.layer_info:
                    self.layer_info[name] = LayerInfo(
                        name=name,
                        module_type=type(module).__name__,
                        output_shape=tuple(activation.shape),
                        num_activations=activation.shape[0] if len(activation.shape) > 0 else 1
                    )
        
        return hook
    
    def register_hooks(
        self, 
        layer_filter: Optional[Callable[[str, nn.Module], bool]] = None,
        layer_names: Optional[List[str]] = None,
        layer_types: Optional[List[type]] = None
    ):
        """
        注册 hooks 到指定的层
        
        Parameters:
        -----------
        layer_filter : Callable, optional
            过滤函数，接收 (name, module) 返回 bool
            例如: lambda name, m: 'layer' in name
        layer_names : List[str], optional
            直接指定层名称列表
        layer_types : List[type], optional
            指定要提取的层类型列表
            例如: [nn.Linear, nn.Conv2d]
        """
        registered_count = 0
        
        for name, module in self.model.named_modules():
            should_register = False
            
            # 检查是否应该注册这一层
            if layer_names is not None and name in layer_names:
                should_register = True
            elif layer_types is not None and any(isinstance(module, t) for t in layer_types):
                should_register = True
            elif layer_filter is not None and layer_filter(name, module):
                should_register = True
            elif layer_filter is None and layer_names is None and layer_types is None:
                # 默认：只注册叶子节点
                if len(list(module.children())) == 0:
                    should_register = True
            
            if should_register:
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)
                registered_count += 1
        
        logger.info(f"✓ 已注册 {registered_count} 个 hooks")
        return registered_count
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """获取所有捕获的激活值"""
        return self.activations
    
    def get_layer_info(self) -> Dict[str, LayerInfo]:
        """获取层信息"""
        return self.layer_info
    
    def clear_activations(self):
        """清空已保存的激活值"""
        self.activations.clear()
    
    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("✓ 已移除所有 hooks")
    
    def __enter__(self):
        """支持 context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """自动清理 hooks"""
        self.remove_hooks()
    
    def print_layer_summary(self):
        """打印层信息摘要"""
        print("\n" + "="*80)
        print("Layer Summary")
        print("="*80)
        print(f"{'Layer Name':<40} {'Type':<20} {'Output Shape':<20}")
        print("-"*80)
        
        for name, info in self.layer_info.items():
            shape_str = str(info.output_shape)
            print(f"{name:<40} {info.module_type:<20} {shape_str:<20}")
        
        print("="*80)
        print(f"Total layers: {len(self.layer_info)}")
        print("="*80 + "\n")


class GradientExtractor(ActivationExtractor):
    """
    梯度提取器 - 扩展 ActivationExtractor 来提取梯度
    
    Example:
    --------
    >>> extractor = GradientExtractor(model.model)
    >>> extractor.register_hooks(layer_filter=lambda name, m: 'layer' in name)
    >>> 
    >>> # 前向传播
    >>> output = model(input_ids, attention_mask)
    >>> loss = criterion(output, labels)
    >>> 
    >>> # 反向传播
    >>> loss.backward()
    >>> 
    >>> # 获取梯度
    >>> gradients = extractor.get_gradients()
    """
    
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.gradients: Dict[str, torch.Tensor] = OrderedDict()
    
    def _create_hook(self, name: str) -> Callable:
        """创建 backward hook 来捕获梯度"""
        def hook(module, grad_input, grad_output):
            # 保存输出梯度
            if isinstance(grad_output, tuple):
                grad = grad_output[0]
            else:
                grad = grad_output
            
            if isinstance(grad, torch.Tensor):
                self.gradients[name] = grad.detach()
        
        return hook
    
    def register_hooks(self, **kwargs):
        """注册 backward hooks"""
        layer_filter = kwargs.get('layer_filter')
        layer_names = kwargs.get('layer_names')
        layer_types = kwargs.get('layer_types')
        
        registered_count = 0
        
        for name, module in self.model.named_modules():
            should_register = False
            
            if layer_names is not None and name in layer_names:
                should_register = True
            elif layer_types is not None and any(isinstance(module, t) for t in layer_types):
                should_register = True
            elif layer_filter is not None and layer_filter(name, module):
                should_register = True
            
            if should_register:
                hook = module.register_full_backward_hook(self._create_hook(name))
                self.hooks.append(hook)
                registered_count += 1
        
        logger.info(f"✓ 已注册 {registered_count} 个梯度 hooks")
        return registered_count
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """获取所有捕获的梯度"""
        return self.gradients


def get_all_layer_names(model: nn.Module, filter_fn: Optional[Callable] = None) -> List[str]:
    """
    获取模型中所有层的名称
    
    Parameters:
    -----------
    model : nn.Module
        模型
    filter_fn : Callable, optional
        过滤函数
    
    Returns:
    --------
    layer_names : List[str]
        层名称列表
    """
    layer_names = []
    for name, module in model.named_modules():
        if filter_fn is None or filter_fn(name, module):
            layer_names.append(name)
    return layer_names


def extract_activations_from_model(
    model: nn.Module,
    input_data: torch.Tensor,
    layer_filter: Optional[Callable] = None,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    便捷函数：从模型提取激活值
    
    Parameters:
    -----------
    model : nn.Module
        模型
    input_data : torch.Tensor
        输入数据
    layer_filter : Callable, optional
        层过滤函数
    device : str
        设备
    
    Returns:
    --------
    activations : Dict[str, torch.Tensor]
        激活值字典
    """
    model.eval()
    model.to(device)
    input_data = input_data.to(device)
    
    with ActivationExtractor(model) as extractor:
        extractor.register_hooks(layer_filter=layer_filter)
        
        with torch.no_grad():
            _ = model(input_data)
        
        activations = extractor.get_activations()
    
    return activations

