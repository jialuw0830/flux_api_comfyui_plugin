#!/usr/bin/env python3
"""
测试脚本：验证FLUX API插件的Lora支持功能
"""

import sys
import os

# 添加插件目录到Python路径
plugin_dir = os.path.dirname(os.path.abspath(__file__))
nodes_dir = os.path.join(plugin_dir, "nodes")
sys.path.insert(0, nodes_dir)

try:
    from flux_api_node import FluxAPINode
    
    print("✅ 插件导入成功！")
    print(f"插件目录: {plugin_dir}")
    print(f"节点目录: {nodes_dir}")
    
    # 创建节点实例
    node = FluxAPINode()
    print(f"✅ 节点创建成功: {node.__class__.__name__}")
    
    # 检查INPUT_TYPES
    input_types = node.INPUT_TYPES()
    print("\n📋 节点输入参数:")
    
    print("\n必需参数:")
    for param_name, param_config in input_types["required"].items():
        print(f"  - {param_name}: {param_config[0]}")
    
    print("\n可选参数:")
    for param_name, param_config in input_types["optional"].items():
        print(f"  - {param_name}: {param_config[0]}")
    
    # 检查Lora参数
    lora_params = []
    for param_name in input_types["required"]:
        if "lora" in param_name.lower():
            lora_params.append(param_name)
    
    for param_name in input_types["optional"]:
        if "lora" in param_name.lower():
            lora_params.append(param_name)
    
    print(f"\n🎯 发现的Lora参数数量: {len(lora_params)}")
    for param in lora_params:
        print(f"  - {param}")
    
    if len(lora_params) >= 6:  # 3个lora_name + 3个lora_weight
        print("✅ 插件支持多个Lora！")
    else:
        print("❌ 插件Lora支持不完整")
    
    # 检查方法签名
    import inspect
    method_sig = inspect.signature(node.generate_image)
    print(f"\n🔍 generate_image方法参数:")
    for param_name, param in method_sig.parameters.items():
        print(f"  - {param_name}: {param.annotation}")
    
    print("\n✅ 测试完成！")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保你在插件目录中运行此脚本")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
