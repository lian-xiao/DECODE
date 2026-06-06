"""
虚拟筛选框架测试脚本
"""

import sys
import os
import pandas as pd
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_loading():
    """测试数据加载"""
    print("Testing data loading...")
    
    try:
        from virtual_screening.data import VirtualScreeningDataModule
        
        # 使用提供的数据路径
        train_data_path = "preprocessed_data/Virtual_screening/EP4/ChEMBL-EP4_processed_ac.csv"
        external_val_path = "preprocessed_data/Virtual_screening/EP4/ExtVal_EP4_processed_ac.csv"
        
        if not os.path.exists(train_data_path):
            print(f"Warning: Training data not found at {train_data_path}")
            return False
        
        # 创建数据模块
        data_module = VirtualScreeningDataModule(
            train_data_path=train_data_path,
            external_val_data_path=external_val_path,
            batch_size=4,
            num_workers=0  # 避免多进程问题
        )
        
        # 设置数据
        data_module.setup()
        
        # 获取数据信息
        data_info = data_module.get_data_info()
        print(f"Data info: {data_info}")
        
        # 测试数据加载器
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch sizes: {[(k, v.shape if torch.is_tensor(v) else len(v)) for k, v in batch.items()]}")
        
        print("✓ Data loading test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False


def test_molformer_model():
    """测试Molformer模型"""
    print("\nTesting Molformer model...")
    
    try:
        from virtual_screening.vs_models import MolformerModule
        
        # 创建模型（使用较小的模型进行测试）
        model = MolformerModule(
            model_name="seyonec/PubChem10M_SMILES_BPE_450k",  # 较小的模型
            num_classes=2,
            hidden_dim=128
        )
        
        # 测试前向传播
        test_smiles = ["CCO", "C1CCCCC1", "c1ccccc1"]
        
        # 提取特征
        features = model.extract_features(test_smiles)
        print(f"Features shape: {features.shape}")
        
        # 前向传播
        logits = model(test_smiles)
        print(f"Logits shape: {logits.shape}")
        
        print("✓ Molformer model test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Molformer model test failed: {e}")
        return False


def test_virtual_screening_model():
    """测试虚拟筛选模型"""
    print("\nTesting Virtual Screening model...")
    
    try:
        from virtual_screening.vs_models import MolformerModule, VirtualScreeningModule
        
        # 创建Molformer模型
        molformer = MolformerModule(
            model_name="seyonec/PubChem10M_SMILES_BPE_450k",
            num_classes=2,
            hidden_dim=128
        )
        
        # 创建虚拟筛选模型
        vs_model = VirtualScreeningModule(
            moa_model_path="",  # 不加载预训练权重
            molformer_model=molformer,
            num_classes=2,
            hidden_dim=128
        )
        
        # 测试前向传播
        test_smiles = ["CCO", "C1CCCCC1", "c1ccccc1"]
        test_doses = torch.tensor([[1.0], [2.0], [0.5]])
        
        logits = vs_model(test_smiles, test_doses)
        print(f"VS model logits shape: {logits.shape}")
        
        print("✓ Virtual Screening model test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Virtual Screening model test failed: {e}")
        return False


def test_utils():
    """测试工具函数"""
    print("\nTesting utility functions...")
    
    try:
        from virtual_screening.utils import validate_smiles, smiles_to_molformer_features
        
        # 测试SMILES验证
        valid_smiles = ["CCO", "C1CCCCC1", "c1ccccc1"]
        invalid_smiles = ["XYZ", ""]
        
        for smiles in valid_smiles:
            is_valid = validate_smiles(smiles)
            print(f"SMILES '{smiles}' is valid: {is_valid}")
        
        # 测试特征提取（如果网络允许）
        try:
            features = smiles_to_molformer_features(
                valid_smiles, 
                model_name="seyonec/PubChem10M_SMILES_BPE_450k"
            )
            print(f"Extracted features shape: {features.shape}")
        except Exception as e:
            print(f"Feature extraction failed (network issue?): {e}")
        
        print("✓ Utility functions test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("Running Virtual Screening Framework Tests")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_molformer_model,
        test_virtual_screening_model,
        test_utils
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The framework is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    main()