"""
消融实验批次训练多模态MOA预测模型
支持一键运行多种消融实验，包括去除重建、噪声、GAU、对比学习、MOA预测等
每个消融实验都会记录对应的模型名称和参数配置
"""

import os
import sys
import argparse
import yaml
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import copy

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
from models.distangle_multimodal.train_multimodal_two_stage_predictor import (
    MOADataModule, load_config, create_model, create_callbacks, train_moa_model
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AblationExperimentConfig:
    """消融实验配置管理类"""
    
    @staticmethod
    def get_ablation_configs() -> Dict[str, Dict[str, Any]]:
        """
        定义解耦多模态两阶段训练的消融实验配置
        主要消融组件：
        1. 共享特征对比学习 (shared_contrastive_loss_weight)
        2. 专有特征正交损失 (orthogonal_loss_weight) 
        3. 第一阶段预训练 (independent_training)
        4. 第二阶段任务组合 (reconstruction + classification)
        """
        
        ablation_configs = {
            
            # ========== 基线模型 ==========
            # 'full_model_sequential': {
            #     'model_name': 'PRISM-Full-Sequential',
            #     'description': 'Complete two-stage model with all components (sequential training)',
            #     'stage1_params_override': {
            #         'model_config': {
            #             'is_stage1': True,
            #             'classification_loss_weight': 0,
            #             'reconstruction_loss_weight': 1.0,
            #             'shared_contrastive_loss_weight': 1.0,
            #             'orthogonal_loss_weight': 0.5,
            #         },
            #         'training': {
            #             'independent_training': False,
            #             'early_stopping': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min',
            #                 'filename': 'stage1-{epoch:02d}-{val_loss:.6f}'
            #             }
            #         }
            #     },
            #     'stage2_params_override': {
            #         'model_config': {
            #             'is_stage1': False,
            #             'classification_loss_weight': 1.0,
            #             'reconstruction_loss_weight': 0.0,
            #             'shared_contrastive_loss_weight': 0.0,
            #             'orthogonal_loss_weight': 0.0,
            #         },
            #         'training': {
            #             'independent_training': False,
            #             'early_stopping': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max',
            #                 'filename': 'stage2-{epoch:02d}-{val_no_missing_moa_f1_macro:.6f}'
            #             }
            #         }
            #     }
            # },
            
            'full_model_joint_stage2': {
                'model_name': 'PRISM-Full-Joint',
                'description': 'Two-stage model with joint reconstruction+classification in stage2',
                'stage1_params_override': {
                    'model_config': {
                        'is_stage1': True,
                        'classification_loss_weight': 0,
                        'reconstruction_loss_weight': 1.0,
                        'shared_contrastive_loss_weight': 1,
                        'orthogonal_loss_weight': 0.5,
                    },
                    'training': {
                        'independent_training': False,
                        'early_stopping': {
                            'monitor': 'val_loss',
                            'mode': 'min'
                        },
                        'checkpoint': {
                            'monitor': 'val_loss',
                            'mode': 'min',
                            'filename': 'stage1-{epoch:02d}-{val_loss:.6f}'
                        }
                    }
                },
                'stage2_params_override': {
                    'model_config': {
                        'is_stage1': False,
                        'classification_loss_weight': 1.0,
                        'reconstruction_loss_weight': 1.0,  # 同时进行重建和分类
                        'shared_contrastive_loss_weight': 0.0,
                        'orthogonal_loss_weight': 0.0,
                    },
                    'training': {
                        'independent_training': False,
                        'early_stopping': {
                            'monitor': 'val_no_missing_moa_f1_macro',
                            'mode': 'max'
                        },
                        'checkpoint': {
                            'monitor': 'val_no_missing_moa_f1_macro',
                            'mode': 'max',
                            'filename': 'stage2-{epoch:02d}-{val_no_missing_moa_f1_macro:.6f}'
                        }
                    }
                }
            },
            
            'stage2_reconstruction_only': {
                'model_name': 'PRISM-Stage2-ReconOnly',
                'description': 'Two-stage model with Stage2 performing reconstruction only (no classification)',
                'stage1_params_override': {
                    'model_config': {
                        'is_stage1': True,
                        'classification_loss_weight': 0,
                        'reconstruction_loss_weight': 1.0,
                        'shared_contrastive_loss_weight': 1.0,
                        'orthogonal_loss_weight': 0.5,
                    },
                    'training': {
                        'independent_training': False,
                        'early_stopping': {
                            'monitor': 'val_loss',
                            'mode': 'min'
                        },
                        'checkpoint': {
                            'monitor': 'val_loss',
                            'mode': 'min',
                            'filename': 'stage1-{epoch:02d}-{val_loss:.6f}'
                        }
                    }
                },
                'stage2_params_override': {
                    'model_config': {
                        'is_stage1': False,
                        'classification_loss_weight': 0.0,  # 去除分类任务
                        'reconstruction_loss_weight': 1.0,  # 仅进行重建任务
                        'shared_contrastive_loss_weight': 0.0,
                        'orthogonal_loss_weight': 0.0,
                    },
                    'training': {
                        'independent_training': False,
                        'early_stopping': {
                            'monitor': 'val_loss',  # 监控重建损失
                            'mode': 'min'
                        },
                        'checkpoint': {
                            'monitor': 'val_loss',
                            'mode': 'min',
                            'filename': 'stage2-recon-{epoch:02d}-{val_loss:.6f}'
                        }
                    }
                }
            },

            # # ========== 单个组件消融 ==========
            
            # 'no_shared_contrastive': {
            #     'model_name': 'PRISM-NoSharedContrastive',
            #     'description': 'Model without shared contrastive learning',
            #     'stage1_params_override': {
            #         'model_config': {
            #             'is_stage1': True,
            #             'classification_loss_weight': 0,
            #             'reconstruction_loss_weight': 1.0,
            #             'shared_contrastive_loss_weight': 0.0,  # 去除共享对比学习
            #             'orthogonal_loss_weight': 0.5,
            #         },
            #         'training': {
            #             'independent_training': False,
            #             'early_stopping': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min',
            #                 'filename': 'stage1-{epoch:02d}-{val_loss:.6f}'
            #             }
            #         }
            #     },
            #     'stage2_params_override': {
            #         'model_config': {
            #             'is_stage1': False,
            #             'classification_loss_weight': 1.0,
            #             'reconstruction_loss_weight': 0.0,
            #             'shared_contrastive_loss_weight': 0.0,
            #             'orthogonal_loss_weight': 0.0,
            #         },
            #         'training': {
            #             'independent_training': False,
            #             'early_stopping': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max',
            #                 'filename': 'stage2-{epoch:02d}-{val_no_missing_moa_f1_macro:.6f}'
            #             }
            #         }
            #     }
            # },
            
            # 'no_orthogonal': {
            #     'model_name': 'PRISM-NoOrthogonal',
            #     'description': 'Model without orthogonal loss for private features',
            #     'stage1_params_override': {
            #         'model_config': {
            #             'is_stage1': True,
            #             'classification_loss_weight': 0,
            #             'reconstruction_loss_weight': 1.0,
            #             'shared_contrastive_loss_weight': 1.0,
            #             'orthogonal_loss_weight': 0.0,  # 去除正交损失
            #         },
            #         'training': {
            #             'independent_training': False,
            #             'early_stopping': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min',
            #                 'filename': 'stage1-{epoch:02d}-{val_loss:.6f}'
            #             }
            #         }
            #     },
            #     'stage2_params_override': {
            #         'model_config': {
            #             'is_stage1': False,
            #             'classification_loss_weight': 1.0,
            #             'reconstruction_loss_weight': 0.0,
            #             'shared_contrastive_loss_weight': 0.0,
            #             'orthogonal_loss_weight': 0.0,
            #         },
            #         'training': {
            #             'independent_training': False,
            #             'early_stopping': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max',
            #                 'filename': 'stage2-{epoch:02d}-{val_no_missing_moa_f1_macro:.6f}'
            #             }
            #         }
            #     }
            # },
            
            # 'independent_training': {
            #     'model_name': 'PRISM-Independent',
            #     'description': 'Model with independent training (no stage1 pretraining)',
            #     'stage1_params_override': {
            #         'model_config': {
            #             'is_stage1': True,
            #             'classification_loss_weight': 0,
            #             'reconstruction_loss_weight': 1.0,
            #             'shared_contrastive_loss_weight': 1.0,
            #             'orthogonal_loss_weight': 0.5,
            #         },
            #         'training': {
            #             'independent_training': True,  # 独立训练
            #             'early_stopping': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min',
            #                 'filename': 'stage1-{epoch:02d}-{val_loss:.6f}'
            #             }
            #         }
            #     },
            #     'stage2_params_override': {
            #         'model_config': {
            #             'is_stage1': False,
            #             'classification_loss_weight': 1.0,
            #             'reconstruction_loss_weight': 0.0,
            #             'shared_contrastive_loss_weight': 1.0,
            #             'orthogonal_loss_weight': 0.5,
            #         },
            #         'training': {
            #             'independent_training': True,  # 独立训练
            #             'early_stopping': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max',
            #                 'filename': 'stage2-{epoch:02d}-{val_no_missing_moa_f1_macro:.6f}'
            #             }
            #         }
            #     }
            # },

            # # ========== 两个组件组合消融 ==========
            
            # 'no_shared_contrastive_orthogonal': {
            #     'model_name': 'PRISM-NoSharedContrastive-NoOrthogonal',
            #     'description': 'Model without shared contrastive learning and orthogonal loss',
            #     'stage1_params_override': {
            #         'model_config': {
            #             'is_stage1': True,
            #             'classification_loss_weight': 0.0,
            #             'reconstruction_loss_weight': 1.0,
            #             'shared_contrastive_loss_weight': 0.0,  # 去除共享对比学习
            #             'orthogonal_loss_weight': 0.0,  # 去除正交损失
            #         },
            #         'training': {
            #             'independent_training': False,
            #             'early_stopping': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min',
            #                 'filename': 'stage1-{epoch:02d}-{val_loss:.6f}'
            #             }
            #         }
            #     },
            #     'stage2_params_override': {
            #         'model_config': {
            #             'is_stage1': False,
            #             'classification_loss_weight': 1.0,
            #             'reconstruction_loss_weight': 0.0,
            #             'shared_contrastive_loss_weight': 0.0,
            #             'orthogonal_loss_weight': 0.0,
            #         },
            #         'training': {
            #             'independent_training': False,
            #             'early_stopping': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max',
            #                 'filename': 'stage2-{epoch:02d}-{val_no_missing_moa_f1_macro:.6f}'
            #             }
            #         }
            #     }
            # },
            
            # 'no_shared_contrastive_independent': {
            #     'model_name': 'PRISM-NoSharedContrastive-Independent',
            #     'description': 'Model without shared contrastive learning and independent training',
            #     'stage1_params_override': {
            #         'model_config': {
            #             'is_stage1': True,
            #             'classification_loss_weight': 0.0,
            #             'reconstruction_loss_weight': 1.0,
            #             'shared_contrastive_loss_weight': 0.0,  # 去除共享对比学习
            #             'orthogonal_loss_weight': 0.5,
            #         },
            #         'training': {
            #             'independent_training': True,  # 独立训练
            #             'early_stopping': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min',
            #                 'filename': 'stage1-{epoch:02d}-{val_loss:.6f}'
            #             }
            #         }
            #     },
            #     'stage2_params_override': {
            #         'model_config': {
            #             'is_stage1': False,
            #             'classification_loss_weight': 1.0,
            #             'reconstruction_loss_weight': 0.0,
            #             'shared_contrastive_loss_weight': 0.0,  # 去除共享对比学习
            #             'orthogonal_loss_weight': 0.5,
            #         },
            #         'training': {
            #             'independent_training': True,  # 独立训练
            #             'early_stopping': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max',
            #                 'filename': 'stage2-{epoch:02d}-{val_no_missing_moa_f1_macro:.6f}'
            #             }
            #         }
            #     }
            # },
            
            # 'no_orthogonal_independent': {
            #     'model_name': 'PRISM-NoOrthogonal-Independent',
            #     'description': 'Model without orthogonal loss and independent training',
            #     'stage1_params_override': {
            #         'model_config': {
            #             'is_stage1': True,
            #             'classification_loss_weight': 0.0,
            #             'reconstruction_loss_weight': 1.0,
            #             'shared_contrastive_loss_weight': 1.0,
            #             'orthogonal_loss_weight': 0.0,  # 去除正交损失
            #         },
            #         'training': {
            #             'independent_training': True,  # 独立训练
            #             'early_stopping': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min',
            #                 'filename': 'stage1-{epoch:02d}-{val_loss:.6f}'
            #             }
            #         }
            #     },
            #     'stage2_params_override': {
            #         'model_config': {
            #             'is_stage1': False,
            #             'classification_loss_weight': 1.0,
            #             'reconstruction_loss_weight': 0.0,
            #             'shared_contrastive_loss_weight': 1.0,
            #             'orthogonal_loss_weight': 0.0,  # 去除正交损失
            #         },
            #         'training': {
            #             'independent_training': True,  # 独立训练
            #             'early_stopping': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max',
            #                 'filename': 'stage2-{epoch:02d}-{val_no_missing_moa_f1_macro:.6f}'
            #             }
            #         }
            #     }
            # },

            # # ========== 三个组件全部消融 ==========
            
            # 'no_all_components': {
            #     'model_name': 'PRISM-NoAll',
            #     'description': 'Model without shared contrastive, orthogonal loss, and independent training',
            #     'stage1_params_override': {
            #         'model_config': {
            #             'is_stage1': True,
            #             'classification_loss_weight': 0.0,
            #             'reconstruction_loss_weight': 1.0,
            #             'shared_contrastive_loss_weight': 0.0,  # 去除共享对比学习
            #             'orthogonal_loss_weight': 0.0,  # 去除正交损失
            #         },
            #         'training': {
            #             'independent_training': True,  # 独立训练
            #             'early_stopping': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_loss',
            #                 'mode': 'min',
            #                 'filename': 'stage1-{epoch:02d}-{val_loss:.6f}'
            #             }
            #         }
            #     },
            #     'stage2_params_override': {
            #         'model_config': {
            #             'is_stage1': False,
            #             'classification_loss_weight': 1.0,
            #             'reconstruction_loss_weight': 0.0,
            #             'shared_contrastive_loss_weight': 0.0,
            #             'orthogonal_loss_weight': 0.0,
            #         },
            #         'training': {
            #             'independent_training': True,  # 独立训练
            #             'early_stopping': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max'
            #             },
            #             'checkpoint': {
            #                 'monitor': 'val_no_missing_moa_f1_macro',
            #                 'mode': 'max',
            #                 'filename': 'stage2-{epoch:02d}-{val_no_missing_moa_f1_macro:.6f}'
            #             }
            #         }
            #     }
            # },
        
        
        }
        
        return ablation_configs
    
    @staticmethod
    def get_custom_ablation_configs() -> Dict[str, Dict[str, Any]]:
        """
        用户可以自定义的消融实验配置
        用户可以根据需要修改这个函数来添加自己的消融实验
        """
        
        custom_configs = {
            # 示例：不同的权重配置
            'balanced_weights': {
                'model_name': 'BalancedWeights',
                'description': 'Model with balanced loss weights',
                'params_override': {
                    'model': {
                        'moa_weight': 1.0,
                        'reconstruction_weight': 1.0,
                        'contrastive_weight': 0.5
                    }
                }
            },
            
            # 示例：高dropout
            'high_dropout': {
                'model_name': 'HighDropout',
                'description': 'Model with high dropout rate',
                'params_override': {
                    'model': {
                        'dropout_rate': 0.5,
                        'feature_dropout_rate': 0.3
                    }
                }
            }
        }
        
        return custom_configs


class AblationResultCollector:
    """消融实验结果收集器"""
    
    def __init__(self, output_dir: str, experiment_name: str = "AblationExperiment"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.results = []
        
        # 创建结果文件
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"{experiment_name}_ablation_results_{self.timestamp}.csv"
        self.summary_file = self.output_dir / f"{experiment_name}_ablation_summary_{self.timestamp}.txt"
        self.config_file = self.output_dir / f"{experiment_name}_ablation_configs_{self.timestamp}.json"
        
        # 存储所有实验的配置信息
        self.experiment_configs = {}
        
        logger.info(f"AblationResultCollector initialized:")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Results file: {self.results_file.name}")
        logger.info(f"  Summary file: {self.summary_file.name}")
        logger.info(f"  Config file: {self.config_file.name}")
    
    def extract_test_results_from_training(self, training_results: Dict[str, Any], 
                                          split_info: Dict[str, Any],
                                          ablation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        从两阶段训练结果中提取测试指标，并添加消融实验信息
        
        Args:
            training_results: 来自train_moa_model的训练结果
            split_info: split信息
            ablation_config: 消融实验配置信息
            
        Returns:
            包含所有场景测试结果和消融信息的字典
        """
        
        model_name = ablation_config['model_name']
        logger.info(f"Extracting test results for {model_name} - split {split_info.get('split_index', 0)}...")
        
        scenario_results = {}
        
        # 从训练结果中获取测试结果 - 使用智能合并后的结果
        test_results = training_results.get('test_results', {})
        
        if test_results:
            # 定义场景和对应的指标前缀
            scenarios = ['no_missing', 'pheno_missing', 'rna_missing', 'both_missing']
            
            for scenario in scenarios:
                scenario_metrics = {}
                
                # 提取MOA分类指标 - 统一格式，支持多种键名
                for metric_type in ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']:
                    # 尝试多种可能的键名格式
                    possible_keys = [
                        f'test_{scenario}_moa_{metric_type}',  # 原始格式
                        f'{scenario}_moa_{metric_type}',       # 简化格式
                        f'moa_{metric_type}_{scenario}',       # 反序格式
                        f'classification_{scenario}_moa_{metric_type}',  # 带前缀格式
                        f'stage2_test_{scenario}_moa_{metric_type}',  # Stage2前缀格式
                        f'stage1_test_{scenario}_moa_{metric_type}'   # Stage1前缀格式
                    ]
                    
                    for key in possible_keys:
                        if key in test_results:
                            scenario_metrics[f'moa_{metric_type}'] = float(test_results[key])
                            break
                
                # 提取重建指标 - RNA（针对缺失场景）
                if scenario != 'no_missing':
                    for metric_type in ['mse', 'mae', 'r2', 'pearson', 'spearman']:
                        possible_keys = [
                            f'test_{scenario}_rna_{metric_type}',     # 原始格式
                            f'{scenario}_rna_{metric_type}',          # 简化格式  
                            f'rna_{metric_type}_{scenario}',          # 反序格式
                            f'stage1_rna_recon_{scenario}_{metric_type}',  # Stage1重建格式
                            f'stage2_rna_recon_{scenario}_{metric_type}',  # Stage2重建格式
                            f'stage1_test_{scenario}_rna_{metric_type}',   # Stage1测试格式
                            f'stage2_test_{scenario}_rna_{metric_type}'    # Stage2测试格式
                        ]
                        
                        for key in possible_keys:
                            if key in test_results:
                                scenario_metrics[f'rna_{metric_type}'] = float(test_results[key])
                                break
                    
                    # 提取重建指标 - Pheno（针对缺失场景）
                    for metric_type in ['mse', 'mae', 'r2', 'pearson', 'spearman']:
                        possible_keys = [
                            f'test_{scenario}_pheno_{metric_type}',   # 原始格式
                            f'{scenario}_pheno_{metric_type}',        # 简化格式
                            f'pheno_{metric_type}_{scenario}',        # 反序格式
                            f'stage1_pheno_recon_{scenario}_{metric_type}',  # Stage1重建格式
                            f'stage2_pheno_recon_{scenario}_{metric_type}',  # Stage2重建格式
                            f'stage1_test_{scenario}_pheno_{metric_type}',   # Stage1测试格式
                            f'stage2_test_{scenario}_pheno_{metric_type}'    # Stage2测试格式
                        ]
                        
                        for key in possible_keys:
                            if key in test_results:
                                scenario_metrics[f'pheno_{metric_type}'] = float(test_results[key])
                                break
                
                # 如果有任何指标，则添加到结果中
                if scenario_metrics:
                    scenario_results[scenario] = scenario_metrics
                    
                    # 打印关键指标
                    if 'moa_accuracy' in scenario_metrics:
                        logger.info(f"  {scenario}: MOA Accuracy = {scenario_metrics['moa_accuracy']:.4f}")
                    if 'rna_r2' in scenario_metrics:
                        logger.info(f"  {scenario}: RNA R² = {scenario_metrics['rna_r2']:.4f}")
                    if 'pheno_r2' in scenario_metrics:
                        logger.info(f"  {scenario}: Pheno R² = {scenario_metrics['pheno_r2']:.4f}")
                else:
                    logger.warning(f"No metrics found for scenario: {scenario}")
                    
        else:
            logger.warning("No test results provided")
        
        # 如果没有提取到任何结果，记录调试信息
        if not scenario_results:
            logger.warning(f"No scenario results extracted for {model_name}")
            logger.info(f"Available test result keys: {list(test_results.keys()) if test_results else 'None'}")
            
            # 尝试提取任何包含MOA的指标作为fallback
            fallback_metrics = {}
            for key, value in test_results.items():
                if 'moa' in key.lower() and 'accuracy' in key.lower():
                    fallback_metrics['moa_accuracy'] = float(value)
                    break
            
            if fallback_metrics:
                scenario_results['no_missing'] = fallback_metrics
                logger.info(f"Using fallback metrics: {fallback_metrics}")
        
        # 组织最终结果，包含消融实验信息
        final_results = {
            'split_info': split_info,
            'ablation_info': {
                'model_name': model_name,
                'description': ablation_config['description'],
                'stage1_params_override': ablation_config.get('stage1_params_override', {}),
                'stage2_params_override': ablation_config.get('stage2_params_override', {})
            },
            'scenario_results': scenario_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return final_results
    
    def add_result(self, result: Dict[str, Any]):
        """添加一个实验结果"""
        
        split_info = result['split_info']
        ablation_info = result['ablation_info']
        scenario_results = result['scenario_results']
        
        # 存储实验配置
        experiment_key = f"{ablation_info['model_name']}_split_{split_info.get('split_index', 0)}"
        self.experiment_configs[experiment_key] = {
            'model_name': ablation_info['model_name'],
            'description': ablation_info['description'],
            'stage1_params_override': ablation_info.get('stage1_params_override', {}),
            'stage2_params_override': ablation_info.get('stage2_params_override', {}),
            'split_info': split_info
        }
        
        # 为每个场景创建一行记录
        for scenario, metrics in scenario_results.items():
            record = {
                # Split信息
                'split_strategy': split_info.get('split_strategy', 'unknown'),
                'split_index': split_info.get('split_index', 0),
                'split_seed': split_info.get('split_seed', None),
                
                # 消融实验信息
                'model_name': ablation_info['model_name'],
                'model_description': ablation_info['description'],
                'scenario': scenario,
                'timestamp': result.get('timestamp', ''),
                
                # MOA分类指标
                'moa_accuracy': metrics.get('moa_accuracy', None),
                'moa_f1_macro': metrics.get('moa_f1_macro', None),
                'moa_f1_weighted': metrics.get('moa_f1_weighted', None),
                'moa_precision_macro': metrics.get('moa_precision_macro', None),
                'moa_recall_macro': metrics.get('moa_recall_macro', None),
                
                # RNA重建指标
                'rna_mse': metrics.get('rna_mse', None),
                'rna_mae': metrics.get('rna_mae', None),
                'rna_r2': metrics.get('rna_r2', None),
                'rna_pearson': metrics.get('rna_pearson', None),
                'rna_spearman': metrics.get('rna_spearman', None),
                
                # 表型重建指标
                'pheno_mse': metrics.get('pheno_mse', None),
                'pheno_mae': metrics.get('pheno_mae', None),
                'pheno_r2': metrics.get('pheno_r2', None),
                'pheno_pearson': metrics.get('pheno_pearson', None),
                'pheno_spearman': metrics.get('pheno_spearman', None)
            }
            
            self.results.append(record)
        
        logger.info(f"Added results for {ablation_info['model_name']} - split {split_info.get('split_index', 0)} - {len(scenario_results)} scenarios")
    
    def save_results(self):
        """保存所有结果"""
        
        if not self.results:
            logger.warning("No results to save")
            return
        
        # 保存详细结果CSV
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_file, index=False)
        logger.info(f"Detailed results saved to: {self.results_file}")
        
        # 保存实验配置
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_configs, f, indent=2, ensure_ascii=False)
        logger.info(f"Experiment configs saved to: {self.config_file}")
        
        # 生成汇总报告
        self._generate_summary_report(df)
        
        return {
            'results_file': self.results_file,
            'summary_file': self.summary_file,
            'config_file': self.config_file,
            'num_models': len(df['model_name'].unique()),
            'num_splits': len(df['split_index'].unique()),
            'num_scenarios': len(df['scenario'].unique()),
            'total_records': len(df)
        }
    
    def _generate_summary_report(self, df: pd.DataFrame):
        """生成消融实验汇总报告"""
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"ABLATION EXPERIMENT SUMMARY REPORT\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Total models: {len(df['model_name'].unique())}\n")
            f.write(f"Total splits: {len(df['split_index'].unique())}\n")
            f.write(f"Total scenarios: {len(df['scenario'].unique())}\n")
            f.write(f"Total records: {len(df)}\n\n")
            
            # 列出所有消融模型
            f.write("ABLATION MODELS:\n")
            f.write("-" * 50 + "\n")
            for model_name in sorted(df['model_name'].unique()):
                model_df = df[df['model_name'] == model_name]
                description = model_df['model_description'].iloc[0]
                f.write(f"📦 {model_name}: {description}\n")
            f.write("\n")
            
            # 策略信息
            strategies = df['split_strategy'].unique()
            f.write(f"Split strategies: {list(strategies)}\n")
            for strategy in strategies:
                strategy_df = df[df['split_strategy'] == strategy]
                splits = strategy_df['split_index'].unique()
                f.write(f"  {strategy}: {len(splits)} splits ({min(splits)} to {max(splits)})\n")
            f.write("\n")
            
            # 按场景分析模型性能
            scenarios = ['no_missing', 'pheno_missing', 'rna_missing', 'both_missing']
            scenario_names = {
                'no_missing': 'No Missing (All modalities)',
                'pheno_missing': 'Phenotype Missing',
                'rna_missing': 'RNA Missing',
                'both_missing': 'Both Missing'
            }
            
            f.write("MODEL PERFORMANCE BY SCENARIO:\n")
            f.write("=" * 80 + "\n")
            
            for scenario in scenarios:
                if scenario in df['scenario'].values:
                    scenario_df = df[df['scenario'] == scenario]
                    f.write(f"\n📊 {scenario_names.get(scenario, scenario)}:\n")
                    f.write("-" * 50 + "\n")
                    
                    # MOA分类性能比较
                    if 'moa_accuracy' in scenario_df.columns and scenario_df['moa_accuracy'].notna().any():
                        f.write("🎯 MOA Classification Performance:\n")
                        
                        # 按模型汇总性能
                        model_performance = scenario_df.groupby('model_name').agg({
                            'moa_accuracy': ['mean', 'std', 'count'],
                            'moa_f1_macro': ['mean', 'std']
                        }).round(4)
                        
                        # 按准确率排序
                        model_performance = model_performance.sort_values(('moa_accuracy', 'mean'), ascending=False)
                        
                        for model_name in model_performance.index:
                            acc_mean = model_performance.loc[model_name, ('moa_accuracy', 'mean')]
                            acc_std = model_performance.loc[model_name, ('moa_accuracy', 'std')]
                            f1_mean = model_performance.loc[model_name, ('moa_f1_macro', 'mean')]
                            count = int(model_performance.loc[model_name, ('moa_accuracy', 'count')])
                            
                            f.write(f"  {model_name:<20}: Acc={acc_mean:.4f}±{acc_std:.4f}, "
                                   f"F1={f1_mean:.4f} (n={count})\n")
                        
                        # 找到最佳模型
                        best_model = model_performance.index[0]
                        best_acc = model_performance.loc[best_model, ('moa_accuracy', 'mean')]
                        f.write(f"  🏆 Best model: {best_model} (Acc: {best_acc:.4f})\n")
                    
                    # 重建性能比较（针对缺失场景）
                    if scenario != 'no_missing':
                        f.write("\n🔧 Reconstruction Performance:\n")
                        
                        # RNA重建性能
                        if 'rna_r2' in scenario_df.columns and scenario_df['rna_r2'].notna().any():
                            rna_performance = scenario_df.groupby('model_name')['rna_r2'].agg(['mean', 'std', 'count']).round(4)
                            rna_performance = rna_performance.sort_values('mean', ascending=False)
                            
                            f.write("  RNA Reconstruction (R²):\n")
                            for model_name in rna_performance.index:
                                mean_r2 = rna_performance.loc[model_name, 'mean']
                                std_r2 = rna_performance.loc[model_name, 'std']
                                count = int(rna_performance.loc[model_name, 'count'])
                                f.write(f"    {model_name:<18}: {mean_r2:.4f}±{std_r2:.4f} (n={count})\n")
                        
                        # Pheno重建性能
                        if 'pheno_r2' in scenario_df.columns and scenario_df['pheno_r2'].notna().any():
                            pheno_performance = scenario_df.groupby('model_name')['pheno_r2'].agg(['mean', 'std', 'count']).round(4)
                            pheno_performance = pheno_performance.sort_values('mean', ascending=False)
                            
                            f.write("  Pheno Reconstruction (R²):\n")
                            for model_name in pheno_performance.index:
                                mean_r2 = pheno_performance.loc[model_name, 'mean']
                                std_r2 = pheno_performance.loc[model_name, 'std']
                                count = int(pheno_performance.loc[model_name, 'count'])
                                f.write(f"    {model_name:<18}: {mean_r2:.4f}±{std_r2:.4f} (n={count})\n")
            
            # 消融组件影响分析
            f.write(f"\n\nABLATION COMPONENT IMPACT ANALYSIS:\n")
            f.write("=" * 50 + "\n")
            
            # 以完整模型为基线进行比较
            if 'FullModel' in df['model_name'].values:
                baseline_df = df[df['model_name'] == 'FullModel']
                baseline_performance = {}
                
                for scenario in scenarios:
                    if scenario in baseline_df['scenario'].values:
                        scenario_baseline = baseline_df[baseline_df['scenario'] == scenario]
                        if 'moa_accuracy' in scenario_baseline.columns and scenario_baseline['moa_accuracy'].notna().any():
                            baseline_performance[scenario] = scenario_baseline['moa_accuracy'].mean()
                
                f.write("MOA Classification Impact (vs FullModel baseline):\n")
                for scenario in scenarios:
                    if scenario in baseline_performance:
                        baseline_acc = baseline_performance[scenario]
                        f.write(f"\n{scenario_names.get(scenario, scenario)}:\n")
                        f.write(f"  Baseline (FullModel): {baseline_acc:.4f}\n")
                        
                        scenario_df = df[df['scenario'] == scenario]
                        for model_name in sorted(df['model_name'].unique()):
                            if model_name != 'FullModel':
                                model_df = scenario_df[scenario_df['model_name'] == model_name]
                                if not model_df.empty and 'moa_accuracy' in model_df.columns:
                                    model_acc = model_df['moa_accuracy'].mean()
                                    impact = baseline_acc - model_acc
                                    impact_pct = (impact / baseline_acc) * 100 if baseline_acc > 0 else 0
                                    
                                    status = "📈" if impact < 0 else "📉" if impact > 0.01 else "➡️"
                                    f.write(f"  {status} {model_name:<18}: {model_acc:.4f} "
                                           f"(Δ: {impact:+.4f}, {impact_pct:+.1f}%)\n")
            
            # 模型稳定性分析
            f.write(f"\n\nMODEL STABILITY ANALYSIS:\n")
            f.write("=" * 30 + "\n")
            
            for model_name in sorted(df['model_name'].unique()):
                model_df = df[df['model_name'] == model_name]
                f.write(f"\n{model_name}:\n")
                
                for scenario in scenarios:
                    scenario_model_df = model_df[model_df['scenario'] == scenario]
                    if len(scenario_model_df) > 1 and 'moa_accuracy' in scenario_model_df.columns:
                        acc_values = scenario_model_df['moa_accuracy'].dropna()
                        if len(acc_values) > 1:
                            cv = acc_values.std() / acc_values.mean() if acc_values.mean() > 0 else float('inf')
                            stability = 'High' if cv < 0.05 else 'Medium' if cv < 0.1 else 'Low'
                            f.write(f"  {scenario:<15}: CV={cv:.4f} ({stability})\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("📝 Notes:\n")
            f.write("- CV: Coefficient of Variation (std/mean) - lower is more stable\n")
            f.write("- Δ: Change compared to baseline (negative means improvement)\n")
            f.write("- All metrics are averaged across splits\n")
            f.write("=" * 100 + "\n")
        
        logger.info(f"Summary report saved to: {self.summary_file}")


def apply_config_override(base_config: Dict[str, Any], ablation_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    为两阶段训练应用消融实验配置覆盖
    
    Args:
        base_config: 基础配置字典
        ablation_config: 消融实验配置字典
        
    Returns:
        包含stage1和stage2配置的字典
    """
    
    def _recursive_update(base_dict, override_dict):
        result = copy.deepcopy(base_dict)
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _recursive_update(result[key], value)
            else:
                result[key] = value
        return result
    
    # 应用Stage1配置覆盖
    stage1_config = copy.deepcopy(base_config)
    if 'stage1_params_override' in ablation_config:
        stage1_config = _recursive_update(stage1_config, ablation_config['stage1_params_override'])
    
    # 应用Stage2配置覆盖
    stage2_config = copy.deepcopy(base_config)
    if 'stage2_params_override' in ablation_config:
        stage2_config = _recursive_update(stage2_config, ablation_config['stage2_params_override'])
    
    # 如果没有提供特定的stage配置，则基于模型配置推断
    if 'stage1_params_override' not in ablation_config and 'stage2_params_override' not in ablation_config:
        # 使用传统的params_override来生成两个阶段的配置
        if 'params_override' in ablation_config:
            override_config = ablation_config['params_override']
            stage1_config = _recursive_update(stage1_config, override_config)
            stage2_config = _recursive_update(stage2_config, override_config)
    
    return {
        'stage1_config': stage1_config,
        'stage2_config': stage2_config
    }


def train_single_ablation_split(base_config: Dict[str, Any], 
                               ablation_config: Dict[str, Any],
                               data_module: MOADataModule, 
                               output_dir: str, 
                               split_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    训练单个消融实验的单个split
    
    Args:
        base_config: 基础配置字典
        ablation_config: 消融实验配置
        data_module: 数据模块
        output_dir: 输出目录
        split_info: split信息
        
    Returns:
        训练结果，包含test_results
    """
    
    model_name = ablation_config['model_name']
    split_strategy = split_info.get('split_strategy', 'random')
    split_index = split_info.get('split_index', 0)
    split_seed = split_info.get('split_seed', None)
    
    logger.info(f"Training {model_name} - split {split_index} with {split_strategy} strategy (seed={split_seed})")
    
    # 应用消融配置覆盖
    configs = apply_config_override(base_config, ablation_config)
    stage1_config = configs['stage1_config']
    stage2_config = configs['stage2_config']
    
    # 创建实验特定的输出目录
    exp_output_dir = Path(output_dir) / f"{model_name}_split_{split_index}"
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打印split后的数据信息
    data_info = data_module.get_data_info()
    logger.info(f"Split {split_index} data info:")
    logger.info(f"  Train samples: {data_info.get('train_size', 0)}")
    logger.info(f"  Val samples: {data_info.get('val_size', 0)}")
    logger.info(f"  Test samples: {data_info.get('test_size', 0)}")
    
    # 保存实验配置
    exp_config_path = exp_output_dir / 'experiment_config.yaml'
    exp_config_to_save = {
        'base_config': base_config,
        'ablation_config': ablation_config,
        'stage1_config': stage1_config,
        'stage2_config': stage2_config,
        'split_info': split_info,
        'data_info': data_info
    }
    
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(exp_config_to_save, f, default_flow_style=False, allow_unicode=True)
    
    # 更新实验名称
    experiment_name = f"{model_name}_split_{split_index}_{split_strategy}"
    if split_seed is not None:
        experiment_name += f"_seed{split_seed}"
    
    # 使用train_moa_model进行两阶段训练
    logger.info(f"Starting two-stage training for {model_name}...")
    
    # 确定训练模式（sequential 或 independent）
    independent_training = stage1_config.get('training', {}).get('independent_training', False)
    use_stage1_weights = not independent_training
    
    logger.info(f"Training mode: {'Sequential' if use_stage1_weights else 'Independent'}")
    
    results = train_moa_model(
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        data_module=data_module,
        output_dir=str(exp_output_dir),
        experiment_name=experiment_name,
        cleanup_checkpoints=True,
        use_stage1_weights=use_stage1_weights
    )
    
    # 添加实验信息到结果中
    results['split_info'] = split_info
    results['ablation_config'] = ablation_config
    
    logger.info(f"{model_name} - split {split_index} training completed")
    
    return results


def batch_ablation_experiment(
    base_config: Dict[str, Any],
    data_config: Dict[str, Any], 
    output_dir: str,
    split_indices: List[int],
    split_strategy: str = 'random',
    ablation_names: Optional[List[str]] = None,
    include_custom: bool = False
) -> Dict[str, Any]:
    """
    批次运行消融实验
    
    Args:
        base_config: 基础配置字典
        data_config: 数据配置字典
        output_dir: 输出目录
        split_indices: split索引列表
        split_strategy: split策略
        ablation_names: 要运行的消融实验名称列表，None表示运行所有
        include_custom: 是否包含自定义消融实验
        
    Returns:
        批次实验结果
    """
    
    # 获取消融配置
    ablation_configs = AblationExperimentConfig.get_ablation_configs()
    if include_custom:
        custom_configs = AblationExperimentConfig.get_custom_ablation_configs()
        ablation_configs.update(custom_configs)
    
    # 过滤要运行的实验
    if ablation_names:
        ablation_configs = {name: config for name, config in ablation_configs.items() 
                          if name in ablation_names}
    
    logger.info(f"Starting ablation experiments with {len(split_indices)} splits")
    logger.info(f"Split strategy: {split_strategy}")
    logger.info(f"Split indices: {split_indices}")
    logger.info(f"Ablation experiments: {list(ablation_configs.keys())}")
    
    # 创建结果收集器
    result_collector = AblationResultCollector(output_dir, "AblationExperiment")
    
    # 运行每个消融实验
    total_experiments = len(ablation_configs) * len(split_indices)
    current_experiment = 0
    successful_experiments = 0
    failed_experiments = 0
    
    for ablation_name, ablation_config in ablation_configs.items():
        logger.info(f"\n{'='*100}")
        logger.info(f"RUNNING ABLATION EXPERIMENT: {ablation_name}")
        logger.info(f"Description: {ablation_config['description']}")
        logger.info(f"{'='*100}")
        
        for i, split_index in enumerate(split_indices):
            current_experiment += 1
            
            split_info = {
                'split_strategy': split_strategy,
                'split_index': split_index,
                'split_seed': data_config['random_seed']
            }
        
            logger.info(f"\n{'-'*80}")
            logger.info(f"EXPERIMENT {current_experiment}/{total_experiments}: {ablation_name} - Split {split_index}")
            logger.info(f"{'-'*80}")
            
            # 创建数据模块副本
            split_data_module = MOADataModule(
                data_dir=data_config['data_dir'],
                dataset_name=data_config['dataset_name'],
                batch_size=data_config.get('batch_size', 32),
                num_workers=data_config.get('num_workers', 4),
                pin_memory=data_config.get('pin_memory', True),
                split_strategy=split_strategy,
                train_split=data_config.get('train_split', 0.8),
                val_split=data_config.get('val_split', 0.1),
                test_split=data_config.get('test_split', 0.1),
                preload_features=data_config.get('preload_features', True),
                preload_metadata=data_config.get('preload_metadata', True),
                return_metadata=data_config.get('return_metadata', True),
                feature_groups_only=data_config.get('feature_groups_only', None),
                metadata_columns_only=data_config.get('metadata_columns_only', None),
                device=data_config.get('device', 'cpu'),
                moa_column=data_config.get('moa_column', 'Metadata_moa'),
                save_label_encoder=data_config.get('save_label_encoder', True),
                feature_group_mapping=data_config.get('feature_group_mapping', None),
                normalize_features=data_config.get('normalize_features', False),
                normalization_method=data_config.get('normalization_method', 'standardize'),
                exclude_modalities=data_config.get('exclude_modalities', None),
                save_scalers=data_config.get('save_scalers', True),
                random_seed=data_config.get('random_seed', 2025)
            )
            
            # 设置数据模块
            split_data_module.setup(split_index=split_index)
            
            # 训练消融实验
            split_result = train_single_ablation_split(
                base_config, ablation_config, split_data_module, output_dir, split_info
            )
            
            # 从训练结果中提取测试指标
            test_results = result_collector.extract_test_results_from_training(
                split_result, 
                split_info,
                ablation_config
            )
            
            # 添加到结果收集器
            result_collector.add_result(test_results)
            
            successful_experiments += 1
            logger.info(f"✅ {ablation_name} - Split {split_index} completed successfully")
            
            # 清理内存
            del split_result
            del test_results
            del split_data_module
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    
    # 保存所有结果
    logger.info(f"\n{'='*100}")
    logger.info("SAVING ABLATION EXPERIMENT RESULTS")
    logger.info(f"{'='*100}")
    
    save_info = result_collector.save_results()
    
    # 打印汇总信息
    logger.info(f"\n📊 ABLATION EXPERIMENT SUMMARY:")
    logger.info(f"  Total experiments: {total_experiments}")
    logger.info(f"  Successful: {successful_experiments}")
    logger.info(f"  Failed: {failed_experiments}")
    logger.info(f"  Success rate: {successful_experiments/total_experiments*100:.1f}%")
    logger.info(f"  Models tested: {len(ablation_configs)}")
    logger.info(f"  Splits per model: {len(split_indices)}")
    logger.info(f"  Results file: {save_info['results_file'].name}")
    logger.info(f"  Summary file: {save_info['summary_file'].name}")
    logger.info(f"  Config file: {save_info['config_file'].name}")
    
    return {
        'successful_experiments': successful_experiments,
        'failed_experiments': failed_experiments,
        'total_experiments': total_experiments,
        'success_rate': successful_experiments / total_experiments,
        'models_tested': len(ablation_configs),
        'splits_per_model': len(split_indices),
        'save_info': save_info,
        'output_dir': output_dir
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Ablation Experiment Batch Training for MultiModal MOA Prediction')
    # 基础参数CDRP-BBBC047-Bray/nvs_addnegcontrue  LINCS-Pilot1/nvs_negnormfalse_addnegcontrue
    # 基础参数
    parser.add_argument('--config', type=str, 
                       default='models/distangle_multimodal/config_distatngle_multimodal_moa_predictor.yaml',
                       help='Path to base config file')
    parser.add_argument('--data_dir', type=str,
                       default='preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue')
    parser.add_argument('--output_dir', type=str,
                       default='results_distangle/ablation_cdrp2',
                       help='Path to output directory')
    # 实验参数
    parser.add_argument('--split_indices', nargs='+', type=int, 
                       default=[0, 1, 2,3,4],# 3, 4
                       help='List of split indices to train')
    parser.add_argument('--split_strategy', type=str, default='plate',
                       choices=['random', 'scaffold', 'plate'],
                       help='Data split strategy')
    
    # 消融实验选择
    parser.add_argument('--ablation_names', nargs='+', type=str, default=None,
                       help='List of ablation experiment names to run (default: all)')
    parser.add_argument('--include_custom', default=False,
                       help='Include custom ablation experiments')
    parser.add_argument('--list_ablations', action='store_true',
                       help='List available ablation experiments and exit')
    
    # 其他参数
    parser.add_argument('--experiment_name', type=str, 
                       default='ablation_experiment',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # 列出可用的消融实验
    if args.list_ablations:
        ablation_configs = AblationExperimentConfig.get_ablation_configs()
        custom_configs = AblationExperimentConfig.get_custom_ablation_configs()
        
        print("\n📦 Available Ablation Experiments:")
        print("=" * 50)
        
        print("\n🔬 Standard Ablations:")
        for name, config in ablation_configs.items():
            print(f"  • {name:<20}: {config['description']}")
        
        print("\n🛠️  Custom Ablations:")
        for name, config in custom_configs.items():
            print(f"  • {name:<20}: {config['description']}")
        
        print(f"\nUsage example:")
        print(f"  python {sys.argv[0]} --ablation_names full_model no_reconstruction no_noise")
        return 0
    
    # 加载基础配置
    logger.info(f"Loading base config from: {args.config}")
    base_config = load_config(args.config)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存运行参数
    run_info = {
        'base_config_file': args.config,
        'data_dir': args.data_dir,
        'output_dir': str(output_dir),
        'split_indices': args.split_indices,
        'split_strategy': args.split_strategy,
        'ablation_names': args.ablation_names,
        'include_custom': args.include_custom,
        'experiment_name': args.experiment_name,
        'timestamp': timestamp,
        'command_line_args': vars(args)
    }
    
    run_info_file = output_dir / 'run_info.json'
    with open(run_info_file, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    logger.info(f"Run info saved to: {run_info_file}")
    
    # 准备数据配置
    data_config = base_config.get('data', {})
    pl.seed_everything(seed=data_config.get('random_seed', 2025), workers=True)
    data_config['data_dir'] = args.data_dir
    
    # 运行消融实验
    ablation_results = batch_ablation_experiment(
        base_config=base_config,
        data_config=data_config,
        output_dir=str(output_dir),
        split_indices=args.split_indices,
        split_strategy=args.split_strategy,
        ablation_names=args.ablation_names,
        include_custom=args.include_custom
    )
    
    logger.info(f"\n🎉 ABLATION EXPERIMENTS COMPLETED!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Success rate: {ablation_results['success_rate']*100:.1f}%")
    logger.info(f"Models tested: {ablation_results['models_tested']}")
    
    # 打印快速访问信息
    save_info = ablation_results['save_info']
    logger.info(f"\n📄 Quick Access Files:")
    logger.info(f"  Detailed results: {save_info['results_file'].name}")
    logger.info(f"  Summary report: {save_info['summary_file'].name}")
    logger.info(f"  Config archive: {save_info['config_file'].name}")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)