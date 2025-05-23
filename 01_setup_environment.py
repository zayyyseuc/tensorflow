# 01_setup_environment.py
"""
环境设置脚本：安装必要的库并验证环境
这就像在实验室中检查所有仪器是否正常工作
"""

import subprocess
import sys
import os

def setup_environment():
    """设置Python环境，安装必要的声音处理库"""
    
    print("=== 声景研究实验室环境设置 ===\n")
    
    # 需要安装的库及其用途说明
    required_packages = {
        'numpy': '数值计算的基础',
        'pandas': '数据组织和管理',
        'librosa': '音频处理',
        'soundfile': '音频文件读写工具',
        'matplotlib': '数据可视化',
        'scipy': '科学计算工具包',
        'scikit-learn': '机器学习工具',
        'tqdm': '进度条'
    }
    
    print("准备安装以下工具库：")
    for package, description in required_packages.items():
        print(f"  - {package}: {description}")
    
    print("\n开始安装...\n")
    
    # 执行安装
    for package in required_packages:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\n环境设置完成！")
    
    # 验证安装
    try:
        import numpy as np
        import pandas as pd
        import librosa
        import soundfile as sf
        import matplotlib.pyplot as plt
        import scipy
        import sklearn
        from tqdm import tqdm
        
        print("\n✓ 所有库都已成功安装！")
        print(f"  - NumPy 版本: {np.__version__}")
        print(f"  - Pandas 版本: {pd.__version__}")
        print(f"  - Librosa 版本: {librosa.__version__}")
        
        # 创建必要的目录结构
        print("\n创建项目目录结构...")
        directories = [
            'datasets/UrbanSound8K',
            'datasets/TAU-urban-acoustic-scenes-2019',
            'selected_samples',
            'metadata',
            'tools',
            'scripts'
        ]
        
        # 创建13个场景文件夹
        scene_folders = ['A11', 'A12', 'A13', 'A21', 'A22', 
                        'B1', 'B2', 'B3', 'B4',
                        'C11', 'C12', 'C21', 'C22']
        
        for scene in scene_folders:
            directories.append(f'selected_samples/{scene}_samples')
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("✓ 目录结构创建完成！")
        
    except ImportError as e:
        print(f"\n✗ 错误：{e}")
        print("请检查安装过程是否有错误提示")

if __name__ == "__main__":
    setup_environment()
