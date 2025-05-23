# 02_explore_datasets.py
"""
数据集探索脚本：了解我们的"原材料"
这个脚本帮助我们理解两个数据集的结构和内容
"""

import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json

class DatasetExplorer:
    """数据集探索器：帮助我们理解声音数据的结构"""
    
    def __init__(self, dataset_base_path):
        self.dataset_base_path = Path(dataset_base_path)
        
    def explore_urbansound8k(self):
        """探索UrbanSound8K数据集
        
        UrbanSound8K的特点：
        - 包含8732个音频片段
        - 10个声音类别（如狗叫、汽车喇叭等）
        - 每个音频最长4秒
        - 来自真实城市环境的录音
        """
        print("\n=== 探索 UrbanSound8K 数据集 ===\n")
        
        urbansound_path = self.dataset_base_path / "UrbanSound8K"
        
        # 检查数据集是否存在
        if not urbansound_path.exists():
            print("⚠️  UrbanSound8K 数据集未找到！")
            print(f"   请将数据集下载并解压到: {urbansound_path}")
            print("   下载地址: https://urbansounddataset.weebly.com/urbansound8k.html")
            return
        
        # 读取元数据文件
        metadata_file = urbansound_path / "metadata" / "UrbanSound8K.csv"
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file)
            
            print("数据集概览：")
            print(f"- 总音频数量: {len(metadata)}")
            print(f"- 音频存储在 {metadata['fold'].nunique()} 个文件夹中")
            
            print("\n声音类别分布：")
            class_names = {
                0: "air_conditioner (空调声)",
                1: "car_horn (汽车喇叭)",
                2: "children_playing (儿童玩耍)",
                3: "dog_bark (狗叫)",
                4: "drilling (钻孔声)",
                5: "engine_idling (引擎空转)",
                6: "gun_shot (枪声)",
                7: "jackhammer (手提钻)",
                8: "siren (警笛)",
                9: "street_music (街头音乐)"
            }
            
            for class_id, class_name in class_names.items():
                count = len(metadata[metadata['classID'] == class_id])
                print(f"  {class_id}: {class_name} - {count} 个样本")
            
            # 分析音频时长分布
            print("\n音频时长分析：")
            durations = metadata.groupby('classID')['end'].max()
            print(f"- 平均时长: {durations.mean():.2f} 秒")
            print(f"- 最短时长: {durations.min():.2f} 秒")
            print(f"- 最长时长: {durations.max():.2f} 秒")
            
            # 这些信息对我们很重要，因为它告诉我们：
            # 1. 这些是单一声音事件，不是完整的环境录音
            # 2. 我们需要思考如何将这些声音与我们的场景对应
            # 3. 某些声音（如狗叫）可能适合多个场景
            
            return metadata
    
    def explore_tau_dataset(self):
        """探索TAU Urban Acoustic Scenes 2019数据集
        
        TAU数据集的特点：
        - 包含多个城市的环境录音
        - 10个场景类别（如机场、购物中心、地铁站等）
        - 每个音频10秒长
        - 专门为场景分类任务设计
        """
        print("\n=== 探索 TAU Urban Acoustic Scenes 2019 数据集 ===\n")
        
        tau_path = self.dataset_base_path / "TAU-urban-acoustic-scenes-2019"
        
        if not tau_path.exists():
            print("⚠️  TAU 数据集未找到！")
            print(f"   请将数据集下载并解压到: {tau_path}")
            print("   下载地址: https://zenodo.org/record/2589280")
            return
        
        # TAU数据集通常有特定的目录结构
        # 让我们探索它的组织方式
        
        # 查找音频文件
        audio_files = list(tau_path.rglob("*.wav"))
        print(f"找到 {len(audio_files)} 个音频文件")
        
        # 分析场景类别
        # TAU的文件名通常包含场景信息
        scenes = {}
        for audio_file in audio_files[:100]:  # 先分析前100个文件
            # 文件名格式通常是: airport-barcelona-0-a.wav
            filename = audio_file.stem
            parts = filename.split('-')
            if len(parts) >= 2:
                scene = parts[0]
                if scene not in scenes:
                    scenes[scene] = 0
                scenes[scene] += 1
        
        print("\n检测到的场景类别：")
        tau_scene_descriptions = {
            'airport': '机场（人流密集，广播声）',
            'shopping_mall': '购物中心（人声嘈杂，背景音乐）',
            'metro_station': '地铁站（列车声，人群声）',
            'street_pedestrian': '步行街（脚步声，交谈声）',
            'public_square': '公共广场（开放空间，多样化声音）',
            'street_traffic': '交通街道（车辆声为主）',
            'tram': '有轨电车（轨道声，刹车声）',
            'bus': '公交车（引擎声，上下车声）',
            'metro': '地铁车厢内（轨道噪声，报站声）',
            'park': '公园（自然声，鸟鸣，远处城市声）'
        }
        
        for scene, description in tau_scene_descriptions.items():
            print(f"  - {scene}: {description}")
        
        # 这些场景与我们的13个目标场景有一定的对应关系
        # 让我展示如何建立这种对应关系
        
        print("\n\n场景映射建议：")
        print("TAU场景 → 您的场景分类")
        print("-" * 40)
        
        mapping_suggestions = {
            'park': ['A22', 'C22'],
            'street_pedestrian': ['C11', 'C12'],
            'street_traffic': ['C21'],
            'public_square': ['B3', 'C11'],
            'shopping_mall': ['B4', 'C12']
        }
        
        for tau_scene, your_scenes in mapping_suggestions.items():
            print(f"{tau_scene} → {', '.join(your_scenes)}")
        
        return scenes
    
    def analyze_audio_sample(self, audio_path):
        """分析单个音频文件的特征
        
        这个函数就像是对声音进行"体检"，
        帮助我们了解声音的各种特征
        """
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=None)
            
            # 计算基本特征
            duration = len(audio) / sr
            
            # 计算能量（响度）
            rms_energy = np.sqrt(np.mean(audio**2))
            
            # 计算频谱质心（声音的"亮度"）
            spectral_centroid = np.mean(
                librosa.feature.spectral_centroid(y=audio, sr=sr)
            )
            
            # 静音比例（安静部分的占比）
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'rms_energy': rms_energy,
                'spectral_centroid': spectral_centroid,
                'silence_ratio': silence_ratio
            }
            
        except Exception as e:
            print(f"分析音频时出错: {e}")
            return None

def main():
    """主函数：执行数据集探索"""
    
    # 创建探索器实例
    explorer = DatasetExplorer("datasets")
    
    # 探索两个数据集
    print("让我们开始探索手头的声音素材库...\n")
    
    urbansound_metadata = explorer.explore_urbansound8k()
    tau_scenes = explorer.explore_tau_dataset()
    
    print("\n\n=== 关键发现总结 ===")
    print("\n这两个数据集各有特点：")
    print("\n1. UrbanSound8K：")
    print("   - 提供单一声音事件（如狗叫、汽车喇叭）")
    print("   - 适合用于理解特定声音的特征")
    print("   - 可以帮助我们识别场景中的关键声音元素")
    
    print("\n2. TAU Urban ASC 2019：")
    print("   - 提供完整的场景录音")
    print("   - 更接近我们需要的"场景音频"")
    print("   - 但场景类别与我们的13个类别不完全对应")
    
    print("\n因此，我们的策略应该是：")
    print("- 优先从TAU中寻找合适的场景音频")
    print("- 使用UrbanSound8K来理解和验证关键声音特征")
    print("- 结合两者的优势来构建我们的数据集")

if __name__ == "__main__":
    main()
