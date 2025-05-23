# 04_intelligent_selector.py
"""
智能音频筛选器：将音频分配到正确的场景类别
这是整个系统的大脑，负责做出筛选决策
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf
from datetime import datetime

# 导入我们之前创建的分析器
from audio_analyzer import AudioFeatureAnalyzer

class SceneProfileManager:
    """场景档案管理器
    
    这个类定义了每个场景的"声学指纹"，
    就像为每个展厅制定的收藏标准
    """
    
    def __init__(self):
        self.profiles = self._create_scene_profiles()
    
    def _create_scene_profiles(self):
        """创建13个场景的详细档案
        
        每个档案都像是一份详细的"招聘要求"，
        描述了我们需要什么样的声音
        """
        profiles = {
            'A11': {
                'name': '旧的乡村建筑',
                'description': '城市边缘的老旧农村住宅，远离主要交通但能听到远处工业声',
                
                # 声学特征要求
                'acoustic_requirements': {
                    'overall_volume': 'low',           # 整体音量低
                    'sound_density': 'sparse',         # 声音稀疏
                    'frequency_focus': 'low_mid',      # 低中频为主
                    'human_activity': 'minimal',       # 人类活动少
                    'natural_sounds': 'prominent',     # 自然声音明显
                    'industrial_bg': 'distant'         # 远处工业背景
                },
                
                # 关键声音元素
                'key_sound_elements': {
                    'must_have': ['distant_machinery', 'wind', 'birds'],
                    'likely': ['dog_barking', 'rooster', 'footsteps'],
                    'avoid': ['heavy_traffic', 'crowds', 'music']
                },
                
                # 时间特征
                'temporal_patterns': {
                    'event_frequency': 'low',          # 声音事件频率低
                    'regularity': 'irregular',         # 不规则
                    'continuity': 'discontinuous'     # 不连续
                },
                
                # 与其他数据集的映射关系
                'dataset_mappings': {
                    'TAU': ['park', 'residential_area'],  # TAU中可能的对应场景
                    'UrbanSound8K': [3, 2],               # 狗叫、儿童玩耍
                },
                
                # 筛选权重（用于评分）
                'selection_weights': {
                    'low_frequency_ratio': 0.25,
                    'silence_ratio': 0.20,
                    'spectral_stability': 0.15,
                    'voice_activity_ratio': -0.20,    # 负权重表示人声越少越好
                    'traffic_noise_ratio': -0.20
                }
            },
            
            'B1': {
                'name': '单位房',
                'description': '规律的集体住宅生活，人口密度高但活动有序',
                
                'acoustic_requirements': {
                    'overall_volume': 'medium',
                    'sound_density': 'dense',
                    'frequency_focus': 'full_spectrum',
                    'human_activity': 'high',
                    'natural_sounds': 'minimal',
                    'industrial_bg': 'none'
                },
                
                'key_sound_elements': {
                    'must_have': ['human_voices', 'doors', 'footsteps'],
                    'likely': ['tv_sounds', 'cooking', 'children', 'phone_rings'],
                    'avoid': ['traffic', 'industrial', 'animals']
                },
                
                'temporal_patterns': {
                    'event_frequency': 'high',
                    'regularity': 'rhythmic',        # 有生活节奏
                    'continuity': 'continuous'
                },
                
                'dataset_mappings': {
                    'TAU': ['shopping_mall', 'metro_station'],
                    'UrbanSound8K': [2, 5],          # 儿童玩耍、引擎空转
                },
                
                'selection_weights': {
                    'voice_activity_ratio': 0.30,
                    'sound_continuity': 0.25,
                    'mid_frequency_ratio': 0.20,
                    'regularity_score': 0.15,
                    'silence_ratio': -0.10
                }
            },
            
            # ... 继续为其他11个场景创建类似的详细档案
            # 这里我展示两个例子，您需要根据场景特点完成其余的
        }
        
        # 让我们完成C11的档案作为另一个例子
        profiles['C11'] = {
            'name': '临街自建商业平房',
            'description': '底层商业顶层住宅的混合建筑，街道噪声与商业活动并存',
            
            'acoustic_requirements': {
                'overall_volume': 'high',
                'sound_density': 'very_dense',
                'frequency_focus': 'full_spectrum',
                'human_activity': 'very_high',
                'natural_sounds': 'none',
                'industrial_bg': 'none'
            },
            
            'key_sound_elements': {
                'must_have': ['traffic', 'human_voices', 'shop_sounds'],
                'likely': ['music', 'cash_register', 'doors', 'motorcycles'],
                'avoid': ['nature_sounds', 'silence']
            },
            
            'temporal_patterns': {
                'event_frequency': 'very_high',
                'regularity': 'chaotic',
                'continuity': 'continuous'
            },
            
            'dataset_mappings': {
                'TAU': ['street_traffic', 'street_pedestrian'],
                'UrbanSound8K': [1, 5, 8],     # 汽车喇叭、引擎、警笛
            },
            
            'selection_weights': {
                'traffic_noise_ratio': 0.25,
                'voice_activity_ratio': 0.25,
                'high_frequency_ratio': 0.20,
                'sound_continuity': 0.20,
                'silence_ratio': -0.10
            }
        }
        
        return profiles

class IntelligentAudioSelector:
    """智能音频筛选器的核心类
    
    这是我们的"声音品鉴大师"，
    能够准确判断每个音频属于哪个场景
    """
    
    def __init__(self, output_base_path):
        self.output_base_path = Path(output_base_path)
        self.scene_manager = SceneProfileManager()
        self.analyzer = AudioFeatureAnalyzer()
        self.selection_log = []
        
        # 创建元数据目录
        self.metadata_path = Path("metadata")
        self.metadata_path.mkdir(exist_ok=True)
        
    def calculate_scene_match_score(self, audio_features, scene_profile):
        """计算音频与场景的匹配分数
        
        这就像计算一个应聘者与职位要求的匹配度
        """
        score = 0
        score_details = {}
        
        # 1. 基于声学要求计算基础分数
        acoustic_score = self._calculate_acoustic_match(
            audio_features, 
            scene_profile['acoustic_requirements']
        )
        score += acoustic_score * 40  # 40分权重
        score_details['acoustic_match'] = acoustic_score
        
        # 2. 基于时间模式计算分数
        temporal_score = self._calculate_temporal_match(
            audio_features,
            scene_profile['temporal_patterns']
        )
        score += temporal_score * 30  # 30分权重
        score_details['temporal_match'] = temporal_score
        
        # 3. 基于加权特征计算分数
        weighted_score = self._calculate_weighted_features(
            audio_features,
            scene_profile['selection_weights']
        )
        score += weighted_score * 30  # 30分权重
        score_details['weighted_features'] = weighted_score
        
        return score, score_details
    
    def _calculate_acoustic_match(self, features, requirements):
        """计算声学特征匹配度"""
        match_score = 0
        matches = 0
        
        # 检查音量级别
        if requirements['overall_volume'] == 'low' and features['rms_energy'] < 0.02:
            matches += 1
        elif requirements['overall_volume'] == 'medium' and 0.02 <= features['rms_energy'] <= 0.1:
            matches += 1
        elif requirements['overall_volume'] == 'high' and features['rms_energy'] > 0.1:
            matches += 1
        
        # 检查声音密度
        if requirements['sound_density'] == 'sparse' and features['silence_ratio'] > 0.3:
            matches += 1
        elif requirements['sound_density'] == 'dense' and features['silence_ratio'] < 0.2:
            matches += 1
        elif requirements['sound_density'] == 'very_dense' and features['silence_ratio'] < 0.1:
            matches += 1
        
        # 检查人声活动
        if requirements['human_activity'] == 'minimal' and features['voice_activity_ratio'] < 0.1:
            matches += 1
        elif requirements['human_activity'] == 'high' and features['voice_activity_ratio'] > 0.3:
            matches += 1
        elif requirements['human_activity'] == 'very_high' and features['voice_activity_ratio'] > 0.5:
            matches += 1
        
        # 计算匹配百分比
        total_checks = 3  # 我们检查了3个方面
        match_score = matches / total_checks
        
        return match_score
    
    def _calculate_temporal_match(self, features, patterns):
        """计算时间模式匹配度"""
        match_score = 0
        
        # 检查事件频率
        onset_rate = features.get('onset_rate', 0)
        if patterns['event_frequency'] == 'low' and onset_rate < 2:
            match_score += 0.33
        elif patterns['event_frequency'] == 'high' and onset_rate > 5:
            match_score += 0.33
        elif patterns['event_frequency'] == 'very_high' and onset_rate > 10:
            match_score += 0.33
        
        # 检查连续性
        continuity = features.get('sound_continuity', 0)
        if patterns['continuity'] == 'discontinuous' and continuity < 0.3:
            match_score += 0.33
        elif patterns['continuity'] == 'continuous' and continuity > 0.7:
            match_score += 0.33
        
        # 检查规律性
        if patterns['regularity'] == 'irregular':
            # 不规则的声音应该有较高的频谱变化
            if features.get('spectral_centroid_std', 0) > 500:
                match_score += 0.34
        elif patterns['regularity'] == 'rhythmic':
            # 有节奏的声音应该有较低的频谱变化
            if features.get('spectral_centroid_std', 0) < 300:
                match_score += 0.34
        
        return match_score
    
    def _calculate_weighted_features(self, features, weights):
        """计算加权特征分数"""
        weighted_sum = 0
        weight_sum = 0
        
        for feature_name, weight in weights.items():
            if feature_name in features:
                # 归一化特征值到0-1范围
                feature_value = features[feature_name]
                if weight > 0:
                    # 正权重：特征值越大越好
                    weighted_sum += feature_value * abs(weight)
                else:
                    # 负权重：特征值越小越好
                    weighted_sum += (1 - feature_value) * abs(weight)
                
                weight_sum += abs(weight)
        
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return 0.5  # 默认中等分数
    
    def process_audio_file(self, audio_path, source_dataset):
        """处理单个音频文件
        
        这是筛选的核心流程，
        决定这个音频应该放在哪个"展厅"
        """
        try:
            # 提取音频特征
            features = self.analyzer.extract_comprehensive_features(audio_path)
            
            # 对每个场景计算匹配分数
            scene_scores = {}
            
            for scene_id, scene_profile in self.scene_manager.profiles.items():
                score, details = self.calculate_scene_match_score(features, scene_profile)
                scene_scores[scene_id] = {
                    'score': score,
                    'details': details,
                    'profile': scene_profile
                }
            
            # 找出最佳匹配的场景
            best_scene = max(scene_scores.items(), key=lambda x: x[1]['score'])
            best_scene_id = best_scene[0]
            best_score = best_scene[1]['score']
            
            # 设置阈值：只有分数超过60分才选择
            if best_score >= 60:
                self._select_audio(
                    audio_path, 
                    best_scene_id, 
                    best_score,
                    source_dataset,
                    features
                )
                return best_scene_id, best_score
            else:
                # 分数太低，不选择
                return None, best_score
                
        except Exception as e:
            print(f"处理音频时出错 {audio_path}: {e}")
            return None, 0
    
    def _select_audio(self, audio_path, scene_id, score, source_dataset, features):
        """选择音频并保存到对应的场景文件夹"""
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 如果音频长度超过4秒，选择最有代表性的4秒片段
        if len(audio) > sr * 4:
            # 找到能量最高的4秒片段
            best_start = self._find_best_segment(audio, sr, 4)
            audio = audio[best_start:best_start + sr * 4]
        else:
            # 如果短于4秒，填充静音
            audio = np.pad(audio, (0, max(0, sr * 4 - len(audio))))
        
        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_name = Path(audio_path).stem
        output_filename = f"{scene_id}_{int(score)}_{source_dataset}_{timestamp}_{original_name}.wav"
        
        # 保存音频
        output_dir = self.output_base_path / f"{scene_id}_samples"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_filename
        
        sf.write(str(output_path), audio, sr)
        
        # 记录选择信息
        selection_info = {
            'timestamp': timestamp,
            'source_file': str(audio_path),
            'source_dataset': source_dataset,
            'target_scene': scene_id,
            'match_score': score,
            'output_file': str(output_path),
            'features': features
        }
        
        self.selection_log.append(selection_info)
        
        # 实时保存日志
        self._save_selection_log()
        
        print(f"✓ 已选择: {Path(audio_path).name} → {scene_id} (分数: {score:.1f})")
    
    def _find_best_segment(self, audio, sr, duration_seconds):
        """找到音频中最有代表性的片段"""
        segment_length = sr * duration_seconds
        
        if len(audio) <= segment_length:
            return 0
        
        # 计算每个可能片段的能量
        max_energy = 0
        best_start = 0
        
        for start in range(0, len(audio) - segment_length, sr // 2):  # 每0.5秒滑动一次
            segment = audio[start:start + segment_length]
            energy = np.sum(segment ** 2)
            
            if energy > max_energy:
                max_energy = energy
                best_start = start
        
        return best_start
    
    def _save_selection_log(self):
        """保存筛选日志"""
        log_path = self.metadata_path / "selection_log.json"
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.selection_log, f, indent=2, ensure_ascii=False)
    
    def process_urbansound8k(self, dataset_path):
        """处理UrbanSound8K数据集
        
        UrbanSound8K主要提供单一声音事件，
        我们需要智能地判断这些声音适合哪些场景
        """
        print("\n=== 处理 UrbanSound8K 数据集 ===\n")
        
        dataset_path = Path(dataset_path)
        metadata_file = dataset_path / "metadata" / "UrbanSound8K.csv"
        
        if not metadata_file.exists():
            print("错误：找不到UrbanSound8K元数据文件")
            return
        
        # 读取元数据
        metadata = pd.read_csv(metadata_file)
        
        # 定义声音类别到场景的映射策略
        sound_to_scene_mapping = {
            0: [],  # air_conditioner - 可能适合B类场景
            1: ['C11', 'C12', 'C21'],  # car_horn - 适合交通场景
            2: ['A11', 'B1', 'B3'],     # children_playing - 适合住宅场景
            3: ['A11', 'A13', 'C22'],   # dog_bark - 适合郊区场景
            4: ['A12', 'A21'],          # drilling - 适合工业/建筑场景
            5: ['B2', 'B4', 'C21'],     # engine_idling - 适合城市场景
            6: [],                       # gun_shot - 特殊情况，可能不适用
            7: ['A21', 'C11'],          # jackhammer - 适合建筑场景
            8: ['C21', 'C12'],          # siren - 适合城市主干道
            9: ['B3', 'C11', 'C12']     # street_music - 适合商业区
        }
        
        # 处理每个音频文件
        processed = 0
        selected = 0
        
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="处理UrbanSound8K"):
            audio_path = dataset_path / "audio" / f"fold{row['fold']}" / row['slice_file_name']
            
            if not audio_path.exists():
                continue
            
            # 获取该声音类别可能的目标场景
            class_id = row['classID']
            potential_scenes = sound_to_scene_mapping.get(class_id, [])
            
            if not potential_scenes:
                continue
            
            # 对每个潜在场景进行评估
            best_scene = None
            best_score = 0
            
            for scene_id in potential_scenes:
                # 提取特征并计算匹配分数
                try:
                    features = self.analyzer.extract_comprehensive_features(str(audio_path))
                    scene_profile = self.scene_manager.profiles[scene_id]
                    score, _ = self.calculate_scene_match_score(features, scene_profile)
                    
                    if score > best_score:
                        best_score = score
                        best_scene = scene_id
                        
                except Exception as e:
                    continue
            
            # 如果找到合适的场景，选择这个音频
            if best_scene and best_score >= 50:  # UrbanSound8K的阈值可以低一些
                self._select_audio(
                    str(audio_path),
                    best_scene,
                    best_score,
                    'UrbanSound8K',
                    features
                )
                selected += 1
            
            processed += 1
        
        print(f"\n处理完成：处理了 {processed} 个文件，选择了 {selected} 个")
    
    def process_tau_dataset(self, dataset_path):
        """处理TAU Urban Acoustic Scenes数据集
        
        TAU提供完整的场景录音，
        更接近我们需要的目标
        """
        print("\n=== 处理 TAU Urban Acoustic Scenes 2019 数据集 ===\n")
        
        dataset_path = Path(dataset_path)
        
        # TAU场景到我们场景的映射
        tau_to_our_scenes = {
            'airport': [],  # 机场场景可能不太适用
            'bus': ['B4'],  # 公交车内部
            'metro': ['B4'],  # 地铁内部
            'metro_station': ['B4', 'C11'],  # 地铁站
            'park': ['A22', 'C22'],  # 公园
            'public_square': ['B3', 'C11'],  # 公共广场
            'shopping_mall': ['B4', 'C12'],  # 购物中心
            'street_pedestrian': ['C11', 'C12'],  # 步行街
            'street_traffic': ['C21', 'C12'],  # 交通街道
            'tram': ['C21']  # 有轨电车
        }
        
        # 查找所有音频文件
        audio_files = list(dataset_path.rglob("*.wav"))
        
        if not audio_files:
            print("错误：在TAU数据集中找不到音频文件")
            return
        
        print(f"找到 {len(audio_files)} 个音频文件")
        
        processed = 0
        selected = 0
        
        for audio_path in tqdm(audio_files, desc="处理TAU数据集"):
            # 从文件名提取场景类型
            filename = audio_path.stem
            tau_scene = filename.split('-')[0] if '-' in filename else None
            
            if not tau_scene or tau_scene not in tau_to_our_scenes:
                continue
            
            potential_scenes = tau_to_our_scenes[tau_scene]
            if not potential_scenes:
                continue
            
            # 对每个潜在场景进行评估
            best_scene = None
            best_score = 0
            
            for scene_id in potential_scenes:
                try:
                    features = self.analyzer.extract_comprehensive_features(str(audio_path))
                    scene_profile = self.scene_manager.profiles[scene_id]
                    score, _ = self.calculate_scene_match_score(features, scene_profile)
                    
                    if score > best_score:
                        best_score = score
                        best_scene = scene_id
                        
                except Exception as e:
                    continue
            
            # 如果找到合适的场景，选择这个音频
            if best_scene and best_score >= 65:  # TAU的阈值可以高一些
                self._select_audio(
                    str(audio_path),
                    best_scene,
                    best_score,
                    'TAU2019',
                    features
                )
                selected += 1
            
            processed += 1
        
        print(f"\n处理完成：处理了 {processed} 个文件，选择了 {selected} 个")
    
    def generate_selection_report(self):
        """生成筛选报告
        
        这个报告帮助我们了解筛选的结果，
        就像展览的清单
        """
        report_path = self.metadata_path / "selection_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 音频筛选报告 ===\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 统计每个场景的样本数
            scene_counts = {}
            for scene_id in self.scene_manager.profiles.keys():
                scene_dir = self.output_base_path / f"{scene_id}_samples"
                if scene_dir.exists():
                    count = len(list(scene_dir.glob("*.wav")))
                    scene_counts[scene_id] = count
                else:
                    scene_counts[scene_id] = 0
            
            f.write("场景样本统计:\n")
            f.write("-" * 50 + "\n")
            
            total_samples = 0
            for scene_id, count in sorted(scene_counts.items()):
                scene_name = self.scene_manager.profiles[scene_id]['name']
                f.write(f"{scene_id} ({scene_name}): {count} 个样本\n")
                total_samples += count
            
            f.write("-" * 50 + "\n")
            f.write(f"总计: {total_samples} 个样本\n\n")
            
            # 分析筛选日志
            if self.selection_log:
                f.write("筛选质量分析:\n")
                f.write("-" * 50 + "\n")
                
                # 计算平均分数
                scores_by_scene = {}
                for entry in self.selection_log:
                    scene = entry['target_scene']
                    score = entry['match_score']
                    
                    if scene not in scores_by_scene:
                        scores_by_scene[scene] = []
                    scores_by_scene[scene].append(score)
                
                for scene_id, scores in sorted(scores_by_scene.items()):
                    avg_score = np.mean(scores)
                    f.write(f"{scene_id}: 平均匹配分数 {avg_score:.1f}\n")
            
            f.write("\n筛选报告生成完毕！\n")
        
        print(f"\n报告已保存到: {report_path}")
        
        return scene_counts

def main():
    """主函数：执行完整的筛选流程"""
    
    print("=== 智能音频筛选系统 ===\n")
    
    # 创建筛选器
    selector = IntelligentAudioSelector("selected_samples")
    
    # 处理两个数据集
    print("开始处理数据集...\n")
    
    # 1. 处理UrbanSound8K
    urbansound_path = "datasets/UrbanSound8K"
    if Path(urbansound_path).exists():
        selector.process_urbansound8k(urbansound_path)
    else:
        print(f"⚠️  跳过UrbanSound8K：数据集不存在于 {urbansound_path}")
    
    # 2. 处理TAU数据集
    tau_path = "datasets/TAU-urban-acoustic-scenes-2019"
    if Path(tau_path).exists():
        selector.process_tau_dataset(tau_path)
    else:
        print(f"⚠️  跳过TAU数据集：数据集不存在于 {tau_path}")
    
    # 生成报告
    print("\n生成筛选报告...")
    scene_counts = selector.generate_selection_report()
    
    # 检查是否需要补充数据
    print("\n=== 数据充足性分析 ===")
    
    min_required = 300  # 每个场景最少需要的样本数
    need_more = []
    
    for scene_id, count in scene_counts.items():
        if count < min_required:
            need_more.append((scene_id, count))
    
    if need_more:
        print(f"\n以下场景需要更多数据（目标：每个场景至少{min_required}个样本）：")
        for scene_id, count in need_more:
            scene_name = selector.scene_manager.profiles[scene_id]['name']
            print(f"  - {scene_id} ({scene_name}): 当前{count}个，还需{min_required-count}个")
        
        print("\n建议：")
        print("1. 考虑降低这些场景的筛选阈值")
        print("2. 寻找更多相关的数据集")
        print("3. 使用数据增强技术扩充现有样本")
        print("4. 进行实地录音收集")
    else:
        print("\n✓ 所有场景都有足够的数据！可以开始训练了。")

if __name__ == "__main__":
    main()
