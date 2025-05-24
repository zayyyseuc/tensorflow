# 04_intelligent_selector_improved.py
"""
改进的智能音频筛选器：更精确的场景匹配
解决了原版本中某些场景无法获得音频的问题
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
from scipy import signal
from scipy.stats import kurtosis, skew

# 导入我们的音频分析器
from audio_analyzer import AudioFeatureAnalyzer

class EnhancedAudioFeatureAnalyzer(AudioFeatureAnalyzer):
    """增强的音频特征分析器
    
    添加了更多专门用于场景识别的特征
    """
    
    def extract_comprehensive_features(self, audio_path):
        """提取全面的音频特征，包括新增的环境特征"""
        
        # 首先获取基础特征
        features = super().extract_comprehensive_features(audio_path)
        
        # 加载音频用于额外分析
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 添加新的特征
        features.update(self._extract_environmental_features(audio, sr))
        features.update(self._extract_acoustic_scene_features(audio, sr))
        features.update(self._extract_sound_event_features(audio, sr))
        
        return features
    
    def _extract_environmental_features(self, audio, sr):
        """提取环境相关特征"""
        features = {}
        
        # 1. 检测水声（通过特定频率范围的能量模式）
        # 水声通常在500-2000Hz有持续的宽带噪声
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        water_freq_mask = (freqs >= 500) & (freqs <= 2000)
        water_energy = np.mean(stft[water_freq_mask, :])
        total_energy = np.mean(stft) + 1e-10
        
        # 水声的频谱平坦度较高
        water_band_flatness = np.mean(
            librosa.feature.spectral_flatness(
                S=stft[water_freq_mask, :]
            )
        )
        
        features['water_sound_likelihood'] = (water_energy / total_energy) * water_band_flatness
        
        # 2. 检测鸟鸣（高频周期性）
        # 鸟鸣通常在2000-8000Hz，有明显的调制
        bird_freq_mask = (freqs >= 2000) & (freqs <= 8000)
        bird_band_energy = stft[bird_freq_mask, :]
        
        # 计算高频段的时间变化性
        bird_temporal_variation = np.std(np.sum(bird_band_energy, axis=0))
        features['bird_sound_likelihood'] = bird_temporal_variation / (np.mean(bird_band_energy) + 1e-10)
        
        # 3. 检测风声（低频宽带噪声）
        wind_freq_mask = freqs <= 500
        wind_energy_ratio = np.sum(stft[wind_freq_mask, :]) / (np.sum(stft) + 1e-10)
        
        # 风声的频谱应该比较平坦
        wind_flatness = np.mean(
            librosa.feature.spectral_flatness(
                S=stft[wind_freq_mask, :]
            )
        )
        
        features['wind_sound_likelihood'] = wind_energy_ratio * wind_flatness
        
        # 4. 检测金属声/碰撞声（通过瞬态检测）
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        peaks = signal.find_peaks(onset_env, height=np.mean(onset_env) * 2)[0]
        
        # 金属声有尖锐的起始和快速衰减
        if len(peaks) > 0:
            # 分析峰值后的衰减速度
            decay_rates = []
            for peak in peaks[:10]:  # 分析前10个峰值
                if peak + 10 < len(onset_env):
                    decay = onset_env[peak] - onset_env[peak + 10]
                    decay_rates.append(decay)
            
            features['metallic_sound_likelihood'] = np.mean(decay_rates) if decay_rates else 0
        else:
            features['metallic_sound_likelihood'] = 0
        
        # 5. 环境混响程度（通过自相关分析）
        # 计算音频的自相关来估计混响
        autocorr = np.correlate(audio[:sr], audio[:sr], mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # 取正延迟部分
        
        # 找到第一个显著峰值（除了0延迟）
        peaks = signal.find_peaks(autocorr[sr//100:], height=0.1*autocorr[0])[0]
        
        if len(peaks) > 0:
            # 混响环境会有多个衰减的峰值
            features['reverb_amount'] = len(peaks) / 10.0  # 归一化
        else:
            features['reverb_amount'] = 0
        
        return features
    
    def _extract_acoustic_scene_features(self, audio, sr):
        """提取声学场景特征"""
        features = {}
        
        # 1. 远近感估计（通过高频衰减）
        # 远处的声音高频成分会衰减更多
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        low_band = stft[freqs < 1000, :].mean()
        high_band = stft[freqs > 4000, :].mean()
        
        features['distance_cue'] = low_band / (high_band + 1e-10)  # 值越大表示声源越远
        
        # 2. 空间开阔度（通过频谱扩散）
        spectral_spread = np.std(librosa.feature.spectral_centroid(y=audio, sr=sr))
        features['spatial_openness'] = spectral_spread / 1000  # 归一化
        
        # 3. 背景噪声稳定性
        # 稳定的背景噪声（如远处交通）vs 间歇性声音
        energy_envelope = librosa.feature.rms(y=audio)[0]
        features['background_stability'] = 1 / (np.std(energy_envelope) / np.mean(energy_envelope) + 1)
        
        # 4. 声音层次丰富度（通过频谱峰值数量）
        mean_spectrum = np.mean(stft, axis=1)
        peaks = signal.find_peaks(mean_spectrum, height=np.mean(mean_spectrum))[0]
        features['sound_layer_richness'] = len(peaks) / 20  # 归一化
        
        return features
    
    def _extract_sound_event_features(self, audio, sr):
        """提取声音事件特征"""
        features = {}
        
        # 1. 交通声特征（引擎的低频周期性）
        # 车辆引擎通常在50-200Hz有周期性
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        engine_band = stft[(freqs >= 50) & (freqs <= 200), :]
        
        # 计算该频段的周期性
        engine_periodicity = []
        for i in range(engine_band.shape[0]):
            autocorr = np.correlate(engine_band[i], engine_band[i], mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # 寻找周期性峰值
            peaks = signal.find_peaks(autocorr[10:100])[0]
            if len(peaks) > 0:
                engine_periodicity.append(1)
            else:
                engine_periodicity.append(0)
        
        features['traffic_presence'] = np.mean(engine_periodicity)
        
        # 2. 人类活动的多样性（通过中频段的时间变化）
        human_band = stft[(freqs >= 300) & (freqs <= 3000), :]
        
        # 计算时间窗口内的频谱变化
        spectral_variation = []
        window_size = sr // 4  # 0.25秒窗口
        hop_size = window_size // 2
        
        for i in range(0, human_band.shape[1] - window_size, hop_size):
            window = human_band[:, i:i+window_size]
            variation = np.std(window) / (np.mean(window) + 1e-10)
            spectral_variation.append(variation)
        
        features['human_activity_diversity'] = np.mean(spectral_variation) if spectral_variation else 0
        
        # 3. 工业/机械声检测（稳定的窄带噪声）
        # 寻找频谱中的稳定峰值
        mean_spectrum = np.mean(stft, axis=1)
        std_spectrum = np.std(stft, axis=1)
        
        # 稳定的窄带噪声会有低的标准差
        narrowband_score = []
        peaks = signal.find_peaks(mean_spectrum, height=np.mean(mean_spectrum)*2)[0]
        
        for peak in peaks:
            if peak < len(std_spectrum):
                stability = mean_spectrum[peak] / (std_spectrum[peak] + 1e-10)
                narrowband_score.append(stability)
        
        features['industrial_sound_presence'] = np.mean(narrowband_score) if narrowband_score else 0
        
        return features


class ImprovedSceneProfileManager:
    """改进的场景档案管理器
    
    使用更精确的匹配规则和多层次判断
    """
    
    def __init__(self):
        self.profiles = self._create_improved_profiles()
    
    def _create_improved_profiles(self):
        """创建改进的场景档案，使用更精确的判断标准"""
        
        profiles = {
            'A11': {
                'name': '旧的乡村建筑',
                'description': '低矮的围墙、空地、远处的高楼天际线。少量植物',
                'primary_features': {
                    # 主要特征：这些必须满足
                    'distance_cue': (3.0, 10.0),  # 远处声音为主
                    'silence_ratio': (0.3, 0.7),   # 中等静音比例
                    'wind_sound_likelihood': (0.2, 0.8),  # 有风声
                },
                'secondary_features': {
                    # 次要特征：满足越多越好
                    'bird_sound_likelihood': (0.1, 0.5),  # 可能有鸟鸣
                    'traffic_presence': (0.1, 0.3),  # 远处交通声
                    'human_activity_diversity': (0.0, 0.2),  # 极少人类活动
                    'spatial_openness': (0.5, 1.0),  # 开阔空间
                },
                'forbidden_features': {
                    # 禁止特征：不应该出现
                    'industrial_sound_presence': (0.5, 1.0),  # 不应有明显工业声
                    'water_sound_likelihood': (0.3, 1.0),  # 不应有水声
                },
                'matching_strategy': 'flexible',  # 使用灵活匹配策略
                'minimum_match_score': 25  # 降低最低匹配分数
            },
            
            'A12': {
                'name': '废弃的低密度工业厂房',
                'description': '低矮的围墙和建筑、远处的高楼天际线。少量植物',
                'primary_features': {
                    'reverb_amount': (0.3, 1.0),  # 有混响（空旷建筑）
                    'metallic_sound_likelihood': (0.2, 1.0),  # 金属碰撞声
                    'silence_ratio': (0.4, 0.8),  # 大量静音
                },
                'secondary_features': {
                    'wind_sound_likelihood': (0.3, 0.9),  # 风穿过建筑
                    'industrial_sound_presence': (0.1, 0.4),  # 远处工业背景
                    'distance_cue': (2.0, 8.0),  # 声音较远
                },
                'forbidden_features': {
                    'human_activity_diversity': (0.3, 1.0),  # 人类活动应该很少
                    'traffic_presence': (0.5, 1.0),  # 交通声不应太多
                },
                'matching_strategy': 'flexible',
                'minimum_match_score': 30
            },
            
            'A13': {
                'name': '建设中或刚开始建设的工厂',
                'description': '一层的平房建筑、电线杆、电线、路灯等基础设施',
                'primary_features': {
                    'onset_rate': (5.0, 30.0),  # 高事件频率
                    'industrial_sound_presence': (0.4, 1.0),  # 明显的工业声
                    'metallic_sound_likelihood': (0.3, 1.0),  # 金属撞击声
                },
                'secondary_features': {
                    'human_activity_diversity': (0.3, 0.8),  # 工人活动
                    'sound_continuity': (0.4, 0.8),  # 半连续的声音
                    'low_freq_ratio': (0.4, 0.8),  # 机械低频
                },
                'forbidden_features': {
                    'bird_sound_likelihood': (0.3, 1.0),  # 不应有太多自然声
                    'water_sound_likelihood': (0.2, 1.0),
                },
                'matching_strategy': 'standard',
                'minimum_match_score': 35
            },
            
            'A21': {
                'name': '已规划未发展完成的土地',
                'description': '低矮的建筑、完善的基础设施、宽阔平坦的道路',
                'primary_features': {
                    'silence_ratio': (0.3, 0.6),  # 中等静音
                    'background_stability': (0.4, 0.8),  # 稳定的背景
                    'distance_cue': (2.0, 6.0),  # 中远距离声音
                },
                'secondary_features': {
                    'traffic_presence': (0.2, 0.5),  # 适度交通
                    'spatial_openness': (0.6, 1.0),  # 开阔
                    'wind_sound_likelihood': (0.2, 0.6),
                },
                'forbidden_features': {
                    'human_activity_diversity': (0.5, 1.0),  # 人类活动不应太多
                    'industrial_sound_presence': (0.6, 1.0),
                },
                'matching_strategy': 'flexible',
                'minimum_match_score': 28
            },
            
            'A22': {
                'name': '空地和田野',
                'description': '人流稀少、无建筑物、不规则废弃物',
                'primary_features': {
                    'wind_sound_likelihood': (0.4, 1.0),  # 强风声
                    'bird_sound_likelihood': (0.3, 0.9),  # 鸟鸣虫鸣
                    'silence_ratio': (0.5, 0.9),  # 大量静音
                },
                'secondary_features': {
                    'spatial_openness': (0.7, 1.0),  # 非常开阔
                    'background_stability': (0.6, 1.0),  # 稳定的自然背景
                    'distance_cue': (4.0, 20.0),  # 声音很远
                },
                'forbidden_features': {
                    'traffic_presence': (0.3, 1.0),  # 交通声应该很少
                    'human_activity_diversity': (0.2, 1.0),
                    'industrial_sound_presence': (0.1, 1.0),
                },
                'matching_strategy': 'strict_natural',  # 严格的自然声匹配
                'minimum_match_score': 25
            },
            
            'B1': {
                'name': '单位房',
                'description': '6层左右的建筑、玻璃顶、道路拥挤、两侧停放车辆',
                'primary_features': {
                    'human_activity_diversity': (0.5, 1.0),  # 高人类活动
                    'voice_activity_ratio': (0.3, 0.8),  # 明显人声
                    'sound_continuity': (0.6, 0.95),  # 连续的声音
                },
                'secondary_features': {
                    'traffic_presence': (0.3, 0.7),  # 中等交通
                    'spatial_openness': (0.1, 0.4),  # 较封闭
                    'onset_rate': (8.0, 25.0),  # 频繁事件
                },
                'forbidden_features': {
                    'wind_sound_likelihood': (0.5, 1.0),  # 室内环境风声少
                    'bird_sound_likelihood': (0.4, 1.0),
                },
                'matching_strategy': 'standard',
                'minimum_match_score': 40
            },
            
            'B2': {
                'name': '塔楼高层商品房',
                'description': '20层以上的高层建筑、植被稀疏、道路狭窄',
                'primary_features': {
                    'distance_cue': (1.5, 5.0),  # 中距离（高层视角）
                    'wind_sound_likelihood': (0.3, 0.7),  # 高层风声
                    'background_stability': (0.5, 0.9),  # 稳定背景
                },
                'secondary_features': {
                    'human_activity_diversity': (0.2, 0.5),  # 中等人类活动
                    'traffic_presence': (0.3, 0.6),  # 远处交通
                    'reverb_amount': (0.2, 0.5),  # 建筑间回声
                },
                'forbidden_features': {
                    'industrial_sound_presence': (0.5, 1.0),
                    'metallic_sound_likelihood': (0.6, 1.0),
                },
                'matching_strategy': 'flexible',
                'minimum_match_score': 32
            },
            
            'B3': {
                'name': '沿河或公园低密度住宅区',
                'description': '层高5-12层、跨灯等设施完善、双向两车道',
                'primary_features': {
                    'water_sound_likelihood': (0.3, 1.0),  # 水声是关键
                    'bird_sound_likelihood': (0.2, 0.7),  # 自然声
                    'human_activity_diversity': (0.3, 0.6),  # 适度人类活动
                },
                'secondary_features': {
                    'spatial_openness': (0.5, 0.8),  # 半开阔
                    'traffic_presence': (0.2, 0.5),  # 适度交通
                    'background_stability': (0.4, 0.7),
                },
                'forbidden_features': {
                    'industrial_sound_presence': (0.4, 1.0),
                    'metallic_sound_likelihood': (0.5, 1.0),
                },
                'matching_strategy': 'water_priority',  # 优先匹配水声
                'minimum_match_score': 25
            },
            
            'B4': {
                'name': '临街商业区',
                'description': '1-2层低矮自建平房、人车流密集',
                'primary_features': {
                    'human_activity_diversity': (0.6, 1.0),  # 极高人类活动
                    'traffic_presence': (0.5, 1.0),  # 大量交通
                    'onset_rate': (10.0, 50.0),  # 极高事件率
                },
                'secondary_features': {
                    'sound_continuity': (0.7, 1.0),  # 持续不断
                    'voice_activity_ratio': (0.4, 0.9),  # 大量人声
                    'spectral_centroid_mean': (1500, 4000),  # 高频成分多
                },
                'forbidden_features': {
                    'silence_ratio': (0.3, 1.0),  # 不应有太多静音
                    'wind_sound_likelihood': (0.5, 1.0),
                },
                'matching_strategy': 'standard',
                'minimum_match_score': 40
            },
            
            'C11': {
                'name': '临街自建商业平房',
                'description': '1-2层低矮自建平房、远景能看到高层住宅塔楼',
                'primary_features': {
                    'traffic_presence': (0.4, 0.9),  # 明显交通
                    'human_activity_diversity': (0.5, 0.9),  # 高人类活动
                    'sound_layer_richness': (0.5, 1.0),  # 声音层次丰富
                },
                'secondary_features': {
                    'onset_rate': (8.0, 30.0),
                    'voice_activity_ratio': (0.3, 0.7),
                    'sound_continuity': (0.6, 0.9),
                },
                'forbidden_features': {
                    'water_sound_likelihood': (0.3, 1.0),
                    'bird_sound_likelihood': (0.4, 1.0),
                },
                'matching_strategy': 'standard',
                'minimum_match_score': 38
            },
            
            'C12': {
                'name': '主干道附近商业带',
                'description': '人口密集有建筑、平房旁、颜色不统一',
                'primary_features': {
                    'traffic_presence': (0.7, 1.0),  # 极高交通噪音
                    'sound_continuity': (0.8, 1.0),  # 几乎不间断
                    'low_freq_ratio': (0.5, 0.9),  # 大量低频（引擎声）
                },
                'secondary_features': {
                    'onset_rate': (15.0, 50.0),  # 极高事件率
                    'human_activity_diversity': (0.4, 0.8),
                    'spectral_centroid_mean': (800, 2500),  # 偏低频
                },
                'forbidden_features': {
                    'silence_ratio': (0.2, 1.0),  # 几乎没有静音
                    'bird_sound_likelihood': (0.2, 1.0),
                },
                'matching_strategy': 'traffic_priority',  # 优先匹配交通声
                'minimum_match_score': 30
            },
            
            'C21': {
                'name': '旧村住宅建筑',
                'description': '低矮老房、不连续绿化、电线杆与电缆',
                'primary_features': {
                    'bird_sound_likelihood': (0.3, 0.8),  # 明显的自然声
                    'silence_ratio': (0.3, 0.7),  # 较多静音
                    'spatial_openness': (0.5, 0.9),  # 开阔
                },
                'secondary_features': {
                    'distance_cue': (2.0, 8.0),  # 声音距离适中
                    'human_activity_diversity': (0.1, 0.4),  # 少量人类活动
                    'traffic_presence': (0.1, 0.3),  # 偶尔的车辆
                },
                'forbidden_features': {
                    'industrial_sound_presence': (0.4, 1.0),
                    'water_sound_likelihood': (0.5, 1.0),
                },
                'matching_strategy': 'rural_priority',  # 乡村特征优先
                'minimum_match_score': 28
            },
            
            'C22': {
                'name': '废弃厂房',
                'description': '人口稀少、无建筑物、不规则废弃物',
                'primary_features': {
                    'reverb_amount': (0.4, 1.0),  # 强混响（空旷）
                    'silence_ratio': (0.6, 0.95),  # 大量静音
                    'metallic_sound_likelihood': (0.2, 0.8),  # 金属碰撞
                },
                'secondary_features': {
                    'wind_sound_likelihood': (0.4, 0.9),  # 风声明显
                    'spatial_openness': (0.6, 1.0),  # 空旷
                    'background_stability': (0.6, 1.0),  # 稳定的环境声
                },
                'forbidden_features': {
                    'human_activity_diversity': (0.2, 1.0),  # 几乎无人
                    'traffic_presence': (0.3, 1.0),  # 远离交通
                    'voice_activity_ratio': (0.1, 1.0),  # 无人声
                },
                'matching_strategy': 'abandoned_priority',  # 废弃环境优先
                'minimum_match_score': 22
            }
        }
        
        return profiles


class IntelligentAudioSelectorV2:
    """改进版的智能音频筛选器
    
    使用多层次匹配策略和更灵活的判断标准
    """
    
    def __init__(self, output_base_path):
        self.output_base_path = Path(output_base_path)
        self.scene_manager = ImprovedSceneProfileManager()
        self.analyzer = EnhancedAudioFeatureAnalyzer()
        self.selection_log = []
        
        # 创建元数据目录
        self.metadata_path = Path("metadata")
        self.metadata_path.mkdir(exist_ok=True)
        
        # 创建所有场景的输出目录
        for scene_id in self.scene_manager.profiles.keys():
            scene_dir = self.output_base_path / f"{scene_id}_samples"
            scene_dir.mkdir(exist_ok=True, parents=True)
        
        # 记录每个场景的选择数量，用于平衡
        self.scene_selection_count = {scene_id: 0 for scene_id in self.scene_manager.profiles.keys()}
    
    def calculate_scene_match_score_v2(self, audio_features, scene_profile):
        """改进的场景匹配分数计算
        
        使用多层次判断和更灵活的匹配策略
        """
        score = 0
        details = {
            'primary_matches': 0,
            'secondary_matches': 0,
            'forbidden_violations': 0,
            'strategy_bonus': 0
        }
        
        # 1. 检查主要特征（权重40%）
        primary_score = 0
        primary_count = 0
        
        for feature_name, (min_val, max_val) in scene_profile['primary_features'].items():
            if feature_name in audio_features:
                value = audio_features[feature_name]
                if min_val <= value <= max_val:
                    primary_score += 1
                    details['primary_matches'] += 1
                # 部分匹配也给分
                elif min_val * 0.8 <= value <= max_val * 1.2:
                    primary_score += 0.5
                primary_count += 1
        
        if primary_count > 0:
            score += (primary_score / primary_count) * 40
        
        # 2. 检查次要特征（权重30%）
        secondary_score = 0
        secondary_count = 0
        
        for feature_name, (min_val, max_val) in scene_profile['secondary_features'].items():
            if feature_name in audio_features:
                value = audio_features[feature_name]
                if min_val <= value <= max_val:
                    secondary_score += 1
                    details['secondary_matches'] += 1
                elif min_val * 0.7 <= value <= max_val * 1.3:
                    secondary_score += 0.3
                secondary_count += 1
        
        if secondary_count > 0:
            score += (secondary_score / secondary_count) * 30
        
        # 3. 检查禁止特征（权重20%）
        forbidden_penalty = 0
        forbidden_count = 0
        
        for feature_name, (min_val, max_val) in scene_profile['forbidden_features'].items():
            if feature_name in audio_features:
                value = audio_features[feature_name]
                if min_val <= value <= max_val:
                    forbidden_penalty += 1
                    details['forbidden_violations'] += 1
                forbidden_count += 1
        
        if forbidden_count > 0:
            # 违反禁止特征会扣分
            score -= (forbidden_penalty / forbidden_count) * 20
        else:
            # 没有禁止特征则加分
            score += 20
        
        # 4. 应用特殊匹配策略（权重10%）
        strategy = scene_profile['matching_strategy']
        strategy_bonus = self._apply_matching_strategy(audio_features, scene_profile, strategy)
        score += strategy_bonus * 10
        details['strategy_bonus'] = strategy_bonus
        
        # 5. 场景平衡奖励
        # 给选择较少的场景额外加分，促进平衡
        scene_id = self._get_scene_id_from_profile(scene_profile)
        if scene_id:
            selection_ratio = self.scene_selection_count[scene_id] / (sum(self.scene_selection_count.values()) + 1)
            if selection_ratio < 0.05:  # 如果该场景选择比例低于5%
                balance_bonus = 10 * (1 - selection_ratio)
                score += balance_bonus
        
        return max(0, min(100, score)), details
    
    def _apply_matching_strategy(self, features, profile, strategy):
        """应用特殊的匹配策略"""
        
        if strategy == 'flexible':
            # 灵活策略：只要有一些关键特征就给高分
            key_features = ['distance_cue', 'silence_ratio', 'spatial_openness']
            matches = sum(1 for f in key_features if f in features and f in profile['primary_features'])
            return matches / len(key_features) if key_features else 0
            
        elif strategy == 'strict_natural':
            # 严格自然声策略：必须有强烈的自然特征
            natural_score = 0
            if 'bird_sound_likelihood' in features and features['bird_sound_likelihood'] > 0.3:
                natural_score += 0.5
            if 'wind_sound_likelihood' in features and features['wind_sound_likelihood'] > 0.4:
                natural_score += 0.5
            if 'traffic_presence' in features and features['traffic_presence'] < 0.2:
                natural_score += 0.5
            return min(1.0, natural_score)
            
        elif strategy == 'water_priority':
            # 水声优先策略：有水声就大幅加分
            if 'water_sound_likelihood' in features and features['water_sound_likelihood'] > 0.3:
                return 1.0
            return 0
            
        elif strategy == 'traffic_priority':
            # 交通声优先策略
            if 'traffic_presence' in features and features['traffic_presence'] > 0.7:
                return 1.0
            elif 'traffic_presence' in features and features['traffic_presence'] > 0.5:
                return 0.5
            return 0
            
        elif strategy == 'rural_priority':
            # 乡村特征优先
            rural_score = 0
            if 'bird_sound_likelihood' in features and features['bird_sound_likelihood'] > 0.3:
                rural_score += 0.3
            if 'silence_ratio' in features and features['silence_ratio'] > 0.3:
                rural_score += 0.3
            if 'human_activity_diversity' in features and features['human_activity_diversity'] < 0.4:
                rural_score += 0.4
            return rural_score
            
        elif strategy == 'abandoned_priority':
            # 废弃环境优先
            abandoned_score = 0
            if 'reverb_amount' in features and features['reverb_amount'] > 0.4:
                abandoned_score += 0.4
            if 'metallic_sound_likelihood' in features and features['metallic_sound_likelihood'] > 0.2:
                abandoned_score += 0.3
            if 'silence_ratio' in features and features['silence_ratio'] > 0.6:
                abandoned_score += 0.3
            return abandoned_score
            
        else:  # 'standard'
            return 0.5  # 标准策略给中等分数
    
    def _get_scene_id_from_profile(self, profile):
        """从profile找到对应的scene_id"""
        for scene_id, p in self.scene_manager.profiles.items():
            if p == profile:
                return scene_id
        return None
    
    def process_audio_file(self, audio_path, source_dataset):
        """处理单个音频文件，使用改进的匹配策略"""
        try:
            # 提取音频特征
            features = self.analyzer.extract_comprehensive_features(audio_path)
            
            # 对每个场景计算匹配分数
            scene_scores = {}
            
            for scene_id, scene_profile in self.scene_manager.profiles.items():
                score, details = self.calculate_scene_match_score_v2(features, scene_profile)
                scene_scores[scene_id] = {
                    'score': score,
                    'details': details,
                    'profile': scene_profile
                }
            
            # 排序场景分数
            sorted_scenes = sorted(scene_scores.items(), 
                                 key=lambda x: x[1]['score'], 
                                 reverse=True)
            
            # 选择最佳场景
            best_scene_id = None
            best_score = 0
            
            for scene_id, scene_data in sorted_scenes:
                score = scene_data['score']
                min_required = scene_data['profile']['minimum_match_score']
                
                # 检查是否满足最低要求
                if score >= min_required:
                    best_scene_id = scene_id
                    best_score = score
                    break
            
            # 如果没有场景满足最低要求，使用更宽松的标准
            if not best_scene_id and sorted_scenes:
                # 选择分数最高的场景，如果它至少达到了其最低要求的70%
                top_scene = sorted_scenes[0]
                scene_id = top_scene[0]
                score = top_scene[1]['score']
                min_required = top_scene[1]['profile']['minimum_match_score']
                
                if score >= min_required * 0.7:
                    best_scene_id = scene_id
                    best_score = score
            
            # 选择音频
            if best_scene_id:
                self._select_audio(
                    audio_path, 
                    best_scene_id, 
                    best_score,
                    source_dataset,
                    features
                )
                # 更新选择计数
                self.scene_selection_count[best_scene_id] += 1
                return best_scene_id, best_score
            else:
                return None, best_score if sorted_scenes else 0
                
        except Exception as e:
            print(f"处理音频时出错 {audio_path}: {e}")
            return None, 0
    
    def _select_audio(self, audio_path, scene_id, score, source_dataset, features):
        """选择音频并保存到对应的场景文件夹"""
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 如果音频长度超过4秒，选择最有代表性的4秒片段
        if len(audio) > sr * 4:
            # 根据场景特点选择片段
            best_start = self._find_best_segment_for_scene(audio, sr, 4, scene_id)
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
            'key_features': {
                k: v for k, v in features.items() 
                if k in ['water_sound_likelihood', 'bird_sound_likelihood', 
                        'traffic_presence', 'human_activity_diversity']
            }
        }
        
        self.selection_log.append(selection_info)
        
        # 实时保存日志
        self._save_selection_log()
        
        print(f"✓ 已选择: {Path(audio_path).name} → {scene_id} (分数: {score:.1f})")
    
    def _find_best_segment_for_scene(self, audio, sr, duration_seconds, scene_id):
        """根据场景特点找到最合适的音频片段"""
        
        segment_length = sr * duration_seconds
        if len(audio) <= segment_length:
            return 0
        
        # 根据场景类型使用不同的选择策略
        scene_profile = self.scene_manager.profiles[scene_id]
        
        if 'water_sound_likelihood' in scene_profile['primary_features']:
            # 对于需要水声的场景，找到水声特征最明显的片段
            return self._find_water_segment(audio, sr, segment_length)
        
        elif 'bird_sound_likelihood' in scene_profile['primary_features']:
            # 对于需要鸟鸣的场景，找到高频活动最多的片段
            return self._find_nature_segment(audio, sr, segment_length)
        
        elif 'traffic_presence' in scene_profile['primary_features']:
            # 对于交通场景，找到低频能量最高的片段
            return self._find_traffic_segment(audio, sr, segment_length)
        
        else:
            # 默认：找到能量适中的片段
            return self._find_balanced_segment(audio, sr, segment_length)
    
    def _find_water_segment(self, audio, sr, segment_length):
        """找到水声特征最明显的片段"""
        best_score = 0
        best_start = 0
        
        hop_length = sr // 2  # 每0.5秒检查一次
        
        for start in range(0, len(audio) - segment_length, hop_length):
            segment = audio[start:start + segment_length]
            
            # 计算水声特征（500-2000Hz的平坦噪声）
            stft = np.abs(librosa.stft(segment))
            freqs = librosa.fft_frequencies(sr=sr)
            water_band = stft[(freqs >= 500) & (freqs <= 2000), :]
            
            flatness = np.mean(librosa.feature.spectral_flatness(S=water_band))
            energy = np.mean(water_band)
            
            score = flatness * energy
            
            if score > best_score:
                best_score = score
                best_start = start
        
        return best_start
    
    def _find_nature_segment(self, audio, sr, segment_length):
        """找到自然声特征最明显的片段"""
        best_score = 0
        best_start = 0
        
        hop_length = sr // 2
        
        for start in range(0, len(audio) - segment_length, hop_length):
            segment = audio[start:start + segment_length]
            
            # 计算高频活动（鸟鸣通常在2000-8000Hz）
            stft = np.abs(librosa.stft(segment))
            freqs = librosa.fft_frequencies(sr=sr)
            high_band = stft[(freqs >= 2000) & (freqs <= 8000), :]
            
            # 高频的时间变化性
            temporal_variation = np.std(np.sum(high_band, axis=0))
            
            # 同时要求整体音量不要太大（自然环境通常较安静）
            overall_energy = np.mean(segment**2)
            
            score = temporal_variation / (overall_energy + 1e-10)
            
            if score > best_score:
                best_score = score
                best_start = start
        
        return best_start
    
    def _find_traffic_segment(self, audio, sr, segment_length):
        """找到交通声特征最明显的片段"""
        best_score = 0
        best_start = 0
        
        hop_length = sr // 2
        
        for start in range(0, len(audio) - segment_length, hop_length):
            segment = audio[start:start + segment_length]
            
            # 计算低频能量（交通声主要在低频）
            stft = np.abs(librosa.stft(segment))
            freqs = librosa.fft_frequencies(sr=sr)
            low_band = stft[freqs <= 500, :]
            
            low_energy = np.mean(low_band)
            
            # 持续性也很重要
            energy_std = np.std(librosa.feature.rms(y=segment)[0])
            continuity = 1 / (energy_std + 1e-10)
            
            score = low_energy * continuity
            
            if score > best_score:
                best_score = score
                best_start = start
        
        return best_start
    
    def _find_balanced_segment(self, audio, sr, segment_length):
        """找到能量平衡的片段（默认策略）"""
        # 计算每个片段的能量
        hop_length = sr // 2
        energies = []
        
        for start in range(0, len(audio) - segment_length, hop_length):
            segment = audio[start:start + segment_length]
            energy = np.sum(segment ** 2)
            energies.append((start, energy))
        
        # 选择能量接近中位数的片段
        energies.sort(key=lambda x: x[1])
        median_idx = len(energies) // 2
        
        return energies[median_idx][0]
    
    def _save_selection_log(self):
        """保存筛选日志"""
        log_path = self.metadata_path / "selection_log_v2.json"
        
        # 简化日志以避免文件过大
        simplified_log = []
        for entry in self.selection_log[-1000:]:  # 只保留最近1000条
            simplified_entry = {
                'timestamp': entry['timestamp'],
                'source_file': Path(entry['source_file']).name,
                'source_dataset': entry['source_dataset'],
                'target_scene': entry['target_scene'],
                'match_score': round(entry['match_score'], 1),
                'key_features': entry.get('key_features', {})
            }
            simplified_log.append(simplified_entry)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_log, f, indent=2, ensure_ascii=False)
    
    def process_urbansound8k(self, dataset_path):
        """处理UrbanSound8K数据集，使用改进的策略"""
        print("\n=== 处理 UrbanSound8K 数据集 ===\n")
        
        dataset_path = Path(dataset_path)
        metadata_file = dataset_path / "metadata" / "UrbanSound8K.csv"
        
        if not metadata_file.exists():
            print("错误：找不到UrbanSound8K元数据文件")
            return
        
        # 读取元数据
        metadata = pd.read_csv(metadata_file)
        
        # 改进的声音到场景映射
        # 基于实际声音特征而不是简单的类别映射
        sound_class_hints = {
            0: {'wind_sound_likelihood': 0.3},  # air_conditioner - 持续的背景噪声
            1: {'traffic_presence': 0.8},        # car_horn - 明显的交通标识
            2: {'human_activity_diversity': 0.7}, # children_playing - 人类活动
            3: {'bird_sound_likelihood': 0.2},   # dog_bark - 可能在乡村/郊区
            4: {'industrial_sound_presence': 0.8}, # drilling - 工业声
            5: {'traffic_presence': 0.6},        # engine_idling - 交通相关
            6: {},                              # gun_shot - 不使用
            7: {'industrial_sound_presence': 0.9}, # jackhammer - 建设声
            8: {'traffic_presence': 0.7},        # siren - 城市紧急声音
            9: {'human_activity_diversity': 0.6}  # street_music - 人类活动
        }
        
        processed = 0
        selected = 0
        
        # 可以限制处理数量以加快测试
        # metadata = metadata.head(2000)  # 测试时使用
        
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="处理UrbanSound8K"):
            audio_path = dataset_path / "audio" / f"fold{row['fold']}" / row['slice_file_name']
            
            if not audio_path.exists():
                continue
            
            # 获取该声音类别的提示
            class_id = row['classID']
            class_hints = sound_class_hints.get(class_id, {})
            
            # 处理音频文件
            scene_id, score = self.process_audio_file(str(audio_path), 'UrbanSound8K')
            
            if scene_id:
                selected += 1
            
            processed += 1
            
            # 定期报告进度
            if processed % 100 == 0:
                balance_info = ", ".join([f"{k}:{v}" for k, v in sorted(self.scene_selection_count.items())])
                print(f"已处理: {processed}, 已选择: {selected}")
                print(f"场景分布: {balance_info}")
        
        print(f"\n处理完成：处理了 {processed} 个文件，选择了 {selected} 个")
    
    def process_tau_dataset(self, dataset_path):
        """处理TAU数据集，使用改进的策略"""
        print("\n=== 处理 TAU Urban Acoustic Scenes 2019 数据集 ===\n")
        
        dataset_path = Path(dataset_path)
        
        # 查找所有音频文件
        audio_files = list(dataset_path.rglob("*.wav"))
        
        if not audio_files:
            print("错误：在TAU数据集中找不到音频文件")
            return
        
        print(f"找到 {len(audio_files)} 个音频文件")
        
        processed = 0
        selected = 0
        
        # 可以限制处理数量以加快测试
        # audio_files = audio_files[:1000]  # 测试时使用
        
        for audio_path in tqdm(audio_files, desc="处理TAU数据集"):
            # 处理音频文件
            scene_id, score = self.process_audio_file(str(audio_path), 'TAU2019')
            
            if scene_id:
                selected += 1
            
            processed += 1
            
            # 定期报告进度和平衡情况
            if processed % 50 == 0:
                # 找出数据最少的场景
                min_scenes = sorted(self.scene_selection_count.items(), 
                                  key=lambda x: x[1])[:5]
                print(f"已处理: {processed}, 已选择: {selected}")
                print(f"数据最少的场景: {', '.join([f'{k}:{v}' for k, v in min_scenes])}")
        
        print(f"\n处理完成：处理了 {processed} 个文件，选择了 {selected} 个")
    
    def generate_selection_report(self):
        """生成详细的筛选报告"""
        report_path = self.metadata_path / "selection_report_v2.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 改进版音频筛选报告 ===\n")
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
            f.write("-" * 60 + "\n")
            
            total_samples = 0
            for scene_id, count in sorted(scene_counts.items()):
                scene_name = self.scene_manager.profiles[scene_id]['name']
                total_samples += count
                status = "✓" if count >= 300 else "✗"
                f.write(f"{status} {scene_id} ({scene_name}): {count} 个样本\n")
            
            f.write("-" * 60 + "\n")
            f.write(f"总计: {total_samples} 个样本\n\n")
            
            # 数据平衡性分析
            f.write("数据平衡性分析:\n")
            f.write("-" * 60 + "\n")
            
            if total_samples > 0:
                avg_samples = total_samples / len(scene_counts)
                std_samples = np.std(list(scene_counts.values()))
                cv = std_samples / avg_samples  # 变异系数
                
                f.write(f"平均每场景: {avg_samples:.1f} 个样本\n")
                f.write(f"标准差: {std_samples:.1f}\n")
                f.write(f"变异系数: {cv:.2f} (越小越平衡)\n")
                
                # 计算基尼系数
                sorted_counts = sorted(scene_counts.values())
                n = len(sorted_counts)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
                f.write(f"基尼系数: {gini:.3f} (0=完全平衡, 1=完全不平衡)\n\n")
            
            # 改进效果分析
            f.write("改进效果:\n")
            f.write("-" * 60 + "\n")
            
            previously_empty = ['A11', 'A22', 'B3', 'C22']
            improvements = []
            
            for scene_id in previously_empty:
                count = scene_counts.get(scene_id, 0)
                if count > 0:
                    improvements.append(f"{scene_id}: 0 → {count}")
            
            if improvements:
                f.write("成功为之前没有数据的场景获取了样本:\n")
                for imp in improvements:
                    f.write(f"  {imp}\n")
            
            f.write("\n报告生成完毕！\n")
        
        print(f"\n报告已保存到: {report_path}")
        
        return scene_counts


def main():
    """主函数：执行改进的筛选流程"""
    
    print("=== 改进版智能音频筛选系统 V2 ===\n")
    print("主要改进：")
    print("1. 增加了环境特征检测（水声、鸟鸣、风声、金属声等）")
    print("2. 使用多层次匹配策略，降低了匹配阈值")
    print("3. 添加了场景平衡机制，优先填补数据不足的场景")
    print("4. 改进了音频片段选择算法\n")
    
    # 创建改进版筛选器
    selector = IntelligentAudioSelectorV2("selected_samples")
    
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
    
    # 给出最终建议
    print("\n=== 最终建议 ===")
    
    min_required = 300
    still_insufficient = []
    
    for scene_id, count in scene_counts.items():
        if count < min_required:
            still_insufficient.append((scene_id, count))
    
    if still_insufficient:
        print(f"\n仍有 {len(still_insufficient)} 个场景数据不足：")
        for scene_id, count in still_insufficient:
            scene_name = selector.scene_manager.profiles[scene_id]['name']
            print(f"  {scene_id} ({scene_name}): {count}/{min_required}")
        
        print("\n进一步的建议：")
        print("1. 考虑收集更多包含特定环境声音的数据集")
        print("2. 进行针对性的实地录音，特别是：")
        print("   - 包含水声的环境（河边、公园）")
        print("   - 纯自然环境（田野、空地）")
        print("   - 废弃建筑环境")
        print("3. 使用数据增强技术，如：")
        print("   - 添加环境混响模拟空旷环境")
        print("   - 混合自然声音样本")
        print("   - 调整音频的远近感")
    else:
        print("\n✓ 太棒了！所有场景都获得了足够的数据。")
        print("数据集已经准备好进行模型训练。")


if __name__ == "__main__":
    main()
