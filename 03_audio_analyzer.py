# audio_analyzer_enhanced.py
"""
增强版音频分析器：专门为13个场景设计的特征提取器
解决了原版本特征提取不精确、缺少环境特征的问题
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureAnalyzer:
    """增强版声音特征分析器
    
    这是一个更智能的"声音显微镜"，
    能够识别水声、鸟鸣、风声、混响等环境特征
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def extract_comprehensive_features(self, audio_path):
        """提取全面的音频特征
        
        包括基础特征和专门为13个场景设计的环境特征
        """
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        features = {}
        
        # 1. 基础特征
        features.update(self._extract_basic_features(audio, sr))
        
        # 2. 时域特征
        features.update(self._extract_temporal_features(audio))
        
        # 3. 频域特征
        features.update(self._extract_spectral_features(audio, sr))
        
        # 4. 梅尔频谱特征
        features.update(self._extract_mel_features(audio, sr))
        
        # 5. 节奏特征
        features.update(self._extract_rhythm_features(audio, sr))
        
        # 6. 环境声音特征（新增）
        features.update(self._extract_environmental_sounds(audio, sr))
        
        # 7. 空间特征（新增）
        features.update(self._extract_spatial_features(audio, sr))
        
        # 8. 人类活动特征（改进）
        features.update(self._extract_human_activity_features(audio, sr))
        
        # 9. 交通特征（新增）
        features.update(self._extract_traffic_features(audio, sr))
        
        # 10. 场景特定特征
        features.update(self._extract_scene_specific_features(audio, sr))
        
        return features
    
    def _extract_basic_features(self, audio, sr):
        """提取基础统计特征"""
        features = {}
        
        # 基本统计量
        features['audio_length'] = len(audio) / sr
        features['mean_amplitude'] = np.mean(np.abs(audio))
        features['std_amplitude'] = np.std(audio)
        features['max_amplitude'] = np.max(np.abs(audio))
        
        # 动态范围
        features['dynamic_range_db'] = 20 * np.log10(
            np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-10)
        )
        
        return features
    
    def _extract_temporal_features(self, audio):
        """提取时域特征"""
        features = {}
        
        # 能量相关
        features['rms_energy'] = np.sqrt(np.mean(audio**2))
        features['energy_variance'] = np.var(audio**2)
        features['energy_entropy'] = self._calculate_entropy(np.abs(audio))
        
        # 过零率
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zero_crossing_rate'] = np.mean(zcr)
        features['zcr_variance'] = np.var(zcr)
        
        # 静音检测（改进的多级阈值）
        max_amp = np.max(np.abs(audio))
        silence_threshold_low = 0.01 * max_amp
        silence_threshold_high = 0.05 * max_amp
        
        features['silence_ratio'] = np.sum(np.abs(audio) < silence_threshold_low) / len(audio)
        features['quiet_ratio'] = np.sum(np.abs(audio) < silence_threshold_high) / len(audio)
        
        # 声音的连续性（改进）
        envelope = np.abs(librosa.stft(audio)[:, 0])
        features['sound_continuity'] = 1 - np.std(envelope) / (np.mean(envelope) + 1e-10)
        
        # 声音的粗糙度
        features['roughness'] = np.mean(np.abs(np.diff(audio)))
        
        # 峰度和偏度（识别脉冲声）
        features['kurtosis'] = kurtosis(audio)
        features['skewness'] = abs(skew(audio))
        
        return features
    
    def _extract_spectral_features(self, audio, sr):
        """提取频谱特征"""
        features = {}
        
        # 计算STFT
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # 频谱质心
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        features['spectral_centroid_range'] = np.max(spectral_centroid) - np.min(spectral_centroid)
        
        # 频谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # 频谱滚降点
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # 频谱对比度
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_band_{i}'] = np.mean(spectral_contrast[i])
        
        # 频谱平坦度
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)
        
        # 频率分布（改进）
        total_energy = np.sum(stft) + 1e-10
        
        # 更精确的频段划分
        features['sub_bass_ratio'] = np.sum(stft[freqs < 60, :]) / total_energy  # <60Hz
        features['bass_ratio'] = np.sum(stft[(freqs >= 60) & (freqs < 250), :]) / total_energy  # 60-250Hz
        features['low_mid_ratio'] = np.sum(stft[(freqs >= 250) & (freqs < 500), :]) / total_energy  # 250-500Hz
        features['mid_ratio'] = np.sum(stft[(freqs >= 500) & (freqs < 2000), :]) / total_energy  # 500-2000Hz
        features['high_mid_ratio'] = np.sum(stft[(freqs >= 2000) & (freqs < 4000), :]) / total_energy  # 2000-4000Hz
        features['high_ratio'] = np.sum(stft[freqs >= 4000, :]) / total_energy  # >4000Hz
        
        # 聚合的低频和高频比例
        features['low_freq_ratio'] = features['sub_bass_ratio'] + features['bass_ratio'] + features['low_mid_ratio']
        features['high_freq_ratio'] = features['high_mid_ratio'] + features['high_ratio']
        
        # 频谱稳定性
        spectral_var = np.var(stft, axis=1)
        spectral_mean = np.mean(stft, axis=1) + 1e-10
        features['spectral_stability'] = np.mean(1 / (spectral_var / spectral_mean + 1))
        
        return features
    
    def _extract_mel_features(self, audio, sr):
        """提取梅尔频谱特征"""
        features = {}
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Delta MFCC（一阶差分）
        delta_mfccs = librosa.feature.delta(mfccs)
        for i in range(13):
            features[f'delta_mfcc_{i}_mean'] = np.mean(delta_mfccs[i])
        
        # 梅尔频谱
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        features['mel_energy_mean'] = np.mean(mel_spec)
        features['mel_energy_std'] = np.std(mel_spec)
        features['mel_energy_skew'] = skew(mel_spec.flatten())
        
        return features
    
    def _extract_rhythm_features(self, audio, sr):
        """提取节奏特征"""
        features = {}
        
        # 节拍跟踪
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = tempo
            features['beat_count'] = len(beats)
            features['beat_regularity'] = np.std(np.diff(beats)) if len(beats) > 1 else 0
        except:
            features['tempo'] = 0
            features['beat_count'] = 0
            features['beat_regularity'] = 0
        
        # 起始点检测
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features['onset_rate'] = np.sum(onset_env > np.mean(onset_env)) / (len(audio) / sr)
        features['onset_strength_mean'] = np.mean(onset_env)
        features['onset_strength_std'] = np.std(onset_env)
        
        # 节奏复杂度
        features['rhythm_complexity'] = np.std(onset_env) / (np.mean(onset_env) + 1e-10)
        
        return features
    
    def _extract_environmental_sounds(self, audio, sr):
        """提取环境声音特征（核心改进）"""
        features = {}
        
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # 1. 水声检测（关键特征）
        # 水声特征：500-2000Hz的持续宽带噪声，高频谱平坦度
        water_band_mask = (freqs >= 500) & (freqs <= 2000)
        water_band = stft[water_band_mask, :]
        
        if water_band.size > 0:
            # 水声的频谱平坦度
            water_flatness = np.mean(librosa.feature.spectral_flatness(S=water_band))
            
            # 水声的时间连续性
            water_continuity = 1 - np.std(np.mean(water_band, axis=0)) / (np.mean(water_band) + 1e-10)
            
            # 水声的频谱扩散度
            water_spread = np.std(water_band) / (np.mean(water_band) + 1e-10)
            
            # 综合水声似然度
            features['water_sound_likelihood'] = (water_flatness * 0.4 + 
                                                 water_continuity * 0.4 + 
                                                 water_spread * 0.2)
        else:
            features['water_sound_likelihood'] = 0
        
        # 2. 风声检测
        # 风声特征：低频宽带噪声，<500Hz，高平坦度
        wind_band_mask = freqs < 500
        wind_band = stft[wind_band_mask, :]
        
        if wind_band.size > 0:
            wind_energy_ratio = np.sum(wind_band) / (np.sum(stft) + 1e-10)
            wind_flatness = np.mean(librosa.feature.spectral_flatness(S=wind_band))
            
            # 风声的调制深度（风声通常有缓慢的幅度调制）
            wind_envelope = np.mean(wind_band, axis=0)
            wind_modulation = np.std(wind_envelope) / (np.mean(wind_envelope) + 1e-10)
            
            features['wind_sound_likelihood'] = wind_energy_ratio * wind_flatness * (1 - wind_modulation * 0.5)
        else:
            features['wind_sound_likelihood'] = 0
        
        # 3. 鸟鸣检测
        # 鸟鸣特征：2000-8000Hz的周期性调频信号
        bird_band_mask = (freqs >= 2000) & (freqs <= 8000)
        bird_band = stft[bird_band_mask, :]
        
        if bird_band.size > 0:
            # 高频能量的时间变化性
            bird_temporal_var = np.std(np.sum(bird_band, axis=0))
            
            # 频谱峰值的数量（鸟鸣通常有明显的谐波）
            bird_spectrum = np.mean(bird_band, axis=1)
            peaks, _ = signal.find_peaks(bird_spectrum, height=np.mean(bird_spectrum))
            bird_harmonicity = len(peaks) / bird_band.shape[0]
            
            # 调频特征（鸟鸣常有频率滑动）
            bird_centroid = np.array([np.sum(f * bird_band[i, :]) / (np.sum(bird_band[i, :]) + 1e-10) 
                                     for i, f in enumerate(freqs[bird_band_mask])])
            bird_freq_modulation = np.std(bird_centroid)
            
            features['bird_sound_likelihood'] = (bird_temporal_var * 0.3 + 
                                               bird_harmonicity * 0.4 + 
                                               bird_freq_modulation * 0.3) / 1000  # 归一化
        else:
            features['bird_sound_likelihood'] = 0
        
        # 4. 虫鸣检测
        # 虫鸣特征：高频窄带持续音
        insect_band_mask = (freqs >= 3000) & (freqs <= 7000)
        insect_band = stft[insect_band_mask, :]
        
        if insect_band.size > 0:
            # 窄带性（虫鸣通常是窄带的）
            insect_bandwidth = np.std(np.mean(insect_band, axis=1))
            insect_narrowband = 1 / (insect_bandwidth + 1)
            
            # 持续性
            insect_continuity = 1 - np.std(np.mean(insect_band, axis=0)) / (np.mean(insect_band) + 1e-10)
            
            features['insect_sound_likelihood'] = insect_narrowband * insect_continuity
        else:
            features['insect_sound_likelihood'] = 0
        
        # 5. 雨声检测
        # 雨声特征：全频段白噪声，高平坦度
        rain_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        rain_continuity = 1 - np.std(librosa.feature.rms(y=audio)[0]) / (np.mean(librosa.feature.rms(y=audio)[0]) + 1e-10)
        features['rain_sound_likelihood'] = rain_flatness * rain_continuity
        
        return features
    
    def _extract_spatial_features(self, audio, sr):
        """提取空间特征（新增）"""
        features = {}
        
        # 1. 混响量估计
        # 使用自相关分析
        # 限制自相关长度以提高效率
        max_lag = min(sr // 2, len(audio) // 4)  # 最多0.5秒
        audio_segment = audio[:max_lag * 2]
        
        autocorr = np.correlate(audio_segment, audio_segment[:max_lag], mode='valid')
        autocorr = autocorr / (autocorr[0] + 1e-10)  # 归一化
        
        # 寻找混响相关的峰值
        peaks, properties = signal.find_peaks(autocorr[sr//100:], height=0.1, distance=sr//50)
        
        if len(peaks) > 0:
            # 混响时间估计（基于峰值衰减）
            reverb_decay = np.polyfit(peaks[:5], properties['peak_heights'][:5], 1)[0] if len(peaks) >= 5 else 0
            features['reverb_amount'] = len(peaks) / 10  # 归一化峰值数量
            features['reverb_decay_rate'] = abs(reverb_decay)
        else:
            features['reverb_amount'] = 0
            features['reverb_decay_rate'] = 0
        
        # 2. 空间开阔度
        # 基于频谱扩散和高频衰减
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # 高频衰减率（开阔空间高频衰减更快）
        low_band_energy = np.mean(stft[freqs < 1000, :])
        high_band_energy = np.mean(stft[freqs > 4000, :])
        features['spatial_openness'] = low_band_energy / (high_band_energy + 1e-10) / 10  # 归一化
        
        # 3. 距离感估计
        # 远处的声音高频衰减更多，且可能有更多混响
        features['distance_cue'] = features['spatial_openness'] * (1 + features['reverb_amount'] * 0.5)
        
        # 4. 回声检测
        # 寻找明显的延迟重复
        echo_threshold = 0.3 * np.max(np.abs(audio))
        potential_echoes = autocorr[sr//10:sr//2]  # 0.1-0.5秒延迟
        echo_peaks = np.sum(potential_echoes > echo_threshold)
        features['echo_presence'] = echo_peaks / 10  # 归一化
        
        return features
    
    def _extract_human_activity_features(self, audio, sr):
        """提取人类活动特征（改进版）"""
        features = {}
        
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # 1. 改进的人声检测
        # 使用多个频段和特征组合
        
        # 基频范围（男声：85-180Hz，女声：165-255Hz）
        fundamental_band_mask = (freqs >= 85) & (freqs <= 255)
        fundamental_energy = np.mean(stft[fundamental_band_mask, :])
        
        # 第一共振峰范围（700-1220Hz）
        formant1_mask = (freqs >= 700) & (freqs <= 1220)
        formant1_energy = np.mean(stft[formant1_mask, :])
        
        # 人声谐波结构检测
        voice_band_mask = (freqs >= 85) & (freqs <= 3000)
        voice_band = stft[voice_band_mask, :]
        
        # 计算谐波性（人声有明显的谐波结构）
        voice_harmonicity = 0
        if voice_band.size > 0:
            for t in range(0, voice_band.shape[1], 10):  # 每10帧检查一次
                frame = voice_band[:, t]
                peaks, _ = signal.find_peaks(frame, height=np.mean(frame))
                if len(peaks) > 2:
                    # 检查峰值间距是否呈谐波关系
                    peak_distances = np.diff(peaks)
                    if len(peak_distances) > 0:
                        harmonicity = np.std(peak_distances) / (np.mean(peak_distances) + 1e-10)
                        voice_harmonicity += (1 - harmonicity) if harmonicity < 1 else 0
            
            voice_harmonicity /= (voice_band.shape[1] / 10)
        
        # 语音的时间调制特征（4-16Hz调制对应音节率）
        voice_envelope = np.mean(voice_band, axis=0) if voice_band.size > 0 else np.array([0])
        if len(voice_envelope) > sr // 4:  # 至少0.25秒
            # 计算调制频谱
            mod_fft = np.abs(np.fft.fft(voice_envelope))
            mod_freqs = np.fft.fftfreq(len(voice_envelope), 1/(sr/512))  # 512是hop_length
            syllable_rate_mask = (mod_freqs >= 4) & (mod_freqs <= 16)
            syllable_modulation = np.mean(mod_fft[syllable_rate_mask]) if np.any(syllable_rate_mask) else 0
        else:
            syllable_modulation = 0
        
        # 综合人声活动度
        features['voice_activity_ratio'] = (fundamental_energy * 0.2 + 
                                           formant1_energy * 0.3 + 
                                           voice_harmonicity * 0.3 + 
                                           syllable_modulation * 0.2) / 1000  # 归一化
        
        # 2. 人类活动多样性
        # 基于中频段的时频变化模式
        human_activity_band_mask = (freqs >= 200) & (freqs <= 4000)
        human_band = stft[human_activity_band_mask, :]
        
        if human_band.size > 0:
            # 时间维度的变化性
            temporal_diversity = np.std(np.mean(human_band, axis=0))
            
            # 频率维度的变化性
            spectral_diversity = np.mean(np.std(human_band, axis=1))
            
            # 局部特征变化（检测不同类型的人类活动）
            window_size = sr // 2  # 0.5秒窗口
            hop_size = window_size // 2
            
            local_variations = []
            for i in range(0, human_band.shape[1] - window_size, hop_size):
                window = human_band[:, i:i+window_size]
                local_var = np.std(window) / (np.mean(window) + 1e-10)
                local_variations.append(local_var)
            
            activity_diversity = np.mean(local_variations) if local_variations else 0
            
            features['human_activity_diversity'] = (temporal_diversity * 0.3 + 
                                                  spectral_diversity * 0.3 + 
                                                  activity_diversity * 0.4) / 1000  # 归一化
        else:
            features['human_activity_diversity'] = 0
        
        # 3. 脚步声检测
        # 脚步声特征：低频脉冲（50-200Hz），规律的间隔
        footstep_band_mask = (freqs >= 50) & (freqs <= 200)
        footstep_band = stft[footstep_band_mask, :]
        
        if footstep_band.size > 0:
            footstep_envelope = np.mean(footstep_band, axis=0)
            
            # 检测脉冲
            peaks, _ = signal.find_peaks(footstep_envelope, height=np.mean(footstep_envelope) * 2)
            
            if len(peaks) > 2:
                # 检查间隔规律性（脚步通常是规律的）
                intervals = np.diff(peaks)
                regularity = 1 - np.std(intervals) / (np.mean(intervals) + 1e-10) if len(intervals) > 0 else 0
                features['footstep_likelihood'] = regularity * len(peaks) / 100  # 归一化
            else:
                features['footstep_likelihood'] = 0
        else:
            features['footstep_likelihood'] = 0
        
        return features
    
    def _extract_traffic_features(self, audio, sr):
        """提取交通相关特征（新增）"""
        features = {}
        
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # 1. 引擎声检测
        # 引擎特征：50-200Hz的周期性，可能有谐波
        engine_band_mask = (freqs >= 50) & (freqs <= 200)
        engine_band = stft[engine_band_mask, :]
        
        if engine_band.size > 0:
            # 检测周期性
            engine_periodicity = []
            for i in range(engine_band.shape[0]):
                row = engine_band[i, :]
                if len(row) > 100:
                    autocorr = np.correlate(row, row[:len(row)//2], mode='valid')
                    autocorr = autocorr / (autocorr[0] + 1e-10)
                    
                    # 寻找周期性峰值
                    peaks, _ = signal.find_peaks(autocorr[10:100])
                    if len(peaks) > 0:
                        engine_periodicity.append(1)
                    else:
                        engine_periodicity.append(0)
            
            engine_presence = np.mean(engine_periodicity) if engine_periodicity else 0
            
            # 引擎的稳定性（持续的引擎声比较稳定）
            engine_stability = 1 - np.std(np.mean(engine_band, axis=0)) / (np.mean(engine_band) + 1e-10)
            
            features['engine_sound_presence'] = engine_presence * engine_stability
        else:
            features['engine_sound_presence'] = 0
        
        # 2. 轮胎噪声检测
        # 轮胎噪声：500-2000Hz的宽带噪声
        tire_band_mask = (freqs >= 500) & (freqs <= 2000)
        tire_band = stft[tire_band_mask, :]
        
        if tire_band.size > 0:
            # 宽带特性
            tire_flatness = np.mean(librosa.feature.spectral_flatness(S=tire_band))
            
            # 与速度相关的调制
            tire_envelope = np.mean(tire_band, axis=0)
            tire_modulation = np.std(tire_envelope) / (np.mean(tire_envelope) + 1e-10)
            
            features['tire_noise_presence'] = tire_flatness * (1 - tire_modulation * 0.5)
        else:
            features['tire_noise_presence'] = 0
        
        # 3. 喇叭声检测
        # 喇叭声：通常在300-3000Hz，有明显的基频和谐波
        horn_band_mask = (freqs >= 300) & (freqs <= 3000)
        horn_band = stft[horn_band_mask, :]
        
        if horn_band.size > 0:
            # 检测短时高能量事件
            horn_envelope = np.max(horn_band, axis=0)
            horn_threshold = np.mean(horn_envelope) * 3
            horn_events = horn_envelope > horn_threshold
            
            # 喇叭声通常持续0.5-2秒
            min_duration = int(0.5 * sr / 512)  # 512是hop_length
            max_duration = int(2.0 * sr / 512)
            
            # 检测持续时间合适的事件
            horn_likelihood = 0
            if np.any(horn_events):
                # 使用形态学操作找到连续区域
                from scipy import ndimage
                labeled, num_features = ndimage.label(horn_events)
                
                for i in range(1, num_features + 1):
                    event_length = np.sum(labeled == i)
                    if min_duration <= event_length <= max_duration:
                        horn_likelihood += 1
            
            features['horn_sound_likelihood'] = min(horn_likelihood / 5, 1)  # 归一化
        else:
            features['horn_sound_likelihood'] = 0
        
        # 4. 综合交通存在度
        features['traffic_presence'] = (features['engine_sound_presence'] * 0.4 +
                                       features['tire_noise_presence'] * 0.4 +
                                       features['horn_sound_likelihood'] * 0.2)
        
        # 5. 交通密度估计
        # 基于低频能量的连续性和变化
        traffic_band = stft[freqs < 1000, :]
        if traffic_band.size > 0:
            traffic_continuity = 1 - np.std(np.mean(traffic_band, axis=0)) / (np.mean(traffic_band) + 1e-10)
            traffic_variation = np.std(np.sum(traffic_band, axis=0))
            
            # 高密度交通：连续且有变化
            features['traffic_density'] = traffic_continuity * (1 + traffic_variation / 1000)
        else:
            features['traffic_density'] = 0
        
        return features
    
    def _extract_scene_specific_features(self, audio, sr):
        """提取场景特定特征"""
        features = {}
        
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # 1. 工业声检测
        # 工业声特征：稳定的窄带噪声，可能有机械周期性
        industrial_score = []
        
        # 寻找窄带稳定噪声
        mean_spectrum = np.mean(stft, axis=1)
        std_spectrum = np.std(stft, axis=1)
        
        # 找到能量峰值
        peaks, properties = signal.find_peaks(mean_spectrum, height=np.mean(mean_spectrum) * 2)
        
        for peak in peaks:
            if peak < len(std_spectrum):
                # 稳定性 = 平均值 / 标准差
                stability = mean_spectrum[peak] / (std_spectrum[peak] + 1e-10)
                
                # 窄带性 = 峰值能量 / 周围能量
                start = max(0, peak - 5)
                end = min(len(mean_spectrum), peak + 5)
                narrowband = mean_spectrum[peak] / (np.mean(mean_spectrum[start:end]) + 1e-10)
                
                industrial_score.append(stability * narrowband)
        
        features['industrial_sound_presence'] = np.mean(industrial_score) / 100 if industrial_score else 0
        
        # 2. 机械声周期性
        # 检测低频段的强周期性（机械通常有固定转速）
        mechanical_band = stft[freqs < 500, :]
        
        mechanical_periodicity = 0
        if mechanical_band.size > 0:
            # 检查时间序列的周期性
            for i in range(min(10, mechanical_band.shape[0])):  # 检查前10个频率
                row = mechanical_band[i, :]
                if len(row) > 200:
                    # 计算自相关
                    autocorr = np.correlate(row[:1000], row[:500], mode='valid') if len(row) > 1000 else [0]
                    autocorr = autocorr / (autocorr[0] + 1e-10) if len(autocorr) > 0 else [0]
                    
                    # 寻找强周期性
                    if len(autocorr) > 100:
                        peaks, heights = signal.find_peaks(autocorr[20:200], height=0.5)
                        if len(peaks) > 2:
                            # 检查峰值间隔的规律性
                            intervals = np.diff(peaks)
                            if len(intervals) > 0:
                                regularity = 1 - np.std(intervals) / (np.mean(intervals) + 1e-10)
                                mechanical_periodicity += regularity
            
            features['mechanical_periodicity'] = mechanical_periodicity / 10
        else:
            features['mechanical_periodicity'] = 0
        
        # 3. 金属碰撞声检测
        # 金属声特征：宽频瞬态 + 快速衰减 + 可能的共振
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        peaks, properties = signal.find_peaks(onset_env, height=np.mean(onset_env) * 3)
        
        metallic_score = 0
        if len(peaks) > 0:
            for peak in peaks[:20]:  # 检查前20个峰值
                if peak + 20 < len(onset_env):
                    # 计算衰减速度
                    decay_rate = (onset_env[peak] - onset_env[peak + 20]) / onset_env[peak]
                    
                    # 金属声衰减快
                    if decay_rate > 0.5:
                        # 检查是否有高频成分（金属声通常有）
                        peak_time = peak * 512 / sr  # 转换为时间
                        peak_frame = int(peak_time * sr / 512)
                        
                        if peak_frame < stft.shape[1]:
                            high_freq_energy = np.mean(stft[freqs > 2000, peak_frame])
                            total_energy = np.mean(stft[:, peak_frame]) + 1e-10
                            high_freq_ratio = high_freq_energy / total_energy
                            
                            if high_freq_ratio > 0.3:
                                metallic_score += 1
        
        features['metallic_sound_likelihood'] = min(metallic_score / 10, 1)
        
        # 4. 背景声稳定性
        # 用于区分稳定的城市背景声和变化的场景
        background_band = stft[freqs < 2000, :]
        
        if background_band.size > 0:
            # 计算长时间窗口的稳定性
            window_size = min(background_band.shape[1] // 4, sr // 512)  # 0.25秒或1/4长度
            
            stability_scores = []
            for i in range(0, background_band.shape[1] - window_size, window_size):
                window = background_band[:, i:i+window_size]
                window_mean = np.mean(window)
                window_std = np.std(window)
                stability = 1 - window_std / (window_mean + 1e-10)
                stability_scores.append(stability)
            
            features['background_stability'] = np.mean(stability_scores) if stability_scores else 0
        else:
            features['background_stability'] = 0
        
        # 5. 声音层次丰富度
        # 用于区分复杂场景（如商业区）和简单场景（如空地）
        # 计算不同频段的独立性
        freq_bands = [
            (0, 250),      # 低频
            (250, 1000),   # 中低频
            (1000, 4000),  # 中高频
            (4000, 8000)   # 高频
        ]
        
        band_activities = []
        for low, high in freq_bands:
            band_mask = (freqs >= low) & (freqs < high)
            if np.any(band_mask):
                band = stft[band_mask, :]
                if band.size > 0:
                    activity = np.std(np.mean(band, axis=0))
                    band_activities.append(activity)
        
        if band_activities:
            # 计算各频段活动的独立性
            features['sound_layer_richness'] = np.std(band_activities) / (np.mean(band_activities) + 1e-10)
        else:
            features['sound_layer_richness'] = 0
        
        # 6. 商业活动指标
        # 综合人声、音乐、周期性声音等
        commercial_score = 0
        
        # 检测可能的音乐（规律的节奏 + 和谐的频率关系）
        if 'tempo' in features and features.get('tempo', 0) > 60:
            commercial_score += 0.3
        
        # 多样的人声活动
        if features.get('human_activity_diversity', 0) > 0.5:
            commercial_score += 0.3
        
        # 中高频能量（商业区通常比较"明亮"）
        if features.get('high_freq_ratio', 0) > 0.3:
            commercial_score += 0.2
        
        # 声音的连续性（商业区很少安静）
        if features.get('sound_continuity', 0) > 0.7:
            commercial_score += 0.2
        
        features['commercial_activity_score'] = commercial_score
        
        return features
    
    def _calculate_entropy(self, signal_segment):
        """计算信号的熵"""
        # 将信号转换为概率分布
        hist, _ = np.histogram(signal_segment, bins=50)
        hist = hist / (np.sum(hist) + 1e-10)
        
        # 计算熵
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return entropy
    
    def visualize_audio_features(self, audio_path, save_path=None):
        """可视化音频特征（增强版）"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        fig.suptitle(f'音频特征可视化: {os.path.basename(audio_path)}', fontsize=16)
        
        # 1. 波形图
        axes[0, 0].plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
        axes[0, 0].set_title('波形图')
        axes[0, 0].set_xlabel('时间 (秒)')
        axes[0, 0].set_ylabel('振幅')
        
        # 2. 频谱图
        D = librosa.stft(audio)
        DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz', ax=axes[0, 1])
        axes[0, 1].set_title('频谱图')
        fig.colorbar(img, ax=axes[0, 1], format='%+2.0f dB')
        
        # 3. 环境声音似然度
        features = self.extract_comprehensive_features(audio_path)
        env_sounds = ['water', 'wind', 'bird', 'traffic', 'industrial']
        env_values = [
            features.get('water_sound_likelihood', 0),
            features.get('wind_sound_likelihood', 0),
            features.get('bird_sound_likelihood', 0),
            features.get('traffic_presence', 0),
            features.get('industrial_sound_presence', 0)
        ]
        
        axes[1, 0].bar(env_sounds, env_values)
        axes[1, 0].set_title('环境声音检测')
        axes[1, 0].set_ylabel('似然度')
        axes[1, 0].set_ylim(0, 1)
        
        # 4. 频段能量分布
        freq_bands = ['Sub-bass\n<60Hz', 'Bass\n60-250Hz', 'Low-mid\n250-500Hz', 
                     'Mid\n500-2K', 'High-mid\n2K-4K', 'High\n>4KHz']
        band_values = [
            features.get('sub_bass_ratio', 0),
            features.get('bass_ratio', 0),
            features.get('low_mid_ratio', 0),
            features.get('mid_ratio', 0),
            features.get('high_mid_ratio', 0),
            features.get('high_ratio', 0)
        ]
        
        axes[1, 1].bar(freq_bands, band_values)
        axes[1, 1].set_title('频段能量分布')
        axes[1, 1].set_ylabel('能量比例')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 5. 时间特征
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        time_frames = np.arange(len(onset_env)) * 512 / sr
        axes[2, 0].plot(time_frames, onset_env)
        axes[2, 0].set_title('起始强度包络')
        axes[2, 0].set_xlabel('时间 (秒)')
        axes[2, 0].set_ylabel('强度')
        
        # 6. 空间特征
        spatial_features = ['混响量', '空间开阔度', '距离感', '背景稳定性']
        spatial_values = [
            features.get('reverb_amount', 0),
            features.get('spatial_openness', 0) / 10,  # 缩放以便显示
            features.get('distance_cue', 0) / 10,
            features.get('background_stability', 0)
        ]
        
        axes[2, 1].bar(spatial_features, spatial_values)
        axes[2, 1].set_title('空间特征')
        axes[2, 1].set_ylabel('值')
        axes[2, 1].tick_params(axis='x', rotation=45)
        
        # 7. MFCC热图
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[3, 0])
        axes[3, 0].set_title('MFCC特征')
        axes[3, 0].set_ylabel('MFCC系数')
        fig.colorbar(img, ax=axes[3, 0])
        
        # 8. 场景匹配雷达图
        # 选择关键特征进行雷达图展示
        from matplotlib.patches import Circle, RegularPolygon
        from matplotlib.path import Path
        from matplotlib.projections.polar import PolarAxes
        from matplotlib.projections import register_projection
        from matplotlib.spines import Spine
        from matplotlib.transforms import Affine2D
        
        # 使用普通的极坐标图作为雷达图
        ax = plt.subplot(4, 2, 8, projection='polar')
        
        # 选择要显示的特征
        radar_features = [
            '人声活动',
            '交通噪音', 
            '自然声音',
            '工业声音',
            '空间混响',
            '声音连续性'
        ]
        
        radar_values = [
            features.get('voice_activity_ratio', 0),
            features.get('traffic_presence', 0),
            (features.get('bird_sound_likelihood', 0) + features.get('wind_sound_likelihood', 0)) / 2,
            features.get('industrial_sound_presence', 0),
            features.get('reverb_amount', 0),
            features.get('sound_continuity', 0)
        ]
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
        radar_values += radar_values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax.plot(angles, radar_values, 'o-', linewidth=2)
        ax.fill(angles, radar_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_features)
        ax.set_ylim(0, 1)
        ax.set_title('场景特征雷达图', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        return fig


# 使用示例和测试
if __name__ == "__main__":
    print("增强版音频分析器")
    print("-" * 50)
    
    analyzer = AudioFeatureAnalyzer()
    
    # 测试特征提取
    test_audio = "path/to/test/audio.wav"  # 替换为实际路径
    
    try:
        features = analyzer.extract_comprehensive_features(test_audio)
        
        print(f"提取的特征总数: {len(features)}")
        print("\n关键环境特征:")
        print(f"  水声似然度: {features.get('water_sound_likelihood', 0):.3f}")
        print(f"  风声似然度: {features.get('wind_sound_likelihood', 0):.3f}")
        print(f"  鸟鸣似然度: {features.get('bird_sound_likelihood', 0):.3f}")
        print(f"  交通存在度: {features.get('traffic_presence', 0):.3f}")
        print(f"  工业声存在: {features.get('industrial_sound_presence', 0):.3f}")
        
        print("\n空间特征:")
        print(f"  混响量: {features.get('reverb_amount', 0):.3f}")
        print(f"  空间开阔度: {features.get('spatial_openness', 0):.3f}")
        print(f"  距离感: {features.get('distance_cue', 0):.3f}")
        
        print("\n人类活动特征:")
        print(f"  人声活动比: {features.get('voice_activity_ratio', 0):.3f}")
        print(f"  活动多样性: {features.get('human_activity_diversity', 0):.3f}")
        
        # 可视化
        analyzer.visualize_audio_features(test_audio, "audio_analysis.png")
        
    except Exception as e:
        print(f"注意：需要提供实际的音频文件路径进行测试")
        print(f"错误信息: {e}")
    
    print("\n特征说明:")
    print("- water_sound_likelihood: 0-1，越高表示越可能包含水声")
    print("- wind_sound_likelihood: 0-1，越高表示越可能包含风声")
    print("- bird_sound_likelihood: 0-1，越高表示越可能包含鸟鸣")
    print("- traffic_presence: 0-1，综合交通噪音指标")
    print("- reverb_amount: 0-1，空间混响程度")
    print("- distance_cue: 声源距离估计，值越大表示声源越远")
