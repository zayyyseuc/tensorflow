# 03_audio_analyzer.py
"""
音频分析器：深入理解声音的特征
这个模块是我们的"声音显微镜"，帮助我们看到声音的内部结构
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureAnalyzer:
    """声音特征分析器
    
    想象这是一个多功能的声音分析仪器，
    能够从多个维度解析声音的特性
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def extract_comprehensive_features(self, audio_path):
        """提取全面的音频特征
        
        这个过程就像给声音做全身体检，
        我们要检查它的各个方面
        """
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        features = {}
        
        # 1. 时域特征（直接从波形中提取）
        # 这些特征告诉我们声音的整体"形状"
        features.update(self._extract_temporal_features(audio))
        
        # 2. 频域特征（从频谱中提取）
        # 这些特征告诉我们声音的"颜色"
        features.update(self._extract_spectral_features(audio, sr))
        
        # 3. 梅尔频谱特征（模拟人耳感知）
        # 这些特征告诉我们人耳听到的感受
        features.update(self._extract_mel_features(audio, sr))
        
        # 4. 节奏和动态特征
        # 这些特征告诉我们声音的"节奏感"
        features.update(self._extract_rhythm_features(audio, sr))
        
        # 5. 场景特定特征
        # 这些是我们专门为场景分类设计的特征
        features.update(self._extract_scene_specific_features(audio, sr))
        
        return features
    
    def _extract_temporal_features(self, audio):
        """提取时域特征
        
        时域特征就像看一个人的身高体重，
        是最直观的特征
        """
        features = {}
        
        # 能量相关特征
        features['rms_energy'] = np.sqrt(np.mean(audio**2))
        features['energy_variance'] = np.var(audio**2)
        
        # 过零率（声音穿过零点的频率）
        # 高过零率通常意味着高频成分多
        features['zero_crossing_rate'] = np.mean(
            librosa.feature.zero_crossing_rate(audio)[0]
        )
        
        # 动态范围（声音的强弱对比）
        features['dynamic_range'] = np.max(np.abs(audio)) - np.min(np.abs(audio))
        
        # 静音比例（这对场景识别很重要）
        silence_threshold = 0.01 * np.max(np.abs(audio))
        features['silence_ratio'] = np.sum(np.abs(audio) < silence_threshold) / len(audio)
        
        # 声音的"粗糙度"（变化的剧烈程度）
        features['roughness'] = np.mean(np.abs(np.diff(audio)))
        
        return features
    
    def _extract_spectral_features(self, audio, sr):
        """提取频谱特征
        
        频谱特征就像分析一束光的颜色成分，
        告诉我们声音包含哪些频率
        """
        features = {}
        
        # 计算短时傅里叶变换
        stft = np.abs(librosa.stft(audio))
        
        # 频谱质心（声音的"重心"在哪个频率）
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # 频谱带宽（声音的频率"宽度"）
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        # 频谱滚降点（大部分能量集中在哪个频率以下）
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        # 频谱对比度（不同频段的能量对比）
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_band_{i}'] = np.mean(spectral_contrast[i])
        
        # 频谱平坦度（声音是否接近白噪声）
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        
        return features
    
    def _extract_mel_features(self, audio, sr):
        """提取梅尔频谱特征
        
        梅尔频谱模拟人耳的感知特性，
        是音频分类的"黄金标准"特征
        """
        features = {}
        
        # 计算MFCC（梅尔频率倒谱系数）
        # 这是语音识别中最重要的特征
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # 计算梅尔频谱的统计特征
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        features['mel_energy_mean'] = np.mean(mel_spectrogram)
        features['mel_energy_std'] = np.std(mel_spectrogram)
        
        return features
    
    def _extract_rhythm_features(self, audio, sr):
        """提取节奏和动态特征
        
        这些特征帮助我们理解声音的时间模式，
        比如是规律的还是随机的
        """
        features = {}
        
        # 节拍跟踪
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = tempo
        features['beat_strength'] = np.mean(librosa.onset.onset_strength(y=audio, sr=sr))
        
        # 计算声音事件的密度
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features['onset_rate'] = np.sum(onset_env > np.mean(onset_env)) / (len(audio) / sr)
        
        return features
    
    def _extract_scene_specific_features(self, audio, sr):
        """提取场景特定的特征
        
        这些是我们专门为区分13个场景设计的特征
        """
        features = {}
        
        # 低频能量比例（对区分城市边缘很重要）
        stft = np.abs(librosa.stft(audio))
        freq_bins = stft.shape[0]
        low_freq_bins = freq_bins // 4  # 低频部分
        
        low_freq_energy = np.sum(stft[:low_freq_bins, :])
        total_energy = np.sum(stft)
        features['low_freq_ratio'] = low_freq_energy / (total_energy + 1e-10)
        
        # 声音的连续性（区分繁忙和安静的场景）
        envelope = librosa.onset.onset_strength(y=audio, sr=sr)
        features['sound_continuity'] = 1 - np.std(envelope) / (np.mean(envelope) + 1e-10)
        
        # 频谱的稳定性（工业区vs自然环境）
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_stability'] = 1 / (np.std(spectral_centroid) + 1)
        
        # 人声活动检测（粗略估计）
        # 人声通常在300-3000Hz范围内
        human_voice_range = (300, 3000)
        freqs = librosa.fft_frequencies(sr=sr)
        voice_bins = np.where((freqs >= human_voice_range[0]) & 
                             (freqs <= human_voice_range[1]))[0]
        
        voice_energy = np.sum(stft[voice_bins, :])
        features['voice_activity_ratio'] = voice_energy / (total_energy + 1e-10)
        
        return features
    
    def visualize_audio_features(self, audio_path, save_path=None):
        """可视化音频特征
        
        这个函数帮助我们"看见"声音，
        就像X光片帮助医生看见骨骼
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'音频特征可视化: {os.path.basename(audio_path)}', fontsize=16)
        
        # 1. 波形图
        axes[0, 0].plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
        axes[0, 0].set_title('波形图（时域表示）')
        axes[0, 0].set_xlabel('时间 (秒)')
        axes[0, 0].set_ylabel('振幅')
        
        # 2. 频谱图
        D = librosa.stft(audio)
        DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz', ax=axes[0, 1])
        axes[0, 1].set_title('频谱图（时频表示）')
        fig.colorbar(img, ax=axes[0, 1], format='%+2.0f dB')
        
        # 3. 梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1, 0])
        axes[1, 0].set_title('梅尔频谱图（人耳感知）')
        fig.colorbar(img, ax=axes[1, 0], format='%+2.0f dB')
        
        # 4. MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[1, 1])
        axes[1, 1].set_title('MFCC特征')
        fig.colorbar(img, ax=axes[1, 1])
        
        # 5. 能量包络
        hop_length = 512
        frame_length = 2048
        energy = np.array([
            np.sum(np.abs(audio[i:i+frame_length]**2))
            for i in range(0, len(audio)-frame_length, hop_length)
        ])
        time_frames = np.arange(len(energy)) * hop_length / sr
        axes[2, 0].plot(time_frames, energy)
        axes[2, 0].set_title('能量包络')
        axes[2, 0].set_xlabel('时间 (秒)')
        axes[2, 0].set_ylabel('能量')
        
        # 6. 频谱质心变化
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        frames = range(len(spectral_centroid))
        t = librosa.frames_to_time(frames, sr=sr)
        axes[2, 1].plot(t, spectral_centroid)
        axes[2, 1].set_title('频谱质心变化')
        axes[2, 1].set_xlabel('时间 (秒)')
        axes[2, 1].set_ylabel('Hz')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        return fig

# 使用示例
if __name__ == "__main__":
    analyzer = AudioFeatureAnalyzer()
    
    # 假设我们有一个测试音频
    test_audio = "path/to/test/audio.wav"
    
    # 提取特征
    features = analyzer.extract_comprehensive_features(test_audio)
    
    print("提取的特征数量：", len(features))
    print("\n部分特征示例：")
    for key, value in list(features.items())[:10]:
        print(f"  {key}: {value:.4f}")
