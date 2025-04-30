import os
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa
from tqdm import tqdm

def extract_mfcc(audio, sr=16000, n_mfcc=20):
    """Extract MFCC features from audio"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return features

def extract_spec(audio, sr=16000, n_fft=512, hop_length=256):
    """Extract log mel-spectrogram features from audio"""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=80)
    log_mel_spec = librosa.power_to_db(mel_spec)
    return log_mel_spec

def extract_cqt(audio, sr=16000, hop_length=256):
    """Extract Constant-Q Transform features from audio"""
    cqt = librosa.cqt(y=audio, sr=sr, hop_length=hop_length)
    return np.abs(cqt)

class ASVSpoofDataset(Dataset):
    def __init__(self, root_dir, protocol_file, feature_type='mfcc', max_len=None, is_train=True, use_subsampling=True):
        """
        Args:
            root_dir (string): Directory with all the audio files.
            protocol_file (string): Path to the protocol file.
            feature_type (string): Type of features to extract ('mfcc', 'spec', 'cqt').
            max_len (int): Maximum length of features sequence.
            is_train (bool): Whether this is for training or testing.
            use_subsampling (bool): Whether to subsample data for faster training.
        """
        self.root_dir = root_dir
        self.feature_type = feature_type
        self.max_len = max_len
        self.is_train = is_train
        
        # Read protocol file
        self.data = []
        
        print(f"Reading protocol file: {protocol_file}")
        try:
            with open(protocol_file, 'r') as f:
                lines = f.readlines()
                
                # Use tqdm for loading progress
                for line in tqdm(lines, desc=f"Loading {'training' if is_train else 'evaluation'} protocol"):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        speaker_id = parts[0]
                        file_id = parts[1]
                        label_text = parts[4]
                        label = 0 if label_text == 'bonafide' else 1  # 0 for bonafide, 1 for spoof
                        self.data.append((file_id, label))
            
            # Count number of bonafide and spoof samples
            bonafide_count = sum(1 for _, label in self.data if label == 0)
            spoof_count = sum(1 for _, label in self.data if label == 1)
            
            print(f"Dataset loaded: {len(self.data)} samples ({bonafide_count} bonafide, {spoof_count} spoof)")
            
            if is_train and use_subsampling:
                # Subsample for faster NAS
                if len(self.data) > 5000:
                    print(f"Subsampling training data for faster NAS...")
                    np.random.shuffle(self.data)
                    # Keep balanced class distribution
                    bonafide_samples = [item for item in self.data if item[1] == 0][:2500]
                    spoof_samples = [item for item in self.data if item[1] == 1][:2500]
                    self.data = bonafide_samples + spoof_samples
                    np.random.shuffle(self.data)
                    print(f"Subsampled to {len(self.data)} samples")
        
        except Exception as e:
            print(f"Error loading protocol file: {e}")
            self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_id, label = self.data[idx]
        audio_path = os.path.join(self.root_dir, f"{file_id}.flac")
        
        try:
            audio, sr = sf.read(audio_path)
            
            # Feature extraction
            if self.feature_type == 'mfcc':
                features = extract_mfcc(audio, sr)
            elif self.feature_type == 'spec':
                features = extract_spec(audio, sr)
            elif self.feature_type == 'cqt':
                features = extract_cqt(audio, sr)
            else:
                raise ValueError(f"Unknown feature type: {self.feature_type}")
            
            # Normalize features
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            # Handle sequence length
            seq_len = features.shape[1]
            if self.max_len is not None:
                if seq_len > self.max_len:
                    start = np.random.randint(0, seq_len - self.max_len) if self.is_train else 0
                    features = features[:, start:start+self.max_len]
                elif seq_len < self.max_len:
                    # Pad with zeros
                    pad_width = ((0, 0), (0, self.max_len - seq_len))
                    features = np.pad(features, pad_width, mode='constant')
            
            return torch.FloatTensor(features), torch.LongTensor([label])[0]
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a dummy sample in case of error
            dummy_features = np.zeros((60, 100 if self.max_len is None else self.max_len))
            return torch.FloatTensor(dummy_features), torch.LongTensor([label])[0]