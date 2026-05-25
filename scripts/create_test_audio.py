import numpy as np
import soundfile as sf
import os

def create_test_audio():
    """Test ses dosyaları oluştur"""
    
    # Sample rate
    sr = 16000
    
    # Test 1: Short clean audio (real-like)
    duration = 3.0  # 3 seconds
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # Create a more realistic voice-like signal
    # Combine multiple frequencies to simulate voice
    signal = (
        0.3 * np.sin(2 * np.pi * 150 * t) +  # Fundamental frequency
        0.2 * np.sin(2 * np.pi * 300 * t) +  # First harmonic
        0.1 * np.sin(2 * np.pi * 450 * t) +  # Second harmonic
        0.05 * np.sin(2 * np.pi * 600 * t) +  # Formant
        0.02 * np.random.normal(0, 1, len(t))  # Small noise
    )
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.5
    
    # Save as WAV
    test_real_path = "test_real.wav"
    sf.write(test_real_path, signal, sr)
    print(f"Created test real audio: {test_real_path}")
    
    # Test 2: Audio with artificial patterns (fake-like)
    # Add some periodic patterns that might be typical in synthetic audio
    fake_signal = signal.copy()
    
    # Add some regular patterns
    for i in range(0, len(fake_signal), sr//10):  # Every 100ms
        if i + sr//50 < len(fake_signal):
            fake_signal[i:i+sr//50] += 0.1 * np.sin(2 * np.pi * 50 * np.arange(sr//50) / sr)
    
    # Add some regular envelope
    envelope = np.exp(-t / duration)  # Decaying envelope
    fake_signal = fake_signal * envelope
    
    # Normalize
    fake_signal = fake_signal / np.max(np.abs(fake_signal)) * 0.5
    
    # Save as WAV
    test_fake_path = "test_fake.wav"
    sf.write(test_fake_path, fake_signal, sr)
    print(f"Created test fake audio: {test_fake_path}")
    
    return test_real_path, test_fake_path

if __name__ == "__main__":
    test_real, test_fake = create_test_audio()
    print(f"Test files created:")
    print(f"  - Real-like: {test_real}")
    print(f"  - Fake-like: {test_fake}")