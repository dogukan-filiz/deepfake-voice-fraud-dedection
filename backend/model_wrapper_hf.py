"""Hugging Face model wrapper for Speech-Arena-2025/DF_Arena_1B_V_1."""

from typing import Dict, Any
import numpy as np
import torch
import librosa

class HuggingFaceDeepfakeModel:
    """Wrapper for Speech-Arena-2025/DF_Arena_1B_V_1 Hugging Face model."""
    
    def __init__(self, 
                 model_id: str = "Speech-Arena-2025/DF_Arena_1B_V_1",
                 device: str = None):
        """Initialize the HF model.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_id = model_id
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        print(f"Initializing HF model {model_id} on {self.device}...")
        
        # Try to load the model
        try:
            from transformers import pipeline
            self.pipe = pipeline(
                "antispoofing",  # Use the correct task name
                model=model_id,
                device=0 if self.device.type == "cuda" else -1,
                trust_remote_code=True
            )
            print("HF model loaded successfully!")
        except Exception as e:
            print(f"Could not load HF model: {e}")
            raise e
            
        # Model expects 16kHz audio
        self.sampling_rate = 16000
        
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Run inference on audio features.
        
        Args:
            features: Dictionary containing 'waveform' and 'sr'
            
        Returns:
            Dictionary with prediction results
        """
        waveform = np.asarray(features["waveform"], dtype=np.float32)
        sr = int(features["sr"])
        
        # Ensure correct shape
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)
            
        # Validate audio
        if waveform.size == 0:
            raise ValueError("Empty waveform - no audio data to process.")
        if not np.isfinite(waveform).all():
            raise ValueError("Audio data contains NaN or Inf values.")
            
        # Resample if needed
        if sr != self.sampling_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sampling_rate)
            
        # Normalize audio
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1.0:
            waveform = waveform / peak
            
        try:
            # Run HF model inference
            result = self.pipe(waveform, sampling_rate=self.sampling_rate)
            
            # Parse result - expect list of dicts with label and score
            if isinstance(result, list) and len(result) > 0:
                # Get the top prediction
                top_result = result[0]
                label = top_result.get('label', '').lower()
                score = top_result.get('score', 0.5)
                
                # Map to p_real and p_fake
                if 'bonafide' in label or 'real' in label:
                    p_real = score
                    p_fake = 1.0 - score
                else:  # spoof or fake
                    p_fake = score
                    p_real = 1.0 - score
            else:
                # Fallback
                p_real = 0.5
                p_fake = 0.5
                
        except Exception as e:
            print(f"HF model inference failed: {e}, using fallback")
            # Use heuristic fallback
            p_real = 0.5
            p_fake = 0.5
            
        # Handle spectral residual from features if available
        spectral_resid = float(features.get("spectral_residual_score", 0.0))
        
        return {
            "p_real": round(p_real, 6),
            "p_fake": round(p_fake, 6),
            "spectral_residual": spectral_resid,
            "num_chunks": 1,  # HF model processes whole audio at once
            "max_chunk_p_fake": round(p_fake, 6),
        }