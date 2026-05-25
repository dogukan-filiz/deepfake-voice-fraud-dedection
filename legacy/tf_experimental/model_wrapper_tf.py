import os
import numpy as np
import librosa
from typing import Any, Dict, Optional
import tensorflow as tf
import numpy as np
import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class TensorFlowModelWrapper:
    """TensorFlow Conformer model wrapper for deepfake voice detection"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "models", 
            "tf_conformer", 
            "working_model.h5"
        )
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "models", 
            "tf_conformer", 
            "config.json"
        )
        self.model = None
        self.model_name = "TensorFlow Conformer"
        self.input_sample_rate = 16000
        self.input_duration = 2.88  # 46080 samples / 16000 Hz = 2.88 seconds
        self.is_loaded = False
        self.load_error = None
        
        # Load model during initialization
        self._load_model()
    
    def _load_model(self):
        """Load the TensorFlow model"""
        try:
            logger.info(f"🔄 Loading TensorFlow model from: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load the saved model
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
            self.load_error = None
            
            logger.info(f"✅ Successfully loaded model: {self.model_name}")
            logger.info(f"📊 Model input shape: {self.model.input_shape}")
            logger.info(f"📊 Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            error_msg = f"❌ Failed to load TensorFlow model: {str(e)}"
            logger.error(error_msg)
            self.is_loaded = False
            self.load_error = error_msg
            self.model = None
    
    def _preprocess_audio(self, audio_data) -> np.ndarray:
        """Preprocess audio data for model input"""
        try:
            # Handle different input types
            if isinstance(audio_data, dict):
                # PyTorch format: {"waveform": np.array, "sr": int}
                waveform = audio_data["waveform"]
            else:
                waveform = audio_data
            
            # Normalize audio to [-1, 1] range if not already
            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)
            
            # Normalize to [-1, 1] range
            if np.max(np.abs(waveform)) > 0:
                waveform = waveform / np.max(np.abs(waveform))
            
            # Ensure minimum length (8000 samples = 0.5 seconds @ 16kHz)
            min_samples = 8000
            if len(waveform) < min_samples:
                # Pad with zeros to minimum length
                padding = min_samples - len(waveform)
                waveform = np.pad(waveform, (0, padding), mode='constant')
            
            # Ensure correct length (46080 samples = 2.88 seconds @ 16kHz)
            if len(waveform) > 46080:
                # Trim to correct length
                waveform = waveform[:46080]
            elif len(waveform) < 46080:
                # Pad with zeros to model input size
                padding = 46080 - len(waveform)
                waveform = np.pad(waveform, (0, padding), mode='constant')
            
            # Add batch dimension
            waveform = np.expand_dims(waveform, axis=0)
            
            logger.debug(f"🎵 Preprocessed audio shape: {waveform.shape}")
            return waveform
            
        except Exception as e:
            logger.error(f"❌ Audio preprocessing error: {str(e)}")
            raise
    
    def predict(self, audio_data) -> Dict[str, Any]:
        """Make prediction on audio data"""
        if not self.is_loaded:
            raise RuntimeError(f"Model not loaded. Error: {self.load_error}")
        
        try:
            logger.info("🔍 Starting TensorFlow model prediction...")
            
            # Handle different input types
            if isinstance(audio_data, dict):
                # PyTorch format: {"waveform": np.array, "sr": int}
                waveform = audio_data["waveform"]
                sample_rate = audio_data.get("sr", 16000)
                
                # Resample to 16kHz if needed
                if sample_rate != self.input_sample_rate:
                    waveform = self._resample_audio(waveform, sample_rate, self.input_sample_rate)
                
                audio_data = waveform
            
            # Preprocess audio
            processed_audio = self._preprocess_audio(audio_data)
            
            # Make prediction
            prediction = self.model.predict(processed_audio, verbose=0)
            
            # Extract probability (sigmoid output)
            probability = float(prediction[0][0])
            
            # Determine class based on threshold
            threshold = 0.5
            is_deepfake = probability < threshold
            confidence = probability if not is_deepfake else 1.0 - probability
            
            result = {
                "model_name": self.model_name,
                "p_real": probability,
                "p_fake": 1.0 - probability,
                "spectral_residual": 0.0,  # TensorFlow model doesn't provide this
                "probability": probability,
                "is_deepfake": is_deepfake,
                "confidence": confidence,
                "threshold": threshold,
                "input_sample_rate": self.input_sample_rate,
                "input_duration": self.input_duration,
                "model_type": "tensorflow",
                "timestamp": None  # Will be set by the API
            }
            
            logger.info(f"✅ Prediction completed - Probability: {probability:.4f}, Deepfake: {is_deepfake}")
            return result
            
        except Exception as e:
            error_msg = f"❌ Prediction error: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _resample_audio(self, audio_data, orig_sr, target_sr):
        """Resample audio data using librosa"""
        try:
            if orig_sr == target_sr:
                return audio_data
            
            duration = len(audio_data) / orig_sr
            target_length = int(duration * target_sr)
            
            # Simple resampling using interpolation
            from scipy import signal
            resampled = signal.resample(audio_data, target_length)
            return resampled.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Resampling failed, using original: {e}")
            return audio_data
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "load_error": self.load_error,
            "input_sample_rate": self.input_sample_rate,
            "input_duration": self.input_duration,
            "model_type": "tensorflow",
            "total_params": self.model.count_params() if self.model else 0,
            "input_shape": self.model.input_shape if self.model else None,
            "output_shape": self.model.output_shape if self.model else None
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the model"""
        try:
            if not self.is_loaded:
                return {
                    "status": "error",
                    "message": self.load_error or "Model not loaded",
                    "model_name": self.model_name
                }
            
            # Test inference with dummy data
            dummy_audio = np.random.randn(46080).astype(np.float32) * 0.1
            
            try:
                result = self.predict(dummy_audio)
                return {
                    "status": "healthy",
                    "message": "Model loaded and inference successful",
                    "model_name": self.model_name,
                    "test_prediction": result["probability"]
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Inference failed: {str(e)}",
                    "model_name": self.model_name
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "model_name": self.model_name
            }

# Global model instance
_model_instance = None

def get_model() -> TensorFlowModelWrapper:
    """Get global model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = TensorFlowModelWrapper()
    return _model_instance

def reload_model() -> TensorFlowModelWrapper:
    """Reload the model"""
    global _model_instance
    _model_instance = TensorFlowModelWrapper()
    return _model_instance


def get_model_status() -> Dict[str, Any]:
    """Get model status information."""
    try:
        model = get_model()
        return {
            "loaded": True,
            "type": "TensorFlowConformer",
            "model_path": model.model_path,
            "input_shape": model.model.input_shape if model.model else None,
            "output_shape": model.model.output_shape if model.model else None,
        }
    except Exception as e:
        return {
            "loaded": False,
            "type": "TensorFlowConformer",
            "error": str(e),
            "model_path": None,
        }