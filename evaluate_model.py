#!/usr/bin/env python3
"""
Test deepfake detection system with real audio files
"""

import os
import requests
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

def get_test_files(test_dir, max_files_per_class=20):
    """Get limited number of audio files from test directory for quick testing"""
    audio_files = []
    real_count = 0
    fake_count = 0
    
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                file_path = os.path.join(root, file)
                # Determine label based on directory structure
                label = 'fake' if 'fake' in root.lower() or 'deepfake' in root.lower() else 'real'
                
                # Limit files per class
                if label == 'real' and real_count < max_files_per_class:
                    audio_files.append((file_path, label))
                    real_count += 1
                elif label == 'fake' and fake_count < max_files_per_class:
                    audio_files.append((file_path, label))
                    fake_count += 1
                    
                # Stop if we have enough files
                if real_count >= max_files_per_class and fake_count >= max_files_per_class:
                    break
        if real_count >= max_files_per_class and fake_count >= max_files_per_class:
            break
    
    return audio_files

def analyze_audio(file_path):
    """Analyze a single audio file"""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
            response = requests.post("http://127.0.0.1:8010/analyze", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'prediction': 'fake' if result['is_suspected_fraud'] else 'real',
                    'probability': result['p_real'],
                    'confidence': result['confidence'],
                    'authenticity_score': result['authenticity_score']
                }
            else:
                print(f"❌ {file_path}: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Error analyzing {file_path}: {e}")
        return None

def evaluate_model(test_files):
    """Evaluate model performance"""
    print(f"🧪 Testing with {len(test_files)} audio files...")
    
    predictions = []
    actual_labels = []
    results = []
    
    # Test each file
    for i, (file_path, actual_label) in enumerate(test_files):
        print(f"📊 [{i+1}/{len(test_files)}] Testing: {os.path.basename(file_path)}")
        
        result = analyze_audio(file_path)
        if result:
            predictions.append(result['prediction'])
            actual_labels.append(actual_label)
            results.append({
                'file': file_path,
                'actual': actual_label,
                'predicted': result['prediction'],
                'probability': result['probability'],
                'confidence': result['confidence'],
                'correct': result['prediction'] == actual_label
            })
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    # Calculate metrics
    if len(predictions) > 0:
        accuracy = accuracy_score(actual_labels, predictions)
        f1 = f1_score(actual_labels, predictions, average='weighted')
        
        # Generate detailed report
        report = classification_report(actual_labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(actual_labels, predictions)
        
        return {
            'total_files': len(test_files),
            'successful_analyses': len(results),
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'detailed_results': results
        }
    else:
        return None

def print_results(results):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("📊 DEEPFAKE DETECTION MODEL EVALUATION")
    print("="*60)
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total Files Tested: {results['total_files']}")
    print(f"   Successful Analyses: {results['successful_analyses']}")
    print(f"   Success Rate: {results['successful_analyses']/results['total_files']*100:.1f}%")
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   F1-Score: {results['f1_score']:.4f}")
    
    print(f"\n📋 Classification Report:")
    for class_name in ['real', 'fake']:
        if class_name in results['classification_report']:
            metrics = results['classification_report'][class_name]
            print(f"   {class_name.upper()}:")
            print(f"      Precision: {metrics['precision']:.4f}")
            print(f"      Recall: {metrics['recall']:.4f}")
            print(f"      F1-Score: {metrics['f1-score']:.4f}")
            print(f"      Support: {metrics['support']}")
    
    print(f"\n🔄 Confusion Matrix:")
    cm = results['confusion_matrix']
    print(f"           Predicted")
    print(f"           Real   Fake")
    print(f"   Real   {cm[0,0]:<6} {cm[0,1]:<6}")
    print(f"   Fake   {cm[1,0]:<6} {cm[1,1]:<6}")
    
    print(f"\n✅ Correct Predictions: {np.sum(np.diag(cm))}")
    print(f"❌ Incorrect Predictions: {results['total_files'] - np.sum(np.diag(cm))}")
    
    # Show some detailed results
    print(f"\n📝 Sample Results:")
    correct_samples = [r for r in results['detailed_results'] if r['correct']]
    incorrect_samples = [r for r in results['detailed_results'] if not r['correct']]
    
    print(f"   ✅ Correct Predictions (showing 3):")
    for i, result in enumerate(correct_samples[:3]):
        print(f"      {i+1}. {os.path.basename(result['file'])}: {result['actual']} → {result['predicted']} (confidence: {result['confidence']:.4f})")
    
    print(f"   ❌ Incorrect Predictions (showing 3):")
    for i, result in enumerate(incorrect_samples[:3]):
        print(f"      {i+1}. {os.path.basename(result['file'])}: {result['actual']} → {result['predicted']} (confidence: {result['confidence']:.4f})")

def main():
    test_dir = r"D:\dataset\for-rerecorded\testing"
    
    print("🚀 Starting Deepfake Detection Model Evaluation")
    print("="*60)
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        print("💡 Please update the test_dir variable to point to your dataset")
        return
    
    # Get test files
    test_files = get_test_files(test_dir)
    
    if len(test_files) == 0:
        print(f"❌ No audio files found in: {test_dir}")
        return
    
    print(f"📁 Found {len(test_files)} audio files in test directory")
    
    # Check if backend is running
    try:
        response = requests.get("http://127.0.0.1:8010/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend health check failed")
            return
    except:
        print("❌ Backend not running. Please start it first:")
        print("   cd D:\\Workspace\\deepfake-voice-fraud-dedection")
        print("   .venv\\Scripts\\python.exe -m uvicorn backend.main_tf:app --host 127.0.0.1 --port 8010")
        return
    
    print("✅ Backend is running and healthy")
    
    # Run evaluation
    results = evaluate_model(test_files)
    
    if results:
        print_results(results)
        
        # Save results to file
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: evaluation_results.json")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()