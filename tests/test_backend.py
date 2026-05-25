import subprocess
import sys
import os
import time
import json
import requests
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def start_backend():
    """Start backend server in background"""
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "backend.main_tf:app", 
        "--host", "127.0.0.1", 
        "--port", "8010"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))
    
    print("🚀 Backend server starting...")
    time.sleep(8)  # Wait for server to start
    
    # Test if server is running
    try:
        response = requests.get("http://127.0.0.1:8010/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running!")
            return backend_process
        else:
            print(f"❌ Backend server returned status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Backend server failed to start: {e}")
        return None

def test_backend_health():
    """Test if backend is running"""
    try:
        response = requests.get("http://127.0.0.1:8010/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_test_files(test_dir, max_files_per_class=100):
    """Get test audio files with balanced classes"""
    real_files = []
    fake_files = []
    
    # Get real files
    real_dir = os.path.join(test_dir, "real")
    if os.path.exists(real_dir):
        for file in os.listdir(real_dir):
            if file.lower().endswith('.wav'):
                file_path = os.path.join(real_dir, file)
                real_files.append((file_path, 'real'))
                if len(real_files) >= max_files_per_class:
                    break
    
    # Get fake files
    fake_dir = os.path.join(test_dir, "fake")
    if os.path.exists(fake_dir):
        for file in os.listdir(fake_dir):
            if file.lower().endswith('.wav'):
                file_path = os.path.join(fake_dir, file)
                fake_files.append((file_path, 'fake'))
                if len(fake_files) >= max_files_per_class:
                    break
    
    # Combine and shuffle
    all_files = real_files + fake_files
    np.random.shuffle(all_files)
    
    return all_files

def analyze_file(file_path):
    """Analyze a single audio file"""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
            response = requests.post("http://127.0.0.1:8010/analyze", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                # Check the actual response structure
                print(f"Debug: Response keys: {list(result.keys())}")
                
                return {
                    'success': True,
                    'prediction': 'fake' if result.get('is_suspected_fraud', False) else 'real',
                    'probability': result.get('p_real', 0.5),
                    'confidence': abs(result.get('p_real', 0.5) - 0.5) * 2,  # Calculate confidence
                    'authenticity_score': result.get('authenticity_score', 0.5)
                }
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def calculate_metrics(predictions, actual_labels):
    """Calculate comprehensive metrics"""
    # Basic metrics
    accuracy = accuracy_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions, average='weighted')
    
    # Classification report
    report = classification_report(actual_labels, predictions, output_dict=True, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(actual_labels, predictions)
    
    # Per-class metrics
    real_metrics = {
        'precision': report.get('real', {}).get('precision', 0),
        'recall': report.get('real', {}).get('recall', 0),
        'f1_score': report.get('real', {}).get('f1-score', 0),
        'support': report.get('real', {}).get('support', 0)
    }
    
    fake_metrics = {
        'precision': report.get('fake', {}).get('precision', 0),
        'recall': report.get('fake', {}).get('recall', 0),
        'f1_score': report.get('fake', {}).get('f1-score', 0),
        'support': report.get('fake', {}).get('support', 0)
    }
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'real_metrics': real_metrics,
        'fake_metrics': fake_metrics,
        'confusion_matrix': conf_matrix.tolist(),
        'total_samples': len(predictions)
    }

def main():
    test_dir = r"D:\dataset\for-rerecorded\testing"
    
    print("🧪 Deepfake Detection Model Comprehensive Test")
    print("=" * 60)
    
    # Start backend
    backend_process = start_backend()
    
    if not backend_process:
        print("❌ Failed to start backend server")
        return
    
    # Get test files
    test_files = get_test_files(test_dir, 100)  # Test with up to 100 files per class
    print(f"📁 Found {len(test_files)} test files ({len([f for f in test_files if f[1] == 'real'])} real, {len([f for f in test_files if f[1] == 'fake'])} fake)")
    
    # Test files
    results = []
    successful_analyses = 0
    
    print("\n🔬 Testing audio files...")
    for i, (file_path, actual_label) in enumerate(test_files):
        print(f"[{i+1}/{len(test_files)}] Testing: {os.path.basename(file_path)} ({actual_label})")
        
        result = analyze_file(file_path)
        if result['success']:
            is_correct = result['prediction'] == actual_label
            results.append({
                'file': file_path,
                'actual': actual_label,
                'predicted': result['prediction'],
                'probability': result['probability'],
                'confidence': result['confidence'],
                'correct': is_correct
            })
            successful_analyses += 1
            
            if is_correct:
                print(f"   ✅ Correct: {actual_label} → {result['prediction']} (confidence: {result['confidence']:.4f})")
            else:
                print(f"   ❌ Wrong: {actual_label} → {result['prediction']} (confidence: {result['confidence']:.4f})")
        else:
            print(f"   ❌ Failed: {result['error']}")
        
        time.sleep(0.3)  # Small delay to avoid overwhelming the server
    
    print(f"\n📊 Test Results Summary:")
    print(f"   Total Files Tested: {len(test_files)}")
    print(f"   Successful Analyses: {successful_analyses}")
    print(f"   Success Rate: {successful_analyses/len(test_files)*100:.1f}%")
    
    if results:
        # Calculate metrics
        predictions = [r['predicted'] for r in results]
        actual_labels = [r['actual'] for r in results]
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / len(results)
        
        metrics = calculate_metrics(predictions, actual_labels)
        
        print(f"\n🎯 Performance Metrics:")
        print(f"   Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Overall F1-Score: {metrics['f1_score']:.4f}")
        
        print(f"\n📋 Class-wise Performance:")
        print(f"   Real Class:")
        print(f"      Precision: {metrics['real_metrics']['precision']:.4f}")
        print(f"      Recall: {metrics['real_metrics']['recall']:.4f}")
        print(f"      F1-Score: {metrics['real_metrics']['f1_score']:.4f}")
        print(f"      Support: {metrics['real_metrics']['support']}")
        
        print(f"   Fake Class:")
        print(f"      Precision: {metrics['fake_metrics']['precision']:.4f}")
        print(f"      Recall: {metrics['fake_metrics']['recall']:.4f}")
        print(f"      F1-Score: {metrics['fake_metrics']['f1_score']:.4f}")
        print(f"      Support: {metrics['fake_metrics']['support']}")
        
        print(f"\n🔄 Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"           Predicted")
        print(f"           Real   Fake")
        print(f"   Real   {cm[0,0]:<6} {cm[0,1]:<6}")
        print(f"   Fake   {cm[1,0]:<6} {cm[1,1]:<6}")
        
        print(f"\n✅ Correct Predictions: {correct}")
        print(f"❌ Incorrect Predictions: {len(results) - correct}")
        
        # Show confidence distribution
        real_confidences = [r['confidence'] for r in results if r['actual'] == 'real']
        fake_confidences = [r['confidence'] for r in results if r['actual'] == 'fake']
        
        print(f"\n📊 Confidence Statistics:")
        print(f"   Real - Mean Confidence: {np.mean(real_confidences):.4f}")
        print(f"   Fake - Mean Confidence: {np.mean(fake_confidences):.4f}")
        
        # Save results
        test_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_files': len(test_files),
            'successful_analyses': successful_analyses,
            'success_rate': successful_analyses/len(test_files),
            'metrics': metrics,
            'detailed_results': results
        }
        
        with open('deepfake_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: deepfake_test_results.json")
        
        # Generate summary report
        print(f"\n📝 Summary Report:")
        print(f"   Model: TensorFlow Conformer")
        print(f"   Test Dataset: {test_dir}")
        print(f"   Total Samples: {len(test_files)}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Real Detection Rate: {metrics['real_metrics']['recall']:.4f}")
        print(f"   Fake Detection Rate: {metrics['fake_metrics']['recall']:.4f}")
        
    else:
        print("❌ No successful analyses completed")
    
    # Clean up
    print("\n🛑 Shutting down backend...")
    backend_process.terminate()
    backend_process.wait()
    print("✅ Test completed!")

if __name__ == "__main__":
    import time
    main()