#!/usr/bin/env python3
"""
Simple test script for deepfake detection
"""

import os
import requests
import numpy as np
import time

def test_backend_health():
    """Test if backend is running"""
    try:
        response = requests.get("http://127.0.0.1:8010/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_test_files(test_dir, max_files=10):
    """Get test audio files"""
    files = []
    real_count = 0
    fake_count = 0
    
    for root, dirs, files_in_dir in os.walk(test_dir):
        for file in files_in_dir:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                label = 'fake' if 'fake' in root.lower() else 'real'
                
                if label == 'real' and real_count < max_files//2:
                    files.append((file_path, label))
                    real_count += 1
                elif label == 'fake' and fake_count < max_files//2:
                    files.append((file_path, label))
                    fake_count += 1
                    
                if real_count + fake_count >= max_files:
                    break
        if real_count + fake_count >= max_files:
            break
    
    return files

def analyze_file(file_path):
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
                return None
    except:
        return None

def main():
    test_dir = r"D:\dataset\for-rerecorded\testing"
    
    print("🧪 Testing Deepfake Detection System")
    print("=" * 40)
    
    # Check backend
    if not test_backend_health():
        print("❌ Backend not running")
        return
    
    print("✅ Backend is running")
    
    # Get test files
    test_files = get_test_files(test_dir, 20)
    print(f"📁 Found {len(test_files)} test files")
    
    # Test files
    results = []
    for i, (file_path, actual_label) in enumerate(test_files):
        print(f"[{i+1}/{len(test_files)}] Testing: {os.path.basename(file_path)} ({actual_label})")
        
        result = analyze_file(file_path)
        if result:
            results.append({
                'file': file_path,
                'actual': actual_label,
                'predicted': result['prediction'],
                'probability': result['probability'],
                'confidence': result['confidence'],
                'correct': result['prediction'] == actual_label
            })
        else:
            print(f"❌ Failed to analyze: {file_path}")
        
        time.sleep(0.5)
    
    # Calculate results
    if results:
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / len(results)
        
        print(f"\n📊 Results:")
        print(f"   Total Files: {len(results)}")
        print(f"   Correct: {correct}")
        print(f"   Accuracy: {accuracy:.4f}")
        
        # Show detailed results
        print(f"\n📝 Detailed Results:")
        for r in results:
            status = "✅" if r['correct'] else "❌"
            print(f"   {status} {os.path.basename(r['file'])}: {r['actual']} → {r['predicted']} (confidence: {r['confidence']:.4f})")
        
        # Save results
        import json
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: test_results.json")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()