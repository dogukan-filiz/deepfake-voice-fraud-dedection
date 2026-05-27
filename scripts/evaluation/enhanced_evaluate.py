"""Enhanced evaluation script with configurable threshold."""

import os
import json
from pathlib import Path
import requests
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def find_audio_files(root_dirs):
    """Find audio files in multiple directories."""
    audio_files = []
    extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.webm'}
    
    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"Warning: Directory not found: {root_dir}")
            continue
            
        for category in ['real', 'fake']:
            category_dir = os.path.join(root_dir, category)
            if not os.path.exists(category_dir):
                continue
                
            for file in os.listdir(category_dir):
                if any(file.lower().endswith(ext) for ext in extensions):
                    audio_files.append({
                        'path': os.path.join(category_dir, file),
                        'category': category,
                        'filename': file
                    })
    
    return audio_files

def evaluate_threshold(results, threshold):
    """Evaluate results with given threshold."""
    y_true = [1 if r['true_label'] == 'real' else 0 for r in results]
    y_pred = [1 if r['authenticity_score'] > threshold else 0 for r in results]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'true_positives': cm[1, 1],
        'false_positives': cm[0, 1],
        'true_negatives': cm[0, 0],
        'false_negatives': cm[1, 0]
    }

def evaluate_model():
    """Evaluate model performance on larger dataset."""
    
    # Test URLs
    base_url = "http://127.0.0.1:8010"
    
    # Dataset directories
    dataset_dirs = [
        "D:\\dataset\\for-original",
        "D:\\dataset\\for-rerecorded",
        "D:\\Workspace\\deepfake-voice-fraud-dedection\\test_audio"
    ]
    
    # Find audio files
    audio_files = find_audio_files(dataset_dirs)
    
    print(f"Found {len(audio_files)} audio files")
    
    # Count by category
    real_files = [f for f in audio_files if f['category'] == 'real']
    fake_files = [f for f in audio_files if f['category'] == 'fake']
    
    print(f"Real files: {len(real_files)}, Fake files: {len(fake_files)}")
    
    # Test files (limit to 50 each)
    results = []
    
    print("Testing real samples...")
    tested_real = 0
    for file_info in real_files:
        if tested_real >= 50:
            break
        try:
            # Extract relative path for API
            rel_path = os.path.relpath(file_info['path'])
            parts = rel_path.split(os.sep)
            if len(parts) >= 2:
                category, filename = parts[-2], parts[-1]
                response = requests.post(
                    f"{base_url}/analyze-test?category={category}&filename={filename}",
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        "filename": filename,
                        "true_label": "real",
                        "authenticity_score": data["authenticity_score"],
                        "p_real": data["p_real"],
                        "p_fake": data["p_fake"],
                        "predicted_label": "real" if data["authenticity_score"] > 0.5 else "fake"
                    })
                    print(f"✓ {filename}: {data['authenticity_score']:.3f}")
                    tested_real += 1
                else:
                    print(f"✗ {filename}: Error {response.status_code}")
        except Exception as e:
            print(f"✗ {file_info['filename']}: {e}")
    
    print("\nTesting fake samples...")
    tested_fake = 0
    for file_info in fake_files:
        if tested_fake >= 50:
            break
        try:
            # Extract relative path for API
            rel_path = os.path.relpath(file_info['path'])
            parts = rel_path.split(os.sep)
            if len(parts) >= 2:
                category, filename = parts[-2], parts[-1]
                response = requests.post(
                    f"{base_url}/analyze-test?category={category}&filename={filename}",
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        "filename": filename,
                        "true_label": "fake",
                        "authenticity_score": data["authenticity_score"],
                        "p_real": data["p_real"],
                        "p_fake": data["p_fake"],
                        "predicted_label": "real" if data["authenticity_score"] > 0.5 else "fake"
                    })
                    print(f"✓ {filename}: {data['authenticity_score']:.3f}")
                    tested_fake += 1
                else:
                    print(f"✗ {filename}: Error {response.status_code}")
        except Exception as e:
            print(f"✗ {file_info['filename']}: {e}")
    
    # Calculate basic statistics
    real_scores = [r['authenticity_score'] for r in results if r['true_label'] == 'real']
    fake_scores = [r['authenticity_score'] for r in results if r['true_label'] == 'fake']
    
    print("\n" + "="*80)
    print("BASIC STATISTICS")
    print("="*80)
    print(f"Total samples tested: {len(results)}")
    print(f"Real samples: {len(real_scores)} (mean: {np.mean(real_scores):.3f}, std: {np.std(real_scores):.3f})")
    print(f"Fake samples: {len(fake_scores)} (mean: {np.mean(fake_scores):.3f}, std: {np.std(fake_scores):.3f})")
    
    # Test different thresholds
    thresholds = [0.30, 0.35, 0.40]
    threshold_results = {}
    
    print("\n" + "="*80)
    print("THRESHOLD COMPARISON")
    print("="*80)
    
    for threshold in thresholds:
        eval_result = evaluate_threshold(results, threshold)
        threshold_results[threshold] = eval_result
        
        print(f"\nThreshold: {threshold}")
        print(f"Accuracy: {eval_result['accuracy']:.3f}")
        print(f"Precision: {eval_result['precision']:.3f}")
        print(f"Recall: {eval_result['recall']:.3f}")
        print(f"F1 Score: {eval_result['f1']:.3f}")
        print(f"Confusion Matrix: TN={eval_result['true_negatives']}, FP={eval_result['false_positives']}, FN={eval_result['false_negatives']}, TP={eval_result['true_positives']}")
    
    # Find best threshold
    best_threshold = max(threshold_results.keys(), key=lambda t: threshold_results[t]['f1'])
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    print(f"Best threshold: {best_threshold}")
    print(f"Best F1 Score: {threshold_results[best_threshold]['f1']:.3f}")
    print(f"Best Accuracy: {threshold_results[best_threshold]['accuracy']:.3f}")
    
    # Save detailed results
    output_data = {
        'basic_stats': {
            'total_samples': len(results),
            'real_samples': len(real_scores),
            'fake_samples': len(fake_scores),
            'mean_real_score': float(np.mean(real_scores)),
            'mean_fake_score': float(np.mean(fake_scores)),
            'std_real_score': float(np.std(real_scores)),
            'std_fake_score': float(np.std(fake_scores))
        },
        'threshold_comparison': {
            str(t): {
                'threshold': float(t),
                'accuracy': float(v['accuracy']),
                'precision': float(v['precision']),
                'recall': float(v['recall']),
                'f1': float(v['f1']),
                'confusion_matrix': v['confusion_matrix'],
                'true_positives': int(v['true_positives']),
                'false_positives': int(v['false_positives']),
                'true_negatives': int(v['true_negatives']),
                'false_negatives': int(v['false_negatives'])
            } for t, v in threshold_results.items()
        },
        'individual_results': results
    }
    
    with open('detailed_model_evaluation.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to detailed_model_evaluation.json")
    
    return best_threshold

if __name__ == "__main__":
    best_threshold = evaluate_model()
    print(f"\nRecommended default threshold: {best_threshold}")