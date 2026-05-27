"""Test script to evaluate model performance on real vs fake samples."""

import os
import json
from pathlib import Path
import requests
import numpy as np

def test_audio_samples():
    """Test real and fake audio samples and generate performance report."""
    
    # Test URLs
    base_url = "http://127.0.0.1:8010"
    
    # Sample files from test_audio directory
    real_samples = [
        "real_001.wav", "real_002.wav", "real_003.wav", "real_004.wav", "real_005.wav",
        "real_006.wav", "real_007.wav", "real_008.wav", "real_009.wav", "real_010.wav"
    ]
    
    fake_samples = [
        "fake_001.wav", "fake_002.wav", "fake_003.wav", "fake_004.wav", "fake_005.wav",
        "fake_006.wav", "fake_007.wav", "fake_008.wav", "fake_009.wav", "fake_010.wav"
    ]
    
    results = []
    
    print("Testing real samples...")
    for filename in real_samples:
        try:
            response = requests.post(
                f"{base_url}/analyze-test?category=real&filename={filename}",
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
            else:
                print(f"✗ {filename}: Error {response.status_code}")
        except Exception as e:
            print(f"✗ {filename}: {e}")
    
    print("\nTesting fake samples...")
    for filename in fake_samples:
        try:
            response = requests.post(
                f"{base_url}/analyze-test?category=fake&filename={filename}",
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
            else:
                print(f"✗ {filename}: Error {response.status_code}")
        except Exception as e:
            print(f"✗ {filename}: {e}")
    
    # Generate results table
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    print(f"{'filename':<15} {'true_label':<8} {'authenticity_score':<15} {'p_real':<8} {'p_fake':<8} {'predicted_label':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['filename']:<15} {result['true_label']:<8} {result['authenticity_score']:<15.3f} {result['p_real']:<8.3f} {result['p_fake']:<8.3f} {result['predicted_label']:<15}")
    
    # Calculate summary statistics
    real_scores = [r['authenticity_score'] for r in results if r['true_label'] == 'real']
    fake_scores = [r['authenticity_score'] for r in results if r['true_label'] == 'fake']
    
    correct_real = sum(1 for r in results if r['true_label'] == 'real' and r['predicted_label'] == 'real')
    correct_fake = sum(1 for r in results if r['true_label'] == 'fake' and r['predicted_label'] == 'fake')
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Mean real score: {np.mean(real_scores):.3f} (std: {np.std(real_scores):.3f})")
    print(f"Mean fake score: {np.mean(fake_scores):.3f} (std: {np.std(fake_scores):.3f})")
    print(f"Real samples correct: {correct_real}/10 ({correct_real*10:.0f}%)")
    print(f"Fake samples correct: {correct_fake}/10 ({correct_fake*10:.0f}%)")
    print(f"Total accuracy: {(correct_real + correct_fake)/20*100:.1f}%")
    
    # Suggest better threshold
    all_scores = real_scores + fake_scores
    if len(real_scores) > 0 and len(fake_scores) > 0:
        max_fake_score = max(fake_scores)
        min_real_score = min(real_scores)
        suggested_threshold = (max_fake_score + min_real_score) / 2
        print(f"\nSuggested threshold: {suggested_threshold:.3f}")
        print(f"(based on max fake score: {max_fake_score:.3f}, min real score: {min_real_score:.3f})")
        
        # Test with suggested threshold
        correct_real_new = sum(1 for r in real_scores if r > suggested_threshold)
        correct_fake_new = sum(1 for r in fake_scores if r <= suggested_threshold)
        print(f"Accuracy with suggested threshold: {(correct_real_new + correct_fake_new)/20*100:.1f}%")
    
    # Save results to file
    with open('model_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to model_evaluation_results.json")

if __name__ == "__main__":
    test_audio_samples()