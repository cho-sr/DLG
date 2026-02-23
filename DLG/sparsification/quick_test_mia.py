# -*- coding: utf-8 -*-
"""
Quick Test: MIA Sparsification 빠른 테스트

작은 설정으로 빠르게 테스트하여 시스템 동작 확인
"""

import subprocess
import sys

def run_test(sparsity, label):
    """한 가지 sparsity로 테스트 실행"""
    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"{'='*60}")
    print(f"Sparsity: {sparsity*100}%\n")
    
    cmd = [
        'python', 'fedavg_mia.py',
        '--index', '25',
        '--sparsity', str(sparsity),
        '--mia_iters', '50',  # 빠른 테스트: 50 반복
        '--local_epochs', '1'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"\n✓ Test {label} completed successfully!")
            return True
        else:
            print(f"\n✗ Test {label} failed!")
            return False
    except Exception as e:
        print(f"\n✗ Test {label} error: {str(e)}")
        return False

def main():
    print("="*60)
    print("MIA SPARSIFICATION - QUICK TEST")
    print("="*60)
    print("\nThis quick test runs MIA with 50 iterations (instead of 300)")
    print("to quickly verify system functionality.\n")
    
    tests = [
        (1.0, "No Sparsification (100%)"),
        (0.1, "Light Sparsification (10%)"),
        (0.01, "Heavy Sparsification (1%)")
    ]
    
    results = {}
    
    for sparsity, label in tests:
        success = run_test(sparsity, label)
        results[label] = "PASSED" if success else "FAILED"
    
    # Print Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for label, status in results.items():
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {label}: {status}")
    
    passed = sum(1 for s in results.values() if s == "PASSED")
    total = len(results)
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! System is working correctly.")
        print("Next steps:")
        print("  1. Run full experiments: python batch_mia_experiments.py")
        print("  2. Compare results: python compare_mia_results.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
