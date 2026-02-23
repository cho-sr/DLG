"""
PNG 결과 이미지 뷰어
실험 결과 이미지들을 표시하는 스크립트
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def view_results():
    """Display saved PNG results"""
    
    results_dir = 'results'
    images = [
        ('comprehensive_metrics.png', 'Comprehensive FL Metrics & DLG Attack'),
        ('dlg_convergence.png', 'DLG Convergence Curves'),
        ('reconstruction_comparison.png', 'Reconstruction Quality Comparison')
    ]
    
    print("="*70)
    print("📊 실험 결과 이미지 뷰어")
    print("="*70)
    
    for filename, title in images:
        filepath = os.path.join(results_dir, filename)
        
        if os.path.exists(filepath):
            print(f"\n✅ {title}")
            print(f"   파일: {filepath}")
            
            # Display image
            fig, ax = plt.subplots(figsize=(15, 10))
            img = mpimg.imread(filepath)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.show()
        else:
            print(f"\n⚠️  {title}")
            print(f"   파일 없음: {filepath}")
            print(f"   먼저 'python main.py'를 실행하세요.")
    
    print("\n" + "="*70)
    print("모든 이미지를 확인했습니다.")
    print("="*70)


if __name__ == "__main__":
    view_results()
