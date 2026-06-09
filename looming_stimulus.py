import numpy as np
import matplotlib.pyplot as plt

def generate_looming_stimulus(grid_size=20, time_steps=50, max_radius=10):
    """
    ハエの網膜に投影されるLooming刺激（天敵の接近）を生成する関数
    grid_size: 視野の解像度 (N x N)
    time_steps: シミュレーションの時間ステップ数
    max_radius: 最終的な円の最大半径
    """
    # 3次元配列 (時間, Y座標, X座標) を背景(明るい=1)で初期化
    stimulus_video = np.ones((time_steps, grid_size, grid_size))
    
    # 視野の中心座標
    center_y, center_x = grid_size // 2, grid_size // 2
    
    # Y, Xのグリッド座標を生成
    y, x = np.ogrid[:grid_size, :grid_size]
    
    for t in range(time_steps):
        # 時間経過とともに半径が非線形に大きくなるモデル（簡単な二次関数で代用）
        # 実際は 2 * arctan(R / (D0 - v*t)) などの物理モデルを入れます
        current_radius = max_radius * (t / time_steps)**2
        
        # 中心からの距離を計算し、半径内のピクセルを黒(暗い=0)にする
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = distance_from_center <= current_radius
        
        stimulus_video[t][mask] = 0.0
        
    return stimulus_video

if __name__ == "__main__":
    # 実行して結果を確認
    grid_size = 20
    video = generate_looming_stimulus(grid_size=grid_size)
    
    # 最初、中間、最後のフレームをプロットして確認
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    frames_to_show = [0, 24, 49]
    
    for ax, frame_idx in zip(axes, frames_to_show):
        ax.imshow(video[frame_idx], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Time step: {frame_idx}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()