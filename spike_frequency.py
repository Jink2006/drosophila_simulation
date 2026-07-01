#網膜上の輝度変化をニューロンの最初の入力のスパイクに変換するような形式にする！！
import numpy as np
import matplotlib.pyplot as plt

def generate_poisson_spikes(stimulus_video, dt=0.01, lambda_base=0.0, lambda_max=100.0):
    """
    網膜の輝度変化（0.0~1.0）を、非一様ポアソン過程によるスパイク列（0 or 1）に変換する関数
    
    stimulus_video: (時間, Y, X) の3次元配列。1.0が白、0.0が黒(天敵)
    dt: 1コマあたりの時間（秒）。例: 0.01なら1コマ10ミリ秒
    lambda_base: 平常時の発火率 (Hz)
    lambda_max: 天敵に対する最大発火率 (Hz)
    """
    
    # 【ステップ1】明るさを「発火率(Hz)」に変換する（ベクトル化計算）
    # 映像が1.0(白)のときは lambda_base、0.0(黒)のときは lambda_max になる線形変換
    firing_rate_hz = lambda_base + (lambda_max - lambda_base) * (1.0 - stimulus_video)
    
    # 【ステップ2】発火率(Hz)を、この1コマ(dt)で発火する「確率」に変換
    spike_probability = firing_rate_hz * dt
    
    # 【ステップ3】確率的なくじ引き（ブロードキャストとマスク処理の応用）
    # 映像データと全く同じサイズ(50, 20, 20)の一様乱数(0.0~1.0)を一気に生成
    random_matrix = np.random.rand(*stimulus_video.shape)
    
    # 乱数が発火確率を下回っていれば「1(スパイクあり)」、そうでなければ「0(沈黙)」
    # この1行で 20,000回のくじ引きと条件判定を同時に行っている
    spikes = (random_matrix < spike_probability).astype(float)
    
    return spikes

if __name__ == "__main__":
    # --- 前回のコード（ダミーデータ）の代わり ---
    # 50コマ、20x20のダミー映像（最初は白1.0、だんだん黒0.0になっていく）を作る
    time_steps, grid_size = 50, 20
    t_array = np.linspace(0, 1, time_steps)
    dummy_video = 1.0 - t_array[:, None, None] * np.ones((time_steps, grid_size, grid_size))
    
    # --- スパイク生成の実行 ---
    # 1コマを10ミリ秒(0.01秒)として計算
    spike_train = generate_poisson_spikes(dummy_video, dt=0.01)
    
    # --- 結果の確認（特定の1ピクセルのスパイクを抽出） ---
    target_y, target_x = 10, 10  # ど真ん中の細胞を観察
    single_cell_spikes = spike_train[:, target_y, target_x]
    
    # グラフ描画
    plt.figure(figsize=(10, 3))
    # スパイクが1になっている時間(t)のインデックスを取得して縦線を引く
    spike_times = np.where(single_cell_spikes == 1)[0]
    plt.vlines(spike_times, ymin=0, ymax=1, colors='black')
    
    plt.title(f"Spike Train of a Single Retinal Cell (Grid: {target_y}, {target_x})")
    plt.xlabel("Time Step")
    plt.ylabel("Spike")
    plt.yticks([])
    plt.xlim(0, time_steps)
    plt.show()