import os
import sys
from neuron import h

# ==========================================
# 1. 前準備：出力用フォルダの自動作成（罠回避）
# ==========================================
os.makedirs("./result", exist_ok=True)
os.makedirs("./change", exist_ok=True)

# ==========================================
# 2. GUIと並列計算コンテキストの初期化
# ==========================================
h.load_file("nrngui.hoc")

# 【超重要修正】HOC側のグローバル空間にも「pc」というオブジェクト枠を強制実体化させる
h("objref pc")
h.pc = h.ParallelContext()
pc = h.pc  # Python側からも今まで通り pc.xxx で呼べるようにリンク

# ==========================================
# 3. パラメータの初期設定 (HOCグローバル変数の確実な動的定義)
# ==========================================
def define_hoc_var(name, val):
    if not hasattr(h, name):
        # HOCのインタプリタに直接文字列を流し込んで「強制変数宣言」させる
        h(f"{name} = {val}")
    else:
        setattr(h, name, val)

# ① デフォルト値の上書き系
define_hoc_var("OUT1_E", 0.002)
define_hoc_var("OUT1_I", 0.02)
define_hoc_var("IO_E", 0.001)
define_hoc_var("IO_I2E", 0.008)
define_hoc_var("IO_I2I", 0.005)
define_hoc_var("OUT1_SPON_E_K", 0.12)
define_hoc_var("OUT1_SPON_E_T", 0.32)
define_hoc_var("DOPAMINE", 0.01)
define_hoc_var("LEARNING_RATE", 0.00025)
define_hoc_var("OUT1_SPON_I_K", 0.12)
define_hoc_var("OUT1_SPON_I_T", 0.25)
define_hoc_var("LTD", 1.0)

# ② シミュレーション規模・時間設定系
define_hoc_var("v_init", -65.0)
define_hoc_var("NCELL", 512)
define_hoc_var("NCELL_E", 256)
define_hoc_var("NCELL_OUT1", 400)
define_hoc_var("NCELL_OUT1_E", 320)
define_hoc_var("NCELL_OUT2", 100)
define_hoc_var("NCELL_OUT2_E", 80)
define_hoc_var("NCELL_VTA", 10)
define_hoc_var("NCELL_CS", 10)
define_hoc_var("NSYN", 10)
define_hoc_var("BRANCH_NUM", 10)

define_hoc_var("LEARNING_TIMES", 12) 
define_hoc_var("STIM_DUR", 1000)
define_hoc_var("STIM_DUR_MOVE", 1000000)
define_hoc_var("START_STIM", 20)

define_hoc_var("dt", 0.025)
define_hoc_var("tstop_max", 65000.0)
define_hoc_var("stimInt", 25)
define_hoc_var("stimNum", 36)

# 計算で求まるパラメータ（上の定義により h.NCELL が実体化したため計算可能）
define_hoc_var("NSYN_MAX", (h.NCELL - 1) * h.NSYN)
define_hoc_var("NCELL_FULL", h.NCELL + h.NCELL_OUT1 + h.NCELL_OUT2)

startTest = h.STIM_DUR * 5 * h.LEARNING_TIMES + 500
define_hoc_var("startTest", startTest)#pythonで定義された変数をhocでも定義できるように改良！
dur = h.tstop_max
define_hoc_var("vta_flag", 0)
define_hoc_var("else_flag", 2)
# ==========================================
# 4. サブモジュール（既存HOCファイル）の読み込み
# ==========================================
h.load_file("read.hoc")
h.load_file("cells.hoc")
h.load_file("connectCells.hoc")
h.load_file("setStim.hoc")
h.load_file("signalDA.hoc")

# ==========================================
# 5. ネットワークの構築（細胞作成と結合）
# ==========================================
h.read_mat(h.NCELL_E, h.NCELL_OUT1, h.NCELL_OUT2, h.BRANCH_NUM)
h.makeCells(h.NCELL_FULL, h.BRANCH_NUM)
h.makeConvert(h.NCELL_FULL)
pc.barrier()
print("finish make section")

h.connectInputCells(h.NCELL_E, h.NCELL_VTA, h.NCELL, h.NCELL_FULL, h.BRANCH_NUM)
print("connect input is end")
h.connectOutput1Cells(h.NCELL_OUT1_E, h.NCELL_VTA, h.NCELL_OUT1, h.NCELL, h.NCELL_FULL, h.BRANCH_NUM)
print("connect output1 is end")
h.connectOutput2Cells(h.NCELL_OUT2_E, h.NCELL_VTA, h.NCELL_OUT2, h.NCELL + h.NCELL_OUT1, h.NCELL_FULL, h.BRANCH_NUM)
print("connect output2 is end")
h.connectI2O(h.NCELL_E, h.NCELL_OUT1, h.NCELL_OUT1_E, h.NCELL_OUT2, h.NCELL_OUT2_E, h.NCELL_VTA, h.NCELL, h.BRANCH_NUM)
print("connect i2o is end")

# 配列のリサイズ処理
h.con_order_input.resize(1, 1)
h.con_order_out1.resize(1, 1)
h.con_order_out2.resize(1, 1)
h.con_order_i2o1.resize(1, 1)
h.con_order_i2o2.resize(1, 1)

pc.barrier()
h.set_LR()
pc.barrier()
print("finish connect section")

# 刺激の設定
h.setStim(h.stimNum, h.NCELL_VTA, dur, h.NCELL_CS, h.stimInt, h.NCELL_E, h.LEARNING_TIMES, h.START_STIM, h.NCELL_FULL)

# ==========================================
# 6. スパイク記録の設定 (PythonのVectorを使用)
# ==========================================
tvec = h.Vector()
idvec = h.Vector()

def spikerecord2_py():
    for i in range(int(h.NCELL_FULL)):
        if pc.gid_exists(i):
            pc.spike_record(i, tvec, idvec)

spikerecord2_py()

# ==========================================
# 7. シミュレーションの初期化
# ==========================================
h.finitialize(h.v_init)
h.fcurrent()
print(f"the first DA = {h.cells.object(0).synlist_e.object(0).forDA}")

pc.set_maxstep(10)
h.stdinit()

# ループ制御用の変数
time = 0.0
datiming = 5.0  # 5msごとの手動ループ
stim_timer = h.STIM_DUR
stim_timer_move = h.STIM_DUR_MOVE
stim_flag = 0
stim_flag_move = 0
stim_counter = 0

SECTION_OUT1_E = int(h.NCELL_OUT1_E / 5)
SECTION_OUT1_I = int((h.NCELL_OUT1 - h.NCELL_OUT1_E) / 5)
SECTION_OUT2_E = int(h.NCELL_OUT2_E / 4)
SECTION_OUT2_I = int((h.NCELL_OUT2 - h.NCELL_OUT2_E) / 4)

h.changeFlags1(0, 64, 320, 336, 400, 100, 512, 320, 80)
h.changeFlags2(0, 20, 80, 85, 400, 100, 512, 320, 80)
h.forTestOut1()

rec_timing = 0
rec_counter = 0
weight_rec_i2o = []  # 重みの履歴を保存するPythonのリスト
init_time = h.STIM_DUR
init_interval = h.STIM_DUR
print_time = 1000.0

# ==========================================
# 8. メイン実行ループ（強化学習コマ送り処理）
# ==========================================
while time < h.tstop_max:
    # 5ms分だけシミュレーションを進める
    pc.psolve(time + datiming)
    pc.barrier()
    
    # ドーパミン信号の更新（signalDA.hocの関数）
    h.signalDA(h.NCELL_VTA, h.NCELL_E, h.NSYN_MAX, h.NCELL_FULL, h.NCELL, h.NCELL_OUT1, h.NCELL_OUT2, h.NCELL_OUT1_E, h.NCELL_OUT2_E)
    
    if time >= init_time:
        init_time += init_interval
        h.initDA()
        
    if time >= print_time:
        print(f"now is {int(print_time)} [ms]")
        print_time += 1000.0
        
    if time >= startTest:
        startTest += h.tstop_max
        h.off_plast()
        pc.barrier()
        
    # 重みの定期記録処理
    rec_timing += 1
    if rec_timing == 40 and time < startTest:
        rec_timing = 0
        # 現時点の重みを一括取得
        current_weights = [h.nclist_i2o.o(nc_num).weight for nc_num in range(int(h.nclist_i2o.count()))]
        weight_rec_i2o.append(current_weights)
        rec_counter += 1
        
    # 刺激パターンの切り替え
    if time >= stim_timer:
        stim_timer += h.STIM_DUR
        stim_counter += 1
        if stim_counter <= int(h.stim_order.size()) - 1:
            stim_flag = int(h.stim_order.x[stim_counter])
        h.changeFlags1(0 + stim_flag * SECTION_OUT1_E, 64 + stim_flag * SECTION_OUT1_E,
                       320 + stim_flag * SECTION_OUT1_I, 336 + stim_flag * SECTION_OUT1_I,
                       400, 100, 512, 320, 80)
                       
    if time >= stim_timer_move:
        stim_timer_move += h.STIM_DUR_MOVE
        if stim_flag_move == 3:
            stim_flag_move = 0
        else:
            stim_flag_move += 1
        h.changeFlags2(0 + SECTION_OUT2_E * stim_flag_move, 20 + SECTION_OUT2_E * stim_flag_move,
                       80 + SECTION_OUT2_I * stim_flag_move, 85 + SECTION_OUT2_I * stim_flag_move,
                       400, 100, 512, 320, 80)
                       
    time += datiming

pc.barrier()
print(f"the last da = {h.cells.object(0).synlist_e.object(0).forDA}")

# ==========================================
# 9. 結果のファイル保存（Python標準機能でスマートに記述）
# ==========================================
rank = int(pc.id())

# スパイクデータの保存
spike_filename = f"./result/spike{rank}.txt"
with open(spike_filename, "w") as f:
    for i in range(int(tvec.size())):
        if idvec.x[i] < h.NCELL_FULL:
            f.write(f"{tvec.x[i]}\t{idvec.x[i]}\n")

# 最終的な重みの保存
def save_weights(nclist_obj, filename_str):
    with open(f"./result/{filename_str}{rank}.dat", "w") as f:
        for i in range(int(nclist_obj.count())):
            f.write(f"{nclist_obj.object(i).weight}\t")

save_weights(h.nclist, "nclist")
save_weights(h.nclist_out1, "nclist_out1_")
save_weights(h.nclist_out2, "nclist_out2_")
save_weights(h.nclist_i2o, "nclist_i2o_")

# 重みの時間変化（履歴）の保存
change_filename = f"./change/change_i2o_{rank}.dat"
with open(change_filename, "w") as f:
    for nc_num in range(int(h.nclist_i2o.count())):
        f.write(f"{rank}\t{nc_num}\t")
        for j in range(rec_counter):
            f.write(f"{weight_rec_i2o[j][nc_num]}\t")
        f.write("\n")

# 終了処理
pc.runworker()
pc.done()
print("Simulation successfully finished!")