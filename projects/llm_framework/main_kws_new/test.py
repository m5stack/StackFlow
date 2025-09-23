import time
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import numpy as np
import onnxruntime as ort
import os
import glob

frame_index_global = 0
last_trigger_time_ms = -1e9
count = 0  # 添加缺失的全局计数器

def detect_wakeup(scores_chunk):
    global count, last_trigger_time_ms, frame_index_global
    above = scores_chunk > threshold
    triggered = False
    for flag in above:
        if flag:
            count += 1
            if count >= min_continuous_frames:
                trigger_time_ms = (frame_index_global - min_continuous_frames + 1) * 10
                if (trigger_time_ms - last_trigger_time_ms) >= 2000:
                    print(f"[唤醒触发] 时间: {trigger_time_ms} ms")
                    last_trigger_time_ms = trigger_time_ms
                    triggered = True
        else:
            count = 0
        frame_index_global += 1
    return triggered

def save_mat_to_bin(mat_data, index):
    """保存mat数据为二进制文件，格式与C++兼容"""
    filename = f"mat_{index:03d}.bin"
    
    # mat_data是torch tensor，shape为 [frames, features]
    mat_np = mat_data.numpy()
    rows, cols = mat_np.shape
    
    with open(filename, 'wb') as f:
        # 写入维度信息 (int64)
        f.write(np.array([rows], dtype=np.int64).tobytes())
        f.write(np.array([cols], dtype=np.int64).tobytes())
        # 写入数据 (float32)
        f.write(mat_np.astype(np.float32).tobytes())
    
    print(f"保存mat到 {filename}, shape: ({rows}, {cols})")
    return filename

# ================= 配置参数 =================
wavfile = "./001.wav"                  # 测试音频路径
onnx_model = "./t30.onnx"          # ONNX 模型路径
feat_dim = 80                          # FBank 维度
resample_rate = 16000                  # 目标采样率
chunk_size = 32                        # 推理帧数（1 帧=10ms，32帧≈320ms）
threshold = 0.9                        # 唤醒分数阈值
min_continuous_frames = 5              # 连续高分帧数(>=此值判定唤醒)
device = "cpu"

# ================= 加载模型 =================
ort_sess = ort.InferenceSession(onnx_model)

# ================= 加载音频并预处理 =================
waveform, sample_rate = torchaudio.load(wavfile)
# 音量归一
waveform = waveform * (1 << 15)

cache = torch.zeros(1, 32, 88, dtype=torch.float).numpy()  # 初始化 cache
scores = []          # 存储每帧分数
frame_index = 0      # 已处理全局帧数
save_index = 0
save_dir = "debug_inputs"
os.makedirs(save_dir, exist_ok=True)

samples_per_chunk = int(resample_rate * 0.01 * chunk_size)  # 每个推理块采样点数
total_samples = waveform.shape[1]
print(f"音频总采样点数: {total_samples}, 每块点数: {samples_per_chunk}")

start_sample = 0
while start_sample < total_samples:
    end_sample = start_sample + samples_per_chunk
    audio_chunk = waveform[:, start_sample:end_sample]  # [1, T]
    start_sample += samples_per_chunk
    frame_index += chunk_size  # 累加帧数
    
    # ----------- run_inference1 逻辑的 fbank 计算 -----------
    mat = kaldi.fbank(
        audio_chunk,
        num_mel_bins=feat_dim,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=resample_rate,
        snip_edges=False
    )
    
    print(f"mat.shape={mat.shape}")
    
    # 保存mat为二进制文件
    save_mat_to_bin(mat, save_index)
    
    # 推理
    onnx_inputs = {
        "input": mat.unsqueeze(0).numpy(),
        "cache": cache
    }
    onnx_output = ort_sess.run(None, onnx_inputs)
    out_chunk, cache = onnx_output[0].flatten(), onnx_output[1]  # 更新 cache
    
    if detect_wakeup(out_chunk):
        print(f"[{time.strftime('%H:%M:%S')}] 检测到唤醒词！")
    
    scores.extend(out_chunk)
    save_index += 1

print(f"总共保存了 {save_index} 个mat文件")
