# singingvis_api.py

# --- 網頁與系統 ---
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import shutil
import subprocess
import csv
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, Counter
import sys

# --- 音訊處理 ---
import librosa
import soundfile as sf
import sounddevice as sd
import parselmouth
from pydub import AudioSegment
import yt_dlp

# --- 資料處理與科學計算 ---
import numpy as np
import pandas as pd
import json
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 文字、語音與 NLP ---
import whisper
import jieba
import opencc
import re
from pypinyin import lazy_pinyin, pinyin, Style
import pronouncing

# --- 視覺化與報表 ---
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import openpyxl

app = Flask(__name__)
CORS(app)

CURRENT_FILE_PATH = os.path.abspath(__file__) 
# 獲取檔案所在的目錄 (/SingingVis/src/flask)
CURRENT_DIR = os.path.dirname(CURRENT_FILE_PATH) 
# 根據檔案的位置，向上跳一級 ('..')，然後找到 'original_audio' (/SingingVis/src/)
PARENT_DIR = os.path.join(CURRENT_DIR, '..') 

model = whisper.load_model("medium")  # 可改為 "medium" 或 "large"

# --- API 路由設定 ---

# 取得音訊檔案
@app.route('/api/getAudio/<path:folder>/<folder_name>/<filename>')
def get_audio(folder, folder_name, filename):
    # 💡 指向你存放 original_audio 的絕對路徑
    TARGET_DIR = os.path.join(PARENT_DIR, folder, folder_name)
    
    # 確保檔案存在，否則回傳 404
    if not os.path.exists(os.path.join(TARGET_DIR, filename)):
        return "File not found", 404

    # 使用 send_from_directory 會自動處理 Range Requests (對播放進度條很重要)
    return send_from_directory(TARGET_DIR, filename)

# 取得 csv 檔案
@app.route('/api/getCsv/<path:filePath>')
def get_csv(filePath):
    # 這裡指向你實際存放 CSV 的後端路徑
    csv_path = os.path.join(PARENT_DIR, filePath)
    return send_file(csv_path, mimetype='text/csv')

# 取得 JSON 檔案
@app.route('/api/getJson/<path:filePath>')
def get_json(filePath):
    # 這裡指向你實際存放 JSON 的後端路徑
    json_path = os.path.join(PARENT_DIR, filePath)
    
    # 檢查檔案是否存在，避免回傳 500 錯誤
    if not os.path.exists(json_path):
        return {"error": "File not found"}, 404
        
    return send_file(json_path, mimetype='application/json')

# 全域變數存儲當前進度
current_task_status = {"progress": 0, "status": "等待中"}

# 取得上傳進度
@app.route('/api/getUploadProgress', methods=['GET'])
def getUploadProgress():
    return jsonify(current_task_status)

# ------ 音訊分析與比對相關 API ------

# 產生比較原曲後的資料
@app.route('/api/getCompareData', methods=['POST'])
def getCompareData():
    # 1. 接收文字欄位 (使用 request.form)
    song_name = request.form.get('song_name')
    # 2. 接收檔案欄位 (使用 request.files)
    audio_file = request.files.get('audio_file')

    original_audio_path = os.path.join(PARENT_DIR, 'original_audio', song_name)
    upload_audio_path = os.path.join(PARENT_DIR, 'upload_audio', song_name)

    # --- 建立資料夾 ---
    try:
        # 使用 FINAL_PATH 來建立目標路徑
        # os.makedirs 會建立路徑中的所有不存在的父目錄 (例如：如果 original_audio 不存在也會一併建立)
        os.makedirs(upload_audio_path, exist_ok=True)
        print(f"資料夾 '{song_name}' 已成功在 '{upload_audio_path}' 內建立或已存在。")
    except Exception as e:
        print(f"in建立資料夾時發生錯誤: {e}")

    global current_task_status
    current_task_status = {"progress": 10, "status": "正在平衡音量 ..."}

    # 調整音量與原音檔一致
    file1 = AudioSegment.from_file(audio_file)  # 需要調整的音檔
    file2 = AudioSegment.from_file(f'{original_audio_path}/vocals.wav')  # 目標音量的音檔

    # 計算 RMS
    rms1 = rms(file1)
    rms2 = rms(file2)

    # 計算 dB 差異
    change_in_dB = 20 * np.log10(rms2 / rms1)

    # 調整音量
    file1_matched = file1.apply_gain(change_in_dB + 2)

    # 輸出
    file1_matched.export(f'{upload_audio_path}/vocals.wav', format="wav")

    current_task_status = {"progress": 20, "status": "正在分析上傳音檔 ..."}
    # 匯出 vocals.csv 及 timbre.csv
    vocal_analysis(upload_audio_path)
    analyze_timbre(upload_audio_path)

    current_task_status = {"progress": 40, "status": "正在比對最佳 offset ..."}
    # 參數設定
    window_size = 0.5  # 窗口
    step_size = 0.05   # 一次移動多少
    search_range = 10  # 搜尋 offset 的秒數範圍

    # 讀檔
    t1, p1, n1, m1, mc1, mu1, r1, o1 = load_note_csv(f"{original_audio_path}/vocals.csv")
    t2_orig, p2_orig, n2_orig, m2_orig, mc2_orig, mu2_orig, r2_orig, o2_orig = load_note_csv(f"{upload_audio_path}/vocals.csv")

    offsets = np.arange(-search_range, search_range, step_size)
    best_offset = 0 # 初始化最佳 offset
    min_mismatch = float('inf')

    # 改用單進程優化運算 bestOffset，避免系統崩潰
    for offset in offsets:
        # 直接在 NumPy 數組上操作，不要呼叫 shift_times 建立 DataFrame
        t2_shifted = t2_orig + offset
        
        # 快速計算 mismatch (這裡建議簡化邏輯以提升速度)
        _, mismatch_rate = process_single_offset(offset, t2_shifted, n2_orig, t1, n1, window_size, step_size)
        
        if mismatch_rate < min_mismatch:
            min_mismatch = mismatch_rate
            best_offset = offset

    best_offset_path = os.path.join(upload_audio_path, 'bestOffset.json')
    # 1. 檢查檔案是否存在，若不存在則建立初始 JSON
    if not os.path.exists(best_offset_path):
        initial_data = {"bestOffset": best_offset} # 建立基礎結構
        with open(best_offset_path, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=4, ensure_ascii=False)
        print(f"已建立新的設定檔: {best_offset_path}")
    # 讀取現有的 bestOffset.json
    else:
        with open(best_offset_path, 'r', encoding='utf-8') as f:
            best_offset_json = json.load(f)

        # 3. 更新數值
        best_offset_json["bestOffset"] = best_offset

        # 4. 寫回檔案
        with open(best_offset_path, 'w', encoding='utf-8') as f:
            json.dump(best_offset_json, f, indent=4, ensure_ascii=False)
        print(f"成功更新偏移量為: {best_offset}")

    # best_offset 時間偏移 
    t2, p2, n2, m2, mc2, mu2, r2, o2 = shift_times(t2_orig, p2_orig, n2_orig, m2_orig, mc2_orig, mu2_orig, r2_orig, o2_orig, best_offset)

    current_task_status = {"progress": 60, "status": "正在比對音準 ..."}
    # 比對音準
    df_note_compare = note_compare(t1, p1, t2, p2, window_size, step_size)
    # 篩選音準差異超過閥值區段
    #df_note_mismatch = df_note_compare[(df_note_compare['note_mismatch_range'] > 0.5) | (df_note_compare['note_mismatch_range'] < -0.5)]

    current_task_status = {"progress": 70, "status": "正在比對轉音 ..."}
    # 比對轉音
    df_melisma_compare = melisma_compare(t1, mc1, mu1, t2, mc2, mu2, step_size)
    # 篩選轉音差異超過閥值區段
    #df_melisma_mismatch = df_melisma_compare[df_melisma_compare['melisma_mismatch_ratio'] > 0.5]

    current_task_status = {"progress": 80, "status": "正在比對音量 ..."}
    # 比較 RMS
    df_rms_compare = rms_compare(t1, r1, t2, r2, window_size, step_size)
    # 篩選出錯誤時間點
    #df_rms_mismatch = df_rms_compare[(df_rms_compare['rms_mismatch_range'] > 0.3) | (df_rms_compare['rms_mismatch_range'] < -0.3)]

    current_task_status = {"progress": 90, "status": "正在比對節奏 ..."}
    # 比較節奏
    df_onset_compare = onset_compare(t1, o1, t2, o2)

    # 錯誤的地方建立 DataFrame，要加入到 overview
    df_main = pd.DataFrame({'time': t1})

    # 將 note_compare、melisma_compare、rms_compare 的 time 欄位設為 index，然後 reindex 到 t1
    df_main['noteCompare'] = pd.Series(df_note_compare.assign(time=np.round(df_note_compare['time'], 2)).set_index('time')['note_mismatch_range']).reindex(t1).values
    df_main['melismaChangeCompare'] = pd.Series(df_melisma_compare.assign(time=np.round(df_melisma_compare['time'], 2)).set_index('time')['melisma_mismatch_change_range']).reindex(t1).values
    df_main['melismaUniqueCompare'] = pd.Series(df_melisma_compare.assign(time=np.round(df_melisma_compare['time'], 2)).set_index('time')['melisma_mismatch_unique_range']).reindex(t1).values
    df_main['rmsCompare'] = pd.Series(df_rms_compare.assign(time=np.round(df_rms_compare['time'], 2)).set_index('time')['rms_mismatch_range']).reindex(t1).values

    # 匯出 compare.csv
    df_main.to_csv(f"{upload_audio_path}/compare.csv", index=False, encoding="utf-8-sig")

    # 匯出分段分析資料
    current_task_status = {"progress": 95, "status": "正在生成分段報告 ..."}
    analyze_segment_mismatch(df_main, original_audio_path, upload_audio_path)

    return jsonify({
        'bestOffset': best_offset,
        'noteCompare': df_note_compare.to_dict(),
        'melismaCompare': df_melisma_compare.to_dict(),
        'rmsCompare': df_rms_compare.to_dict(),
        'onsetCompare': df_onset_compare.to_dict(),
    })

# 定義一個處理單一偏移量的函式
def process_single_offset(offset, t2_shifted, n2_orig, t1, n1, window_size, step_size):
    # 這裡的邏輯盡量使用 NumPy 運算
    df_res = offset_compare(t1, n1, t2_shifted, n2_orig, window_size, step_size)
    return offset, df_res['note_mismatch_ratio'].sum()

# 讀 csv
def load_note_csv(path):
    df = pd.read_csv(path)
    return df['Time (s)'].values, df['MIDI Pitch'].values, df['Note'].values, df['Melisma'].values, df['Melisma_change'].values, df['Melisma_unique'].values, df['RMS'].values, df['Onset'].values

# 偏移時間
def shift_times(times, pitch, notes, melisma, melisma_change, melisma_unique, rms, onset, shift_sec):
    df = pd.DataFrame({'Time (s)': times, 'MIDI Pitch': pitch, 'Note': notes, 'Melisma': melisma, 'Melisma_change': melisma_change, 'Melisma_unique': melisma_unique, 'RMS': rms, 'Onset': onset})
    df['Time (s)'] += shift_sec
    return df['Time (s)'].values, df['MIDI Pitch'].values, df['Note'].values, df['Melisma'].values, df['Melisma_change'].values, df['Melisma_unique'].values, df['RMS'].values, df['Onset'].values

# 比對 offset
def offset_compare(t1, n1, t2, n2, window_size, step_size):
    # 設定起始、結束時間點
    start = max(t1[0], t2[0])
    end = min(t1[-1], t2[-1])
    
    results = []
    # 窗口起始位置
    cur = start
    while cur + window_size <= end:
        # 抓窗口內的時間
        mask1 = (t1 >= cur) & (t1 < cur + window_size)
        mask2 = (t2 >= cur) & (t2 < cur + window_size)
        
        # 抓窗口內的 note
        seg1 = n1[mask1]
        seg2 = n2[mask2]

         # 處理不同長度或空段情況
        min_len = min(len(seg1), len(seg2))
        seg1 = seg1[:min_len]
        seg2 = seg2[:min_len]
        
        # 計算有幾個不相同
        mismatches = np.sum(
            ((seg1 != seg2) & ~(pd.isna(seg1) | pd.isna(seg2))) |  # 值不同且不是 NaN
            (pd.isna(seg1) ^ pd.isna(seg2))                        # 僅一個是 NaN
        )
        # 計算錯誤比例
        mismatch_ratio = mismatches / len(seg1)
        results.append((cur, mismatch_ratio))
        
        # 往下個時間點
        cur += step_size

    return pd.DataFrame(results, columns=["time", "note_mismatch_ratio"])

# 比較音準
def note_compare(t1, p1, t2, p2, window_size, step_size):
    # 設定起始、結束時間點
    start = max(t1[0], t2[0])
    end = min(t1[-1], t2[-1])
    
    results = []
    # 窗口起始位置
    cur = start
    while cur + window_size <= end:
        # 抓窗口內的時間
        mask1 = (t1 >= cur) & (t1 < cur + window_size)
        mask2 = (t2 >= cur) & (t2 < cur + window_size)

        seg1 = p1[mask1]
        seg2 = p2[mask2]

        min_len = min(len(seg1), len(seg2))
        seg1 = seg1[:min_len]
        seg2 = seg2[:min_len]

        #用移動平均計算
        if np.all(np.isnan(seg1)):
            average1 = 0
        else:
            average1 = np.nanmean(seg1)

        if np.all(np.isnan(seg2)):
            average2 = 0
        else:
            average2 = np.nanmean(seg2)

        # 如果目前不為 Nan 才判斷
        if ~np.isnan(seg1[0]) or ~np.isnan(seg2[0]):
            results.append((cur, average1 - average2))
        
        # 往下個時間點
        cur += step_size

    return pd.DataFrame(results, columns=["time", "note_mismatch_range"])

# 比較轉音
def melisma_compare(t1, mc1, mu1, t2, mc2, mu2, step_size):
    start = max(t1[0], t2[0])
    end = min(t1[-1], t2[-1])

    results = []
    cur = start

    while cur <= end:
        # 選取時間窗口內的 Melisma
        mask1 = np.isclose(t1, cur, atol=1e-6)
        mask2 = np.isclose(t2, cur, atol=1e-6)

        Melisma_change1 = mc1[mask1][0]
        Melisma_change2 = mc2[mask2][0]

        Melisma_unique1 = mu1[mask1][0]
        Melisma_unique2 = mu2[mask2][0]

        # 有超過轉音標準的才相減
        if (Melisma_change1 > 3 and Melisma_unique1 > 3) or (Melisma_change2 > 3 and Melisma_unique2 > 3):
            results.append((cur, Melisma_change1 - Melisma_change2, Melisma_unique1 - Melisma_unique2))

        cur += step_size

    return pd.DataFrame(results, columns=["time", "melisma_mismatch_change_range", "melisma_mismatch_unique_range"])

# 比較音量
def rms_compare(t1, r1, t2, r2, window_size, step_size):
    # 設定起始、結束時間點
    start = max(t1[0], t2[0])
    end = min(t1[-1], t2[-1])
    
    results = []
    # 窗口起始位置
    cur = start
    while cur + window_size <= end:
        # 抓窗口內的時間
        mask1 = (t1 >= cur) & (t1 < cur + window_size)
        mask2 = (t2 >= cur) & (t2 < cur + window_size)

        seg1 = r1[mask1]
        seg2 = r2[mask2]

        min_len = min(len(seg1), len(seg2))
        seg1 = seg1[:min_len]
        seg2 = seg2[:min_len]

        #用移動平均計算
        if np.all(np.isnan(seg1)):
            average1 = 0
        else:
            average1 = np.nanmean(seg1)

        if np.all(np.isnan(seg2)):
            average2 = 0
        else:
            average2 = np.nanmean(seg2)

        results.append((cur, average1 - average2))
        
        # 往下個時間點
        cur += step_size

    return pd.DataFrame(results, columns=["time", "rms_mismatch_range"])

# 比較節奏（起音點）
def onset_compare(t1, o1, t2, o2):
    # 取出起音點時間
    onsets1 = np.array(t1)[o1]
    onsets2 = np.array(t2)[o2]

    time = []
    nearest_times = []
    diffs = []

    for t in onsets1:
        nearest = onsets2[np.argmin(np.abs(onsets2 - t))]
        time.append(t)
        nearest_times.append(nearest)
        diffs.append(t - nearest)

    # 建立 DataFrame
    df = pd.DataFrame({
        "time": time,
        "nearest_time": nearest_times,
        "onset_mismatch_range": diffs,
    })

    return df

# 分段整理錯誤
def analyze_segment_mismatch(df_main, original_audio_path, upload_audio_path):
    """
    依照 overview.csv 的區段，統計 df_main 中的各類錯誤指標。
    僅保留 segment_id, start, end 欄位，並匯出報告。
    """
    # 根據您的程式碼，來源改為 overview.csv
    segment_file_path = os.path.join(original_audio_path, 'overview.csv')
    
    if not os.path.exists(segment_file_path):
        print(f"跳過分段分析：找不到檔案 {segment_file_path}")
        return None

    # 1. 讀取原始資料
    df_raw = pd.read_csv(segment_file_path)
    
    # 2. 💡 僅保留所需的欄位 (segment_id, start, end)
    # 使用 .copy() 確保不會對原始 DataFrame 造成影響
    df_segments = df_raw[['segment_id', 'start', 'end']].copy()
    
    # 準備儲存各段錯誤總和的列表
    results = {
        'note_mismatch': [],
        'melisma_change_mismatch': [],
        'melisma_unique_mismatch': [],
        'rms_mismatch': []
    }

    # 3. 遍歷區段進行計算
    for _, row in df_segments.iterrows():
        start_t = row['start']
        end_t = row['end']
        
        # 篩選該時間區間內的錯誤資料
        mask = (df_main['time'] >= start_t) & (df_main['time'] <= end_t)
        seg_data = df_main.loc[mask]
        
        if not seg_data.empty:
            # 依照您提供的邏輯使用 .abs().sum()
            results['note_mismatch'].append(seg_data['noteCompare'].abs().sum())
            results['melisma_change_mismatch'].append(seg_data['melismaChangeCompare'].abs().sum())
            results['melisma_unique_mismatch'].append(seg_data['melismaUniqueCompare'].abs().sum())
            results['rms_mismatch'].append(seg_data['rmsCompare'].abs().sum())
        else:
            for key in results:
                results[key].append(0)

    # 4. 將計算結果合併回篩選後的 DataFrame
    for key, values in results.items():
        df_segments[key] = values

    # 5. 匯出至上傳資料夾
    output_path = os.path.join(upload_audio_path, 'overview_mismatch.csv')
    df_segments.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"分段錯誤分析完成 (僅保留核心欄位)，已匯出至: {output_path}")
    
    return df_segments

# 更改音域分界
@app.route('/api/changePitchSegmentation', methods=['POST'])
def changePitchSegmentation():
    # 讀取前端傳來的 JSON 資料
    data = request.get_json()

    # 定義最終目標目錄：PARENT_DIR/original_audio/
    TARGET_DIR = os.path.join(PARENT_DIR, 'original_audio')

    # --- 輸入變數 ---
    p_high = data.get('pitchUp', '')
    p_low = data.get('pitchDown', '')

    # 讀 pitch_segmentation.json
    pitch_cut_path = os.path.join(TARGET_DIR, 'pitch_segmentation.json')
    with open(pitch_cut_path, 'r', encoding='utf-8') as f:
        pitch_cut = json.load(f)

    global current_task_status
    current_task_status = {"progress": 50, "status": "正在重新產生所有概覽資料 ..."}

    # 先重新填分段點
    pitch_cut["pitch_high"] = p_high
    pitch_cut["pitch_low"] = p_low
    with open(pitch_cut_path, 'w', encoding='utf-8') as f:
        # indent=4 讓產出的 JSON 檔案有縮進，人類比較好閱讀
        # ensure_ascii=False 確保中文字不會變成 \u 碼
        json.dump(pitch_cut, f, indent=4, ensure_ascii=False)

    resetSegment(p_low, p_high)

    return jsonify({'status': 'success', 'message': '概覽資料已重新產生。'}), 200

# 上傳音檔資料
@app.route('/api/uploadSongData', methods=['POST'])
def uploadSongData():
    """
    處理並保存上傳的音檔資料。
    """
    data = request.get_json()

    # --- 輸入變數 ---
    song_name = data.get('title', '未知歌曲')
    artist_name = data.get('singer', '未知歌手')
    lyrics = data.get('lyric', '')
    youtube_link = data.get('youtubeUrl', '')
    p_high = data.get('pitchUp', '')
    p_low = data.get('pitchDown', '')

    folder_name = f"{song_name}_{artist_name}"

    # 定義最終目標目錄：PARENT_DIR/original_audio/
    TARGET_DIR = os.path.join(PARENT_DIR, 'original_audio')

    # 建立最終資料夾的完整路徑：TARGET_DIR/folder_name/
    FINAL_PATH = os.path.join(TARGET_DIR, folder_name)

    # --- 執行建立資料夾的動作 ---
    try:
        # 使用 FINAL_PATH 來建立目標路徑
        # os.makedirs 會建立路徑中的所有不存在的父目錄 (例如：如果 original_audio 不存在也會一併建立)
        os.makedirs(FINAL_PATH, exist_ok=True)
        print(f"資料夾 '{folder_name}' 已成功在 '{TARGET_DIR}' 內建立或已存在。")
        print(f"完整路徑為: {FINAL_PATH}")
        
    except Exception as e:
        print(f"in建立資料夾時發生錯誤: {e}")

    # 設定路徑與資料
    csv_file_path = os.path.join(TARGET_DIR, 'song_data.csv')
    new_data = {
        'song': song_name,
        'singer': artist_name,
        'lyric': lyrics.strip(),
        'youtube_url': youtube_link,
    }

    try:
        # 1. 讀取現有資料
        if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
            df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
        else:
            # 如果檔案不存在，建立一個空的 DataFrame 並指定欄位
            df = pd.DataFrame(columns=['song', 'singer', 'lyric', 'youtube_url'])

        # 2. 判斷是否存在重複資料 (歌名與歌手同時符合)
        condition = (df['song'] == song_name) & (df['singer'] == artist_name)

        if not df[condition].empty:
            # --- 情況 A：重複，進行替換 (Update) ---
            # 遍歷 new_data 中的所有 key，將新值覆蓋到符合條件的該列
            for key, value in new_data.items():
                df.loc[condition, key] = value
            print(f"🔄 資料已存在，已成功更新 '{song_name}' - '{artist_name}' 的內容。")
        else:
            # --- 情況 B：不重複，新增列 (Insert) ---
            new_row = pd.DataFrame([new_data])
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"✅ 資料不存在，已成功新增 '{song_name}' 到資料庫。")

        # 3. 寫回 CSV (覆蓋存檔)
        # index=False 避免產生存檔時多出一列數字索引
        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

    # --- 音訊分析處理流程 ---
    global current_task_status

    try:

        current_task_status = {"progress": 10, "status": "正在下載 YouTube 音訊 ..."}    
        #下載mp3
        download_audio(youtube_link, folder_name, FINAL_PATH)

        current_task_status = {"progress": 20, "status": "正在進行人聲分離 (Demucs) ..."}
        # 執行分割並匯出人聲音訊
        demucsAudio(FINAL_PATH, f'{folder_name}.mp3')
        
        current_task_status = {"progress": 30, "status": "正在進行音訊分析 ..."}
        # 執行分析並匯出 csv
        vocal_analysis(FINAL_PATH)

        current_task_status = {"progress": 40, "status": "正在根據歌詞進行音訊分段 ..."}
        # 執行音訊分段
        segmentAudio(FINAL_PATH, lyrics)

        current_task_status = {"progress": 70, "status": "正在產生音色資料 ..."}
        # 產生 timbre.csv
        analyze_timbre(FINAL_PATH)

        current_task_status = {"progress": 80, "status": "正在產生歌詞資料 ..."}
        # 產生 lyrics.csv
        preprocess_to_rows(lyrics, FINAL_PATH)

        # 讀 pitch_segmentation.json
        pitch_cut_path = os.path.join(TARGET_DIR, 'pitch_segmentation.json')
        with open(pitch_cut_path, 'r', encoding='utf-8') as f:
            pitch_cut = json.load(f)

        # 如果分段點一樣
        if pitch_cut['pitch_high'] == p_high and pitch_cut['pitch_low'] == p_low:
            current_task_status = {"progress": 90, "status": "正在產生概覽資料 ..."}

            # 請確保這兩個檔案都在同一個目錄下
            data_file_path = f'{FINAL_PATH}/vocals.csv'
            segment_file_path = f'{FINAL_PATH}/segment.csv'
            audio_file_path = f'{FINAL_PATH}/vocals.wav'
            output_file = f'{FINAL_PATH}/overview.csv'
            # 產生 overview.csv
            analyze_music_data_by_segments(data_file_path, segment_file_path, audio_file_path, p_low, p_high, output_file)
        # 分段點不一樣，所有資料夾重新跑 overview
        else:
            current_task_status = {"progress": 70, "status": "因分段點不同，正在重新產生所有概覽資料 ..."}

            # 先重新填分段點
            pitch_cut["pitch_high"] = p_high
            pitch_cut["pitch_low"] = p_low
            with open(pitch_cut_path, 'w', encoding='utf-8') as f:
                # indent=4 讓產出的 JSON 檔案有縮進，人類比較好閱讀
                # ensure_ascii=False 確保中文字不會變成 \u 碼
                json.dump(pitch_cut, f, indent=4, ensure_ascii=False)

            resetSegment(p_low, p_high)
        
        return jsonify({'status': 'success', 'message': '音檔資料已成功匯入並處理。'}), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': '音檔資料處理失敗。'}), 500

# 下載音訊檔案
def download_audio(url, filename, output_path):
    # 更新 yt-dlp 核心 (保持 nightly)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--pre", "yt-dlp[default]"])

    # 設置輸出模板：{輸出目錄}/{自訂檔案名稱}
    # yt-dlp 的 FFmpeg postprocessor 會自動在檔名後加上正確的副檔名 (.mp3)
    outtmpl_path = os.path.join(output_path, filename)

    ydl_opts = {
        'format': 'bestaudio/best',
        'ffmpeg_location': '/usr/bin/ffmpeg',
        'outtmpl': outtmpl_path, # <--- 這裡改用自訂的路徑和檔名
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'cookiesfrombrowser': ('chrome',),
    }

    # 下載音訊
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"音訊已成功下載，檔案名稱將是: {filename}.mp3")
    except Exception as e:
        print(f"下載失敗: {e}")

# 使用 Demucs 分割音檔
def demucsAudio(path, mp3_file):
    input_path = os.path.join(path, mp3_file)
    file_base_name = os.path.splitext(mp3_file)[0]
    
    # Demucs 預設會產生 separated/htdemucs/檔名/ 的目錄

    print(f"正在使用 Demucs 處理：{input_path}")

    try:
        # --two-stems=vocals 代表只拆成 人聲 + 伴奏
        subprocess.run([
            sys.executable, "-m", "demucs",  # 👈 使用 python -m demucs 啟動
            "--device", "gpu",               # ubuntu 用 cpu 或 gpu
            "--two-stems", "vocals", 
            "-o", path, 
            input_path
        ], check=True)

        # 找到 Demucs 產生的檔案路徑 (通常在 separated/htdemucs/檔案名稱/ 下)
        # 注意：Demucs 預設模型名稱可能是 htdemucs 或 demucs
        demucs_out_folder = os.path.join(path, "htdemucs", file_base_name)
        
        src_vocals = os.path.join(demucs_out_folder, "vocals.wav")
        src_no_vocals = os.path.join(demucs_out_folder, "no_vocals.wav")

        # 移動到你指定的 path 之下，並改名
        shutil.move(src_vocals, os.path.join(path, "vocals.wav"))
        shutil.move(src_no_vocals, os.path.join(path, "accompaniment.wav"))

        # 清理 Demucs 產生的多餘資料夾
        shutil.rmtree(os.path.join(path, "htdemucs"))
        
        print(f"✅ Demucs 分割完成！檔案位於 {path}")

    except Exception as e:
        print(f"❌ Demucs 執行失敗：{e}")

# 音訊分析
def vocal_analysis(path):
    snd = parselmouth.Sound(f"{path}/vocals.wav")
    y, sr = librosa.load(f"{path}/vocals.wav", sr=44100)

    step_size = 0.05   # 一次移動多少

    # 音高分析 ---------------------------------------------------------------------------

    # 取得音高
    pitch = snd.to_pitch(time_step=step_size, pitch_floor=80, pitch_ceiling=1000)
    times = pitch.xs()
    frequencies = pitch.selected_array['frequency']

    # 將無效頻率（如小於等於0或超過合理範圍）設為 NaN
    frequencies[(frequencies <= 0) | (frequencies > 1000)] = np.nan  # 設定合理範圍

    # 自定義固定時間軸：從 0 開始，每 step_size 秒直到音檔結束
    duration = snd.get_total_duration()
    fixed_times = np.arange(0, duration, step_size)

    # 內插：把原始音高資料對齊到我們自己定義的時間軸
    f_interp = interp1d(times, frequencies, kind='linear', bounds_error=False, fill_value=np.nan)

    # 重新設定 frequencies 和 times
    frequencies = f_interp(fixed_times)
    times = fixed_times

    # 轉換為 MIDI 數值
    midi_pitches = librosa.hz_to_midi(frequencies)

    # 將 NaN 轉換為 NaN，保證 NaN 被保留
    midi_pitches = np.where(np.isnan(midi_pitches), np.nan, midi_pitches)

    # 過濾極端值，避免雜訊影響 (分割4, 99.5，人聲2, 99.5)
    percentile_low = np.percentile(midi_pitches[~np.isnan(midi_pitches)], 2)
    percentile_high = np.percentile(midi_pitches[~np.isnan(midi_pitches)], 99.5)
    extreme_indices = (midi_pitches < percentile_low) | (midi_pitches > percentile_high)
    midi_pitches[extreme_indices] = np.nan

    # 轉換 MIDI 數值為音名
    note_labels = [midi_to_note(midi_pitch) for midi_pitch in midi_pitches]

    # 音量分析 ---------------------------------------------------------------------------

    # 計算 RMS
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512) + times[0]

    # 套用平滑
    rms_smooth = smooth_rms(rms, window_size=10)
    #rms_normalized = normalize_rms_minmax(rms_smooth)

    # 插值至 pitch 對齊時間軸
    rms_interp = np.interp(times, rms_times, rms_smooth)

    # onset 抓取 ---------------------------------------------------------------------------

    # 用 librosa.onset_detect() 偵測每個節拍
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', hop_length=512)
    # 將最後一個時間點加入 onsets 陣列
    onsets = np.append(onsets, times[-1])

    # 抓每個起音點的間隔時間
    onset_intervals = np.diff(onsets)

    # 將 onset 對應時間軸
    onset_indices = np.zeros_like(times, dtype=bool)

    # 定義你想檢查的前後秒數範圍
    range_in_seconds = 0.2
    range_in_frames = int(range_in_seconds / step_size)

    for t in onsets:
        idx = int(t / step_size)

        # 計算前後區間的索引範圍
        start_idx = max(0, idx - range_in_frames)
        end_idx = min(len(times) - 1, idx + range_in_frames)

        # 檢查區間內是否有非 NaN 的 midi pitch
        if np.any(~np.isnan(midi_pitches[start_idx:end_idx + 1])) or t == onsets[-1]:
            onset_indices[idx] = True

    # 轉音分析 ---------------------------------------------------------------------------

    melisma_indices = np.zeros_like(midi_pitches, dtype=bool)

    # 轉音條件參數
    note_change_threshold = 4
    min_unique_notes = 4
    
    melisma_unique = np.zeros_like(midi_pitches, dtype=int)
    melisma_change = np.zeros_like(midi_pitches, dtype=int)

    # 根據起音點逐段處理，紀錄音高變動數和幾個音高，和是否為轉音
    for i in range(len(onsets) - 1):
        start_time = onsets[i]
        end_time = onsets[i + 1]

        # 對應時間轉為索引（四捨五入）
        start_idx = int(start_time / step_size)
        end_idx = round(end_time / step_size)

        if end_idx <= start_idx:  # 忽略無效片段
            continue

        segment = midi_pitches[start_idx:end_idx+1]

        valid_notes = ~np.isnan(segment)
        cleaned_notes = np.round(segment[valid_notes])

        unique_notes_count = len(np.unique(cleaned_notes))
        note_changes = np.sum(cleaned_notes[1:] != cleaned_notes[:-1])

        melisma_unique[start_idx:end_idx+1][valid_notes] = unique_notes_count
        melisma_change[start_idx:end_idx+1][valid_notes] = note_changes

        if note_changes >= note_change_threshold and unique_notes_count >= min_unique_notes:
            melisma_indices[start_idx:end_idx+1][valid_notes] = True  # 整段標註為轉音

    # 節奏分析 ---------------------------------------------------------------------------
    
    # 提取頻率
    f0 = librosa.yin(y, fmin=80, fmax=1000, sr=sr)

    # 計算每個節奏樣本的頻率出現次數與總頻率
    freqs, counts = np.unique(f0, return_counts=True)
    total_frequencies = freqs * counts

    # 計算頻率排名百分比
    rank_percent = total_frequencies / np.sum(total_frequencies)

    # 節拍值與二進位表示法

    # 若間距小於某閾值，判為雙擊(2)，否則單擊(1)
    stroke_values = [2 if gap < 0.3 else 1 for gap in onset_intervals]

    # 轉換為 binary
    binary_encoding = [format(v, 'b') for v in stroke_values]

    # 計算節奏密度
    meter_length = len(stroke_values)  # 或你自訂節拍總長
    total_beats = sum(len(b) for b in binary_encoding)
    rhythm_density = meter_length / total_beats

    # 計算Relative Rank % 與不可或缺向量
    relative_rank = rank_percent / np.max(rank_percent)
    indispensability_vector = 1 - relative_rank

    # 計算Syncopation Degree（與 Binary Encoding 相乘）
    syncopation_vector = indispensability_vector[:len(stroke_values)] * np.array(stroke_values)

    # 計算節奏複雜度
    rhythm_complexity = np.sum(syncopation_vector) / meter_length

    print(f'節奏密度：{1 - rhythm_density:.2f}') # 理論為 0-1 (越大越密)
    print(f'節奏複雜度：{rhythm_complexity:.2f}') # 理論為 0-2 (越大越複雜)

    # 初始化節奏標記（長度與 pitch 時間軸一致）
    hit_indices = np.zeros_like(times, dtype=bool)

    # 抓出雙擊：間距小於 0.3 的下一個 onset 時間點
    double_hit_times = np.where(onset_intervals < 0.3)[0]

    # 根據起音點逐段處理，紀錄節奏較快的點
    for i in double_hit_times:
        start_time = onsets[i]
        end_time = onsets[i + 1]

        # 對應時間轉為索引（四捨五入）
        start_idx = int(start_time / step_size)
        end_idx = round(end_time / step_size)

        if end_idx <= start_idx:  # 忽略無效片段
            continue

        segment = midi_pitches[start_idx:end_idx]
        valid_notes = ~np.isnan(segment)

        hit_indices[start_idx:end_idx][valid_notes] = True

    # 統整為 df ---------------------------------------------------------------------------

    # 匯出為 DataFrame
    df = pd.DataFrame({
        "Time (s)": times,
        "MIDI Pitch": midi_pitches,
        "Note": note_labels,
        "Melisma": melisma_indices,
        "Melisma_change": melisma_change,
        "Melisma_unique": melisma_unique,
        "Hit": hit_indices,
        "RMS": rms_interp,
        "Onset": onset_indices
    })

    df.to_csv(f"{path}/vocals.csv", index=False, encoding="utf-8-sig")

# MIDI 轉換為音名（C4、E5）
def midi_to_note(midi_value):
    # 定義 MIDI 對應的音名
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    if np.isnan(midi_value):
        return "NaN"
    note = note_names[int(round(midi_value)) % 12]
    octave = int(round(midi_value)) // 12 - 1
    return f"{note}{octave}"

# 計算音檔音量
def rms(audio):
    samples = np.array(audio.get_array_of_samples())
    return np.sqrt(np.mean(samples.astype(float)**2))

# 音量正規化
def normalize_rms_minmax(rms):
    return (rms - np.min(rms)) / (np.max(rms) - np.min(rms))

# 音量移動平均平滑處理
def smooth_rms(rms, window_size):
    return np.convolve(rms, np.ones(window_size)/window_size, mode='same')

# 語音辨識進行音訊分段
def segmentAudio(path, lyrics):
    # 簡單判斷是否有中文字元
    def contains_chinese(text):
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    # 決定引導語言
    target_lang = 'zh' if contains_chinese(lyrics) else None

    # 清理引導語言（將換行轉空格）
    clean_prompt = lyrics.replace('\n', ' ').strip()

    # 執行辨識
    result = model.transcribe(
        f"{path}/vocals.wav", 
        fp16=False,
        beam_size=1, 
        language=target_lang,  # 如果為 None，Whisper 會自動偵測
        initial_prompt=clean_prompt, # 提供歌詞作為初始提示
        word_timestamps=True # 啟用詞級時間戳記
    )

    converter = opencc.OpenCC('s2t')  # 簡體轉繁體

    # 取得 Whisper 語音辨識結果的 segments
    segments = result["segments"]
    # 將每個 segment 轉成 (辨識文字, 起始時間, 結束時間) 的 tuple
    asr_lines = [(converter.convert(seg["text"]), seg["start"], seg["end"]) for seg in segments]

    # \n\s*\n 代表：換行 -> 任意空白字元(含空格) -> 換行
    paragraphs_raw = re.split(r'\n\s*\n', lyrics.strip())

    # 依段落分，每段再依行分割，並過濾掉純空白行
    paragraphs = [
        [line.strip() for line in p.split('\n') if line.strip()] 
        for p in paragraphs_raw 
        if p.strip()
    ]

    # 懲罰係數 (lambda_penalty 針對長度差異, skip_penalty 針對跳過距離)
    LENGTH_PENALTY_FACTOR = 0.8

    # --- 步驟 3: 定義優化後的相似度函數 (TF-IDF + 長度懲罰) ---
    all_text = ["".join(lines) for lines in paragraphs] + [text for text, start, end in asr_lines]
    vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, lowercase=False)
    vectorizer.fit(all_text)

    def optimized_similarity_score(para_text, combined_asr_text):
        """計算優化後的基礎相似度分數（餘弦相似度 + 長度懲罰）。"""
        if not combined_asr_text or not para_text:
            return 0.0

        tfidf = vectorizer.transform([para_text, combined_asr_text])
        cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

        len_para = len(para_text)
        len_asr = len(combined_asr_text)
        
        length_diff_ratio = abs(len_para - len_asr) / max(len_para, len_asr, 1)
        penalty = max(0, 1.0 - length_diff_ratio * LENGTH_PENALTY_FACTOR) 

        return cosine_sim * penalty

    # --- 步驟 4: 優化後的對齊函數 (V3: 局部優先策略) ---
    def find_best_match_optimized_v3(paragraph_lines, asr_lines, asr_start_idx=0, max_window=20, min_score=0.30):
        para_text = "".join(paragraph_lines) 
        n = len(asr_lines)
        
        # 💡 修正 1：縮小搜尋範圍。既然是「下一段」，通常不會離上一段結束點太遠
        # 將 search_limit 從 100 縮小到 30，強制演算法專注於「接下來」的聲音
        search_limit = min(n, asr_start_idx + 30) 

        best_final_score = -1.0
        best_start, best_end = None, None
        best_end_index = asr_start_idx + 1 # 預設至少推進一格

        if asr_start_idx >= n:
            return None, None, n

        for start_idx in range(asr_start_idx, search_limit):
            # 💡 修正 2：加重距離懲罰。讓演算法「極度不願意」跳過太多的 ASR 行
            skip_distance = start_idx - asr_start_idx
            proximity_penalty = np.exp(-0.25 * skip_distance) # 將 0.05 提高到 0.25

            for win in range(1, max_window + 1):
                current_end_idx = start_idx + win 
                if current_end_idx > n: break
                
                combined_asr_text = "".join([asr_lines[i][0] for i in range(start_idx, current_end_idx)])
                base_score = optimized_similarity_score(para_text, combined_asr_text)
                final_score = base_score * proximity_penalty
                
                if final_score > best_final_score:
                    best_final_score = final_score
                    best_start = asr_lines[start_idx][1]
                    best_end = asr_lines[current_end_idx - 1][2]
                    best_end_index = current_end_idx

        # 💡 修正 3：防止重複時段。如果沒找到好的匹配，強制從當前索引往後推
        if best_final_score < 0.1: # 門檻提高
            # 強制取用當前 ASR 指標的後續內容，確保時間軸一直往後
            best_start = asr_lines[asr_start_idx][1]
            best_end = asr_lines[min(asr_start_idx + 2, n-1)][2]
            best_end_index = asr_start_idx + 2 

        return best_start, best_end, best_end_index

    # --- 步驟 5: 執行對齊並寫入 CSV ---
    output_file_path = f"{path}/segment.csv"

    with open(output_file_path, "w", newline='', encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["segment_id", "start", "end", "lyric"])

        asr_index = 0
        print("\n--- 開始對齊 (V3: 局部優先策略) ---")
        
        for i, para in enumerate(paragraphs):
            
            # 💡 動態窗口邏輯：估算大約需要多少個 ASR segment
            # 假設一個 ASR segment 平均 4-6 個字，視窗不應超過預估值的 1.5 倍
            para_len = len("".join(para))
            dynamic_window = max(5, min(12, int(para_len / 4 * 1.5))) 
            
            start, end, next_asr_index = find_best_match_optimized_v3(
                para, 
                asr_lines, 
                asr_start_idx=asr_index,
                max_window=dynamic_window, # 💡 使用動態窗口，避免過度合併
                min_score=0.30 
            )
            
            lyrics_joined = "\\n".join(para)
            
            if start is not None and end is not None:
                writer.writerow([i + 1, f"{start:.2f}", f"{end:.2f}", lyrics_joined])
            else:
                print(f"❌ 警告：第 {i+1} 段歌詞未能分配時間。內容：{lyrics_joined[:20]}...")

            # 更新下一段開始的 ASR 索引
            asr_index = next_asr_index  
            
        print(f"\n✅ 歌詞分段對齊完成，結果已寫入：{output_file_path}")

# 建立一個反向轉換函數，將 MIDI 數值轉回音名 (選配，方便閱讀)
def midi_to_note_name(midi_val):
    if midi_val < 0 or np.isnan(midi_val): return ""
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = int(midi_val // 12)
    pitch = pitch_names[int(midi_val % 12)]
    return f"{pitch}{octave}"

# 將 Note 字串轉換為 MIDI 數值
def get_note_midi_value(note_str):
    # (保持你的 get_note_midi_value 和 calculate_rhythm_metrics 函數不變)
    pitch_order = [
        "C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb",
        "G", "G#", "Ab", "A", "A#", "Bb", "B"
    ]

    """將 Note 字串轉換為數值，以便排序。"""
    if pd.isna(note_str):
        return -1
    
    note_str = str(note_str).strip().upper()

    match = re.match(r"^([A-G])(#|B)?(\d+)$", note_str)
    
    if not match:
        return -1
    
    pitch_base = match[1]
    accidental = match[2] if match[2] else ""
    octave = int(match[3])
    
    pitch = pitch_base + accidental
    
    if pitch == "DB": pitch = "C#"
    if pitch == "EB": pitch = "D#"
    if pitch == "GB": pitch = "F#"
    if pitch == "AB": pitch = "G#"
    if pitch == "BB": pitch = "A#"

    try:
        pitch_index = pitch_order.index(pitch)
    except ValueError:
        return -1
        
    return octave * 12 + pitch_index

# 計算節奏密度與節奏複雜度
def calculate_rhythm_metrics(y, sr):
    """
    計算給定音訊數據的節奏密度和節奏複雜度。
    
    參數:
    y (np.ndarray): 音訊時域信號。
    sr (int): 音訊的取樣率。
    
    回傳:
    tuple: 包含 (rhythm_density, rhythm_complexity) 的元組。
    """
    if len(y) == 0:  # 處理空音訊數據
        return 0.0, 0.0
    
    # 提取頻率，用以計算頻率排名百分比
    f0 = librosa.yin(y, fmin=80, fmax=1000, sr=sr)
    
    # 計算每個頻率的出現次數與總頻率
    freqs, counts = np.unique(f0, return_counts=True)
    total_frequencies = freqs * counts
    
    # 計算頻率排名百分比
    if np.sum(total_frequencies) == 0:
        return 0.0, 0.0
    rank_percent = total_frequencies / np.sum(total_frequencies)
    
    # 偵測每個節拍的 onset
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    if len(onsets) < 2:
        return 0, 0 # 如果節拍過少，無法計算，返回0
    
    # 計算 onset 之間的時間間隔
    onset_intervals = np.diff(onsets)
    
    # 若間距小於某閾值（0.3秒），判為雙擊(2)，否則單擊(1)
    stroke_values = [2 if gap < 0.3 else 1 for gap in onset_intervals]
    
    # 計算節奏密度
    meter_length = len(stroke_values)
    total_beats = sum(stroke_values)
    rhythm_density = 1 - (meter_length / total_beats) if total_beats > 0 else 0
    
    # 計算不可或缺向量 (Indispensability Vector)
    # 這裡確保向量長度一致，以免索引錯誤
    relative_rank = rank_percent / np.max(rank_percent)
    indispensability_vector = 1 - relative_rank
    
    # 計算 Syncopation Degree
    min_len = min(len(indispensability_vector), len(stroke_values))
    syncopation_vector = indispensability_vector[:min_len] * np.array(stroke_values[:min_len])
    
    # 計算節奏複雜度
    rhythm_complexity = np.sum(syncopation_vector) / meter_length if meter_length > 0 else 0

    return rhythm_density, rhythm_complexity

# 補齊空白段落
def fill_empty_segments(segments_df, song_duration):
    """
    根據已分配的段落，自動補齊未分配的時間區間，並標記為 '無資料' 段落。
    """
    filled_segments = []
    prev_end = 0.0

    for idx, row in segments_df.iterrows():
        start = float(row['start'])
        end = float(row['end'])
        # 若有空白區間，補上無資料段落
        if start > prev_end:
            filled_segments.append({
                'segment_id': 'noData',
                'start': prev_end,
                'end': start
            })
        # 加入原本段落
        filled_segments.append({
            'segment_id': row['segment_id'],
            'start': start,
            'end': end
        })
        prev_end = end

    # 檢查結尾是否有未分配區間
    if prev_end < song_duration:
        filled_segments.append({
            'segment_id': 'noData',
            'start': prev_end,
            'end': song_duration
        })
    return pd.DataFrame(filled_segments)

# 根據 segment 檔案分析音樂數據並匯出 CSV
def analyze_music_data_by_segments(data_file_path, segment_file_path, audio_file_path, cut_point1, cut_point2, output_file=None):
    """
    根據segment檔案定義的分段，分析音樂數據並匯出為 CSV 檔案。
    這個版本會自動在有資料的段落前後增加無資料段落。
    """
    
    # 讀取主要數據檔案和分段檔案
    try:
        df = pd.read_csv(data_file_path)
        segments_df = pd.read_csv(segment_file_path)
    except FileNotFoundError as e:
        print(f"錯誤：找不到檔案。請檢查檔案路徑。錯誤訊息: {e}")
        return

    # 確保所需的欄位存在
    required_data_cols = ['Time (s)', 'Note', 'Melisma', 'Melisma_change', 'Melisma_unique', 'RMS']
    for col in required_data_cols:
        if col not in df.columns:
            print(f"錯誤：數據檔案中找不到 '{col}' 欄位。")
            return
    
    required_segment_cols = ['start', 'end', 'segment_id']
    for col in required_segment_cols:
        if col not in segments_df.columns:
            print(f"錯誤：分段檔案中找不到 '{col}' 欄位。")
            return

    # 處理 'Time (s)' 欄位
    df['Time (s)'] = pd.to_numeric(df['Time (s)'], errors='coerce')
    df.dropna(subset=['Time (s)'], inplace=True)
    
    # 轉換 Note 為可排序的數值
    df['Note_value'] = df['Note'].apply(get_note_midi_value)
    
    # 轉換切割點 Note 為數值
    cut_value1 = get_note_midi_value(cut_point1)
    cut_value2 = get_note_midi_value(cut_point2)

    if cut_value1 == -1 or cut_value2 == -1:
        print("錯誤：您指定的切割點 Note 格式不正確。請使用例如 'C4', 'G#5' 的格式。")
        return
        
    if cut_value1 > cut_value2:
        cut_point1, cut_point2 = cut_point2, cut_point1
        print(f"警告：切割點順序不正確，已自動調整為：{cut_point1}, {cut_point2}")

    print(f"自定義 Note 分段切割點：")
    print(f"  低音/中音切割點：{cut_point1}")
    print(f"  中音/高音切割點：{cut_point2}")
    print("\n")
    
    results_list = []

    # 處理 Melisma start，並將結果加入到主數據框
    df['Melisma_start'] = (df['Melisma'] == True) & (df['Melisma'].shift(1) == False) & (df['Onset'] == True)
    if not df.empty and df.iloc[0]['Melisma'] == True:
        df.loc[df.index[0], 'Melisma_start'] = True
    
    # 載入音訊並取得歌曲總長度
    y_full, sr = librosa.load(audio_file_path, sr=44100)
    song_duration = librosa.get_duration(y=y_full, sr=sr)

    # 讀取原始分段，補齊空白段落
    segments_df = fill_empty_segments(segments_df, song_duration)
    
    # 根據新的分段列表進行分析
    for idx, segment in segments_df.iterrows():
        segment_id = segment['segment_id']
        start_time = segment['start']
        end_time = segment['end']

        # 篩選出當前分段的數據
        group = df[(df['Time (s)'] >= start_time) & (df['Time (s)'] < end_time)]
        notes_in_segment = group.dropna(subset=['Note'])
        
        # 初始化計數器
        total_notes = len(notes_in_segment)
        total_samples = len(group) #計算該區段的總樣本數 (包含有音高與無音高)
        low_count, mid_count, high_count,nan_count, melisma_count = 0, 0, 0, 0, 0
        low_ratio, mid_ratio, high_ratio, nan_ratio = "0.00", "0.00", "0.00", "0.00"
        rms_average = 0.0
        melisma_pitch = ''
        melisma_change = ''
        melisma_unique = ''
        # --- 新增音高統計變數 ---
        avg_pitch_val = 0.0
        max_pitch_val = 0.0
        min_pitch_val = 0.0
        avg_pitch_name = ""
        max_pitch_name = ""
        min_pitch_name = ""
        
        print(f"--- 分析時間段：段落 {segment_id} ({start_time:.2f}-{end_time:.2f}秒) ---")
        
        # 統計 Note 分佈
        if not notes_in_segment.empty:
            
            low_notes = notes_in_segment[notes_in_segment['Note_value'] < cut_value1]
            mid_notes = notes_in_segment[(notes_in_segment['Note_value'] >= cut_value1) & (notes_in_segment['Note_value'] < cut_value2)]
            high_notes = notes_in_segment[notes_in_segment['Note_value'] >= cut_value2]
            
            low_count = int(len(low_notes))
            mid_count = int(len(mid_notes))
            high_count = int(len(high_notes))
            nan_count = total_samples - total_notes #新增：計算無音高的數量與比例
            
            low_ratio = f"{low_count / total_samples:.2f}" if total_samples > 0 else "0.00"
            mid_ratio = f"{mid_count / total_samples:.2f}" if total_samples > 0 else "0.00"
            high_ratio = f"{high_count / total_samples:.2f}" if total_samples > 0 else "0.00"
            nan_ratio = f"{nan_count / total_samples:.2f}" if total_samples > 0 else "0.00"
            
            print(f"Note 分佈比例 (總數: {total_notes})：")
            print(f" 低音 (<{cut_point1}): {low_count} ({low_ratio})")
            print(f" 中音 ({cut_point1}-{cut_point2}): {mid_count} ({mid_ratio})")
            print(f" 高音 (>{cut_point2}): {high_count} ({high_ratio})")
            print(f" 無音高 (NaN): {nan_count} ({nan_ratio})")

            # --- 過濾出現次數大於 1 的音高邏輯 ---
            # 1. 統計該片段內每個 MIDI 值的出現次數
            pitch_value_counts = notes_in_segment['Note_value'].value_counts()
            
            # 2. 篩選出出現次數 > 1 的音高
            frequent_pitches = pitch_value_counts[pitch_value_counts > 1].index
            
            # 3. 計算平均值 (平均值通常建議保留所有數據，或視需求改用頻繁音)
            avg_pitch_val = notes_in_segment['Note_value'].mean()

            # 4. 判斷是否有符合條件（出現 > 1 次）的音高
            if len(frequent_pitches) > 0:
                # 抓取出現次數大於 1 的最高音與最低音
                max_pitch_val = frequent_pitches.max()
                min_pitch_val = frequent_pitches.min()
            else:
                # 💡 Fallback: 如果該段落太短，所有音都只出現一次，則保留原有的極值
                # 或是您可以選擇設為 0/NaN
                max_pitch_val = notes_in_segment['Note_value'].max()
                min_pitch_val = notes_in_segment['Note_value'].min()

            # 轉回音名標籤
            avg_pitch_name = midi_to_note_name(round(avg_pitch_val))
            max_pitch_name = midi_to_note_name(max_pitch_val)
            min_pitch_name = midi_to_note_name(min_pitch_val)

            print(f"音高統計：平均={avg_pitch_name}({avg_pitch_val:.1f}), 最高={max_pitch_name}, 最低={min_pitch_name}")
            
            # 處理轉音
            melisma_count = int(group['Melisma_start'].sum())
            melisma_pitch = ','.join(group.loc[group['Melisma_start'], 'Note'])
            melisma_change = ','.join(map(str, group.loc[group['Melisma_start'], 'Melisma_change']))
            melisma_unique = ','.join(map(str, group.loc[group['Melisma_start'], 'Melisma_unique']))
            
            rms_average = group['RMS'].mean()
        else:
            print("此時間段沒有 Note 數據，將以 0 或 NaN 記錄。")
            
        # 提取該分段的音訊數據
        y_segment = y_full[int(start_time * sr):int(end_time * sr)]

        # 呼叫函式計算節奏指標
        rhythm_density, rhythm_complexity = calculate_rhythm_metrics(y_segment, sr)

        print(f'節奏密度：{rhythm_density:.2f}')
        print(f'節奏複雜度：{rhythm_complexity:.2f}\n')
        
        # 將結果加入列表
        results_list.append({
            'segment_id': segment_id,
            'start': start_time,
            'end': end_time,
            'total_note': total_notes if total_notes > 0 else 0,
            'total_samples': total_samples if total_samples > 0 else 0,
            'low_note_count': low_count,
            'mid_note_count': mid_count,
            'high_note_count': high_count,
            'nan_note_count': nan_count,
            'low_note_ratio': low_ratio,
            'mid_note_ratio': mid_ratio,
            'high_note_ratio': high_ratio,
            'nan_note_ratio': nan_ratio,
            'pitch_avg_note': avg_pitch_name,
            'pitch_max_note': max_pitch_name,
            'pitch_min_note': min_pitch_name,
            'melisma_count': melisma_count,
            'melisma_start_note': melisma_pitch,
            'melisma_change': melisma_change,
            'melisma_unique': melisma_unique,
            'rms_average': rms_average,
            'rhythm_density': rhythm_density,
            'rhythm_complexity': rhythm_complexity
        })

    # 匯出到 CSV
    if output_file:
        df_results = pd.DataFrame(results_list)
        try:
            df_results.to_csv(output_file, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 以確保 Excel 正常顯示中文
            print(f"分析結果已成功匯出到檔案: {output_file}")
        except IOError as e:
            print(f"錯誤：無法寫入檔案 {output_file}。錯誤訊息：{e}")

# 判斷是否含有中文
def is_chinese(text):
    """判斷字串中是否含有中文字元"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# 取得韻部
def get_rhyme_syllable(text, is_zh):
    """根據語言取得韻部"""
    if not text: return None
    
    if is_zh:
        # 中文邏輯：取最後一個字的拼音韻母
        py = pinyin(text[-1], style=Style.FINALS, strict=False)
        return py[0][0] if py and py[0] else None
    else:
        # 英文邏輯：取最後一個單字的音標韻部
        if pronouncing:
            phones_list = pronouncing.phones_for_word(text)
            if phones_list:
                return pronouncing.rhyming_part(phones_list[0])
        return text[-2:] # 備用方案

# 歌詞預處理並匯出 CSV
def preprocess_to_rows(lyrics, path):
    """
    自動判定語言並處理歌詞
    """
    # 1. 判定整首歌的語言（以第一行非空文字為準）
    first_line = next((line for line in lyrics.split('\n') if line.strip()), "")
    is_zh = is_chinese(first_line)
    lang_label = "ZH" if is_zh else "EN"
    print(f"偵測到語言：{lang_label}")

    # 2. 段落處理
    # \n\s*\n 代表：換行 -> 任意空白字元(含空格) -> 換行
    paragraphs_raw = re.split(r'\n\s*\n', lyrics.strip())

    # 依段落分，每段再依行分割，並過濾掉純空白行
    paragraphs = [
        [line.strip() for line in p.split('\n') if line.strip()] 
        for p in paragraphs_raw 
        if p.strip()
    ]
    
    seen_lines = set()
    rows = []

    for p_idx, para in enumerate(paragraphs, 1):
        rhyme_map = defaultdict(list)
        para_lines = [line.strip() for line in para if line.strip()]
        
        # --- 第一階段：掃描段落建立押韻圖 ---
        for line in para_lines:
            if is_zh:
                clean = re.sub(r'[^\u4e00-\u9fff]', '', line)
                last_unit = clean if clean else None
            else:
                clean_words = re.findall(r'\b\w+\b', line.lower())
                last_unit = clean_words[-1] if clean_words else None

            if last_unit:
                rhyme = get_rhyme_syllable(last_unit, is_zh)
                if rhyme:
                    rhyme_map[rhyme].append(line)
        
        rhyming_lines = {line for lines in rhyme_map.values() if len(lines) > 1 for line in lines}

        # --- 第二階段：生成資料列 ---
        for line in para_lines:
            # 計算字數：中文算字數，英文算詞數
            if is_zh:
                word_count = len(re.sub(r'\s+', '', line))
            else:
                word_count = len(line.split())
            
            is_rhyme = "True" if line in rhyming_lines else "False"
            
            # 重複判定（英文不分大小寫）
            compare_key = line.lower().strip() if not is_zh else line.strip()
            is_repeat = "True" if compare_key in seen_lines else "False"
            seen_lines.add(compare_key)

            rows.append({
                "segment_id": p_idx,
                "content": line,
                "word_count": word_count,
                "rhyme": is_rhyme,
                "repeat": is_repeat
            })

    # 3. 匯出
    df = pd.DataFrame(rows)
    df.to_csv(f'{path}/lyrics.csv', index=False, encoding="utf-8-sig")
    print(f"分析完成，已根據 {lang_label} 邏輯匯出。")

# 音色分析
def analyze_timbre(path):
    y, sr = librosa.load(os.path.join(path, 'vocals.wav'), sr=22050)
    # 取得 F0（使用 librosa 的 pyin）
    f0 = librosa.yin(y, fmin=80, fmax=1000, sr=sr)
    # 計算 STFT 頻譜
    S = np.abs(librosa.stft(y, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr)
    mean_spectrum = np.mean(S, axis=1)
    
    # Bright / Dark (使用所有頻率的距離)

    octave_distances = []

    for i in range(S.shape[1]):
        f0_now = f0[i] #np.nanmedian(f0) 
        if not np.isnan(f0_now):
            spectrum_slice = S[:, i]

            # 設定能量門檻（避免低能量干擾）
            threshold = np.max(spectrum_slice) * 0.03

            # 找出所有高於基頻且能量大於門檻的頻率
            valid_freqs = freqs[(spectrum_slice > threshold) & (freqs > f0_now)]

            if len(valid_freqs) > 0:
                # 抓出最高的那一個作為 "最高泛音"
                highest_partial = np.max(valid_freqs)

                # 計算與 f0 的 octave distance
                distance = np.log2(highest_partial / f0_now)
                octave_distances.append(distance)

    # 平均亮度
    mean_octave_distance = np.mean(octave_distances)
    overall_brightness = "Bright" if mean_octave_distance > 4 else "Dark"

    print(f"\n平均 Octave 距離：{mean_octave_distance:.2f}")
    print(f"整段音訊音色判定為：{overall_brightness}")

    # Pure / noisy

    # 設定相對門檻
    relative_threshold_db = 50 # 比主頻低 50dB 視為 halo
    bandwidths = []

    for t in range(S.shape[1]):
        spectrum = S[:, t]
        peak_bin = np.argmax(spectrum)
        peak_amp = spectrum[peak_bin]

        # 主頻 ± window 範圍內的高能量區
        window_size = 10
        start_bin = max(0, peak_bin - window_size)
        end_bin = min(len(freqs), peak_bin + window_size)

        # 抓 halo（主頻附近落在相對門檻以上的範圍）
        high_energy_bins = np.where(spectrum[start_bin:end_bin] > (peak_amp - relative_threshold_db))[0]

        if len(high_energy_bins) > 1:
            freq_band = freqs[start_bin + high_energy_bins[-1]] - freqs[start_bin + high_energy_bins[0]]
            bandwidths.append(freq_band)

    # 平均頻帶寬度
    mean_bandwidth = np.mean(bandwidths)

    print(f"\n平均頻帶寬度（Hz）：{mean_bandwidth:.2f}")

    if mean_bandwidth < 200:
        purity = "Pure"
    else:
        purity = "Noisy"

    print(f"整段音訊音色判定為：{purity}")

    # Harmonic / Inharmonic

    # === 3. 使用 librosa.yin 估算基頻，取中位數作為全段代表 ===
    f0_median = np.nanmedian(f0)

    all_peak_freqs = []
    # === 4. 找出頻譜峰值 ===
    for t in range(S.shape[1]):
        spectrum = S[:, t]
        peaks, _ = find_peaks(spectrum, prominence=5, distance=3)
        peak_freqs = freqs[peaks]
        all_peak_freqs.extend(peak_freqs)

    # 限制分析的頻率範圍，排除極低與極高雜訊（可調整）
    all_peak_freqs = np.array(all_peak_freqs)
    all_peak_freqs = all_peak_freqs[(all_peak_freqs > 80) & (all_peak_freqs < 1000)]

    # === 5. 判斷是否為基頻整數倍（harmonic）===
    tolerance = 0.3
    harmonic_count = 0
    inharmonic_count = 0

    for freq in all_peak_freqs:
        ratio = freq / f0_median
        rounded = round(ratio)
        error = abs(ratio - rounded)
        if error < tolerance:
            harmonic_count += 1
        else:
            inharmonic_count += 1

    # === 6. 統計與分類 ===
    total_tested = harmonic_count + inharmonic_count
    harmonic_ratio = harmonic_count / total_tested if total_tested > 0 else 0

    if harmonic_ratio > 0.85:
        harmony = "非常和諧（Highly Harmonic）"
    elif harmonic_ratio > 0.6:
        harmony = "和諧（Harmonic）"
    elif harmonic_ratio > 0.3:
        harmony = "不太和諧（Mildly Inharmonic）"
    else:
        harmony = "不和諧（Inharmonic）"

    print(f"\n總共分析 {total_tested} 個峰值，其中 {harmonic_count} 個為和諧泛音")
    print(f"判斷結果：{harmonic_ratio:.2f}, {harmony}")

    # Rich / Sparse

    f0_mean = f0[~np.isnan(f0)].mean()

    harmonics = [f0_mean * i for i in range(1, 11)]
    harmonic_bins = [np.argmin(np.abs(freqs - h)) for h in harmonics]
    mean_spectrum = np.mean(S, axis=1)

    harmonic_amps = [mean_spectrum[bin] for bin in harmonic_bins]
    threshold = np.max(harmonic_amps) * 0.2  # 高於最大值20%的泛音視為顯著

    richness = sum(amp > threshold for amp in harmonic_amps)
    print(f"\n偵測到顯著泛音數量: {richness}")
    print(f"屬於 {'Rich' if richness >= 5 else 'Sparse'}")

    result = [{
        "Bright/Dark": mean_octave_distance,
        "Pure/Noisy": mean_bandwidth,
        "Harmonic/Inharmonic": harmonic_ratio,
        "Rich/Sparse": richness
    }]

    # 3. 匯出
    df = pd.DataFrame(result)
    df.to_csv(f'{path}/timbre.csv', index=False, encoding="utf-8-sig")
    print(f"\n分析完成。")

# 全部檔案重新分段
def resetSegment(p_low, p_high):
    """
    處理全部資料夾的分段。
    """

    # 1. 定義目標資料夾路徑
    FINAL_PATH = os.path.join(PARENT_DIR, 'original_audio')

    # 2. 獲取根目錄下所有檔案/資料夾
    # 遍歷 original_audio 底下的每一個子項目
    for folder_name in os.listdir(FINAL_PATH):
        # 建立完整的子資料夾路徑
        folder_path = os.path.join(FINAL_PATH, folder_name)

        # 3. 確保只處理「資料夾」且排除隱藏檔
        if os.path.isdir(folder_path):
            print(f"--- 正在處理資料夾: {folder_name} ---")

            # 4. 動態定義該子資料夾內的所有檔案路徑
            data_file_path = os.path.join(folder_path, 'vocals.csv')
            segment_file_path = os.path.join(folder_path, 'segment.csv')
            audio_file_path = os.path.join(folder_path, 'vocals.wav')
            output_file = os.path.join(folder_path, 'overview.csv')

            # 5. 檢查必要的輸入檔案是否存在，避免程式崩潰
            if not all(os.path.exists(f) for f in [data_file_path, segment_file_path, audio_file_path]):
                print(f"⚠️ 跳過 {folder_name}：缺少必要的 vocals.csv, segment.csv 或 vocals.wav")
                continue

            # 6. 執行你的分析函式
            try:
                # 這裡假設 p_low, p_high 是你已經定義好的全域變數或參數
                analyze_music_data_by_segments(
                    data_file_path, 
                    segment_file_path, 
                    audio_file_path, 
                    p_low, 
                    p_high, 
                    output_file
                )
                print(f"✅ {folder_name} 處理完成，已產生 overview.csv")
            except Exception as e:
                print(f"❌ 處理 {folder_name} 時發生錯誤: {e}")
    return

# 抓選定的音檔資料
@app.route('/api/selectedSongData', methods=['POST'])
def selectedSongData():
    """
    處理並返回指定資料夾中所有 CSV 檔案的數據。
    """
    
    # 從請求中獲取要處理的資料夾名稱列表
    data = request.get_json()
    target_folder_names = data.get("data", [])
    
    # 定義目標資料夾路徑
    TARGET_DATA_DIR = os.path.join(PARENT_DIR, 'original_audio')
    TARGET_DATA_DIR2 = os.path.join(PARENT_DIR, 'upload_audio')
    
    # 指定要尋找的檔案名稱
    needCsv = ['vocals.csv', 'overview.csv', 'segment.csv', 'lyrics.csv', 'timbre.csv']
    # 執行數據處理
    all_combined_data = process_specific_folders_data(TARGET_DATA_DIR, target_folder_names, needCsv)
    
    needCsv2 = ['overview_mismatch.csv', 'timbre.csv']
    # 也處理 upload_audio 底下的資料夾（如果有）
    all_combined_data_upload = process_specific_folders_data(TARGET_DATA_DIR2, target_folder_names, needCsv2)
    
    # 轉換數據為適合 JSON 傳輸的格式
    response_data = {}
    for folder_name, df in all_combined_data.items():
        # 將 DataFrame 轉換為 JSON 格式 (records 模式最適合前端使用)
        response_data[folder_name] = df

    for folder_name, df in all_combined_data_upload.items():
        response_data[f'{folder_name}_upload'] = df

    return jsonify(response_data) # 返回 JSON 響應

def process_specific_folders_data(base_directory, target_folder_names, file_names):
    """
    只在特定的資料夾名稱中，尋找並讀取指定的 CSV 檔案。
    
    Args:
        base_directory (str): 基礎目錄路徑。
        target_folder_names (list): 想要處理的資料夾名稱清單 (例如 ['folder_A', 'folder_B'])。
        file_name (str): 要尋找的檔案名稱。

    Returns:
        dict: {資料夾名稱: DataFrame}
    """
    
    all_data = defaultdict(dict)

    if not os.path.exists(base_directory):
        print(f"錯誤：找不到基礎目錄 '{base_directory}'")
        return all_data

    for folder_name in target_folder_names:
        # 組合出該特定資料夾的完整路徑
        folder_path = os.path.join(base_directory, folder_name)
        
        # 檢查該資料夾是否存在
        if os.path.isdir(folder_path):
            # 遍歷指定的檔案名稱列表
            for file_name in file_names:
                target_file_path = os.path.join(folder_path, file_name)
                print(target_file_path)
                
                # 檢查檔案是否存在
                if os.path.isfile(target_file_path):
                    try:
                        df = pd.read_csv(target_file_path, encoding='utf-8-sig')
                        df = df.where(pd.notnull(df), None)
                        all_data[folder_name][file_name] = df.replace({np.nan: None}).to_dict(orient='records')
                        print(f"✅ 成功讀取：[{folder_name}] 內的 {file_name}")
                    except Exception as e:
                        print(f"❌ 讀取失敗：[{folder_name}]，原因：{e}")
                else:
                    print(f"⚠️ 找不到檔案：資料夾 [{folder_name}] 中沒有 '{file_name}'")
        else:
            print(f"🚫 找不到資料夾：基礎目錄下沒有名為 [{folder_name}] 的資料夾")
    
    return all_data

# 抓全部的歌手與歌名資料
@app.route('/api/singerSongList', methods=['POST'])
def singerSongList():
    TARGET_DATA_DIR = os.path.join(PARENT_DIR, 'original_audio')
    # 取得該目錄下所有資料夾名稱
    folders = [f for f in os.listdir(TARGET_DATA_DIR) if os.path.isdir(os.path.join(TARGET_DATA_DIR, f))]
    
    filter_song = []
    filter_singer = []
    filter_list = []
    for f in folders:
        if '_' in f:
            if not f in filter_list:
                filter_list.append(f)

            name, artist = f.split('_', 1)
            if not name in filter_song:
                filter_song.append(name)
            if not artist in filter_singer:
                filter_singer.append(artist)
            
    return jsonify({
        "list": filter_list,
        "song": filter_song,
        "singer": filter_singer
    })

# 抓全部的音檔資料
@app.route('/api/overviewData', methods=['POST'])
def overviewData():
    """
    處理並返回巢狀資料夾中所有 CSV 檔案的數據。
    """

    # 定義目標資料夾路徑
    TARGET_DATA_DIR = os.path.join(PARENT_DIR, 'original_audio')
    
    # 執行數據處理
    all_combined_data = process_nested_csv_data(TARGET_DATA_DIR, 'overview.csv')
    
    # 💡 轉換數據：因為 process_nested_csv_data 已經把順序排好了
    # 我們只需要確保 DataFrame 被轉成字典格式 (records)，並直接回傳列表
    final_response = []
    
    for item in all_combined_data:
        folder_name = item["folderName"]
        df = item["dataSet"]
        
        # 確保將 DataFrame 轉換為 records 格式
        final_response.append({
            "folderName": folder_name,
            "dataSet": df.to_dict(orient='records')
        })
        
    return jsonify(final_response)

# 找出所有路徑下的資料夾，並找到特定檔名的檔案
def process_nested_csv_data(base_directory, file_name):
    """
    遍歷指定路徑下的所有子資料夾，讀取其中的 CSV 檔案，
    並將資料整理成一個字典，鍵為子資料夾名稱，值為該資料夾內所有 CSV 檔案合併後的 DataFrame。

    Args:
        base_directory (str): 基礎目錄路徑。

    Returns:
        dict: 包含所有資料的字典，例如 {'folder1': DataFrame1, 'folder2': DataFrame2}
    """
    
    # 儲存最終結果的陣列
    all_data = []

    # 獲取基礎目錄下所有項目（資料夾和檔案）的列表
    try:
        # 1. 獲取所有子資料夾名稱
        all_items = [
            f for f in os.listdir(base_directory) 
            if os.path.isdir(os.path.join(base_directory, f))
        ]

        # 2. 💡 核心：依據資料夾的建立時間排序 (由舊到新)
        # os.path.getctime 在 Mac 上代表建立時間；若想用修改時間可改用 getmtime
        all_items.sort(key=lambda x: os.path.getmtime(os.path.join(base_directory, x)), reverse=True)

    except FileNotFoundError:
        print(f"錯誤：找不到基礎目錄 '{base_directory}'")
        return all_data

    # 遍歷資料夾
    for item_name in all_items:
        full_path = os.path.join(base_directory, item_name)

        if os.path.isdir(full_path):
            folder_name = item_name
            target_csv_path = os.path.join(full_path, file_name)

            if os.path.isfile(target_csv_path):
                try:
                    # 使用 utf-8-sig 處理可能的中文編碼問題
                    df = pd.read_csv(target_csv_path, encoding='utf-8-sig')
                    # 💡 在這裡處理 NaN，轉換為 None (對應前端的 null)
                    df = df.where(pd.notnull(df), None)

                    # 💡 每個物件包含資料夾名稱與該資料夾的數據
                    all_data.append({
                        "folderName": folder_name,
                        "dataSet": df
                    })
                    
                    print(f"資料夾 '{folder_name}' 找到檔案 '{file_name}'，共 {len(df)} 筆記錄。")
                except pd.errors.EmptyDataError:
                    print(f"警告: CSV 檔案 '{target_csv_path}' 為空，跳過。")
                except Exception as e:
                    print(f"讀取 CSV 檔案 '{target_csv_path}' 失敗：{e}")
            else:
                print(f"資料夾 '{folder_name}' 內沒有找到檔案 '{file_name}'。")
    
    return all_data

if __name__ == '__main__':
    app.run(port=5000)
