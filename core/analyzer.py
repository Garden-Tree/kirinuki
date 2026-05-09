import os
import logging
import pandas as pd
import numpy as np
import subprocess
import tempfile
from scipy.signal import find_peaks
from core.utils import format_time, normalize_series
from core.ai_scorer import score_with_ai
from core.constants import DEFAULT_WINDOW_SEC, DEFAULT_MOVING_AVERAGE_WINDOW, SIGNAL_WEIGHTS, DEFAULT_KEYWORD_WEIGHTS, DEFAULT_BUFFER_BEFORE, DEFAULT_BUFFER_AFTER

def analyze_audio(video_path, window_sec=DEFAULT_WINDOW_SEC):
    """
    動画から音声のRMS（音量）を短いサブウィンドウ毎に計算し、
    指定ウィンドウ内に「繰り返して大きいリアクション（音量ピーク）」が
    何度あったか（ピークの高さの合計）をスコアとして返します。
    """
    logging.info("音声の音量変化解析（ピーク回数）を開始します...")

    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-ac', '1', '-ar', '8000',
            '-f', 's16le', '-y', tmp_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(tmp_path, 'rb') as f:
            raw_data = f.read()

        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        sample_rate = 8000
        
        # 1秒ごとにRMSを計算
        sub_window_sec = 1.0
        samples_per_sub = int(sample_rate * sub_window_sec)
        total_sub_windows = len(samples) // samples_per_sub
        
        rms_values = []
        for i in range(total_sub_windows):
            chunk = samples[i * samples_per_sub : (i + 1) * samples_per_sub]
            rms = np.sqrt(np.mean(chunk ** 2))
            rms_values.append(float(rms))
            
        rms_array = np.array(rms_values)
        if len(rms_array) == 0:
            return pd.DataFrame(columns=['window', 'audio_score'])

        # ピークの検出
        mean_rms = np.mean(rms_array)
        std_rms = np.std(rms_array)
        threshold = mean_rms + std_rms * 0.5  # 平均より少し上のピーク
        
        peaks, properties = find_peaks(rms_array, height=threshold, distance=2)
        
        df_peaks = pd.DataFrame({
            "time": peaks * sub_window_sec,
            "height": properties["peak_heights"]
        })
        
        if not df_peaks.empty:
            df_peaks["window"] = (df_peaks["time"] // window_sec) * window_sec
            # ウィンドウ内のピークの高さの合計を「繰り返しの大きさ」スコアとする
            df_audio = df_peaks.groupby("window")["height"].sum().reset_index()
            df_audio.rename(columns={"height": "audio_score"}, inplace=True)
            logging.info(f"検出された音量ピーク総数: {len(df_peaks)}回")
        else:
            df_audio = pd.DataFrame(columns=['window', 'audio_score'])

        return df_audio[['window', 'audio_score']]

    except Exception as e:
        logging.error(f"音声解析中にエラーが発生しました: {e}")
        return pd.DataFrame(columns=['window', 'audio_score'])
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def score_superchats(chat_data, window_sec=DEFAULT_WINDOW_SEC):
    """
    チャットデータからスパチャ・メンバーシップイベントを抽出し、
    ウィンドウごとのスコアを返します。
    """
    sc_events = [m for m in chat_data if m.get("type") in ("superchat", "membership")]
    if not sc_events:
        logging.info("スパチャ・メンバーシップイベント: 0件")
        return pd.DataFrame(columns=["window", "sc_score"])

    df_sc = pd.DataFrame(sc_events)
    df_sc["window"] = (df_sc["time_sec"] // window_sec) * window_sec
    # 各イベントに重み（スパチャ=3, メンバーシップ=2）
    df_sc["weight"] = df_sc["type"].map({"superchat": 3.0, "membership": 2.0}).fillna(1.0)
    df_sc_scores = df_sc.groupby("window")["weight"].sum().reset_index()
    df_sc_scores.rename(columns={"weight": "sc_score"}, inplace=True)

    logging.info(f"スパチャ・メンバーシップイベント: {len(sc_events)}件")
    return df_sc_scores

def score_chat_acceleration(chat_data, window_sec=DEFAULT_WINDOW_SEC):
    """
    チャットの密度（件数/ウィンドウ）の変化量（加速度）を計算します。
    「普段1件/分→急に5件/分」のような急増を検出します。
    """
    text_messages = [m for m in chat_data if m.get("type", "text") == "text"]
    if not text_messages:
        return pd.DataFrame(columns=["window", "chat_accel"])

    df_chat = pd.DataFrame(text_messages)
    df_chat["window"] = (df_chat["time_sec"] // window_sec) * window_sec
    df_density = df_chat.groupby("window").size().reset_index(name="count")

    if len(df_density) > 1:
        # 2区間前との差分で加速度を計算（1区間だと敏感すぎる）
        df_density["chat_accel"] = df_density["count"].diff(periods=2).fillna(0).clip(lower=0)
    else:
        df_density["chat_accel"] = 0.0

    return df_density[["window", "chat_accel"]]

def analyze_and_score(chat_data, csv_path=None, engagements_csv_path=None, video_path=None,
                      window_sec=DEFAULT_WINDOW_SEC,
                      keyword_weights=DEFAULT_KEYWORD_WEIGHTS,
                      ma_window=DEFAULT_MOVING_AVERAGE_WINDOW,
                      use_ai=False, api_key=None,
                      ai_cache_path="ai_score_cache.json",
                      use_audio=True,
                      signal_weights=SIGNAL_WEIGHTS):
    """
    全シグナル（チャット・CSV・音声・スパチャ・チャット加速度）を
    パーセンタイル正規化して重み付け合算します。
    """
    logging.info("スコアリングを開始します...")

    max_time = 0
    if chat_data:
        max_time = max(m["time_sec"] for m in chat_data if m["time_sec"] >= 0)

    # --- シグナル1: チャットスコア ---
    if use_ai and api_key:
        df_scores = score_with_ai(chat_data, api_key, window_sec, ai_cache_path=ai_cache_path)
    else:
        scored_messages = []
        for msg in chat_data:
            t = msg["time_sec"]
            if t < 0:
                continue
            text = msg.get("message", "")
            score = 1.0
            for kw, w in keyword_weights.items():
                if kw in text:
                    score += w
            scored_messages.append({"time_sec": t, "chat_score": score})

        if not scored_messages and not csv_path:
            logging.warning("スコアリング用のデータが存在しません。空のDataFrameを返します。")
            return pd.DataFrame()

        df_chat = pd.DataFrame(scored_messages)
        if not df_chat.empty:
            df_chat["window"] = (df_chat["time_sec"] // window_sec) * window_sec
            df_scores = df_chat.groupby("window")["chat_score"].sum().reset_index()
        else:
            df_scores = pd.DataFrame(columns=["window", "chat_score"])

    # 全時間を網羅するベースフレーム
    if max_time == 0 and not df_scores.empty:
        max_time = df_scores['window'].max()
    all_windows = pd.DataFrame({"window": np.arange(0, max_time + window_sec, window_sec)})
    df_scores = pd.merge(all_windows, df_scores, on='window', how='left').fillna(0)

    # --- シグナル2: CSVのエンゲージメント ---
    df_scores['csv_score'] = 0
    df_scores['retention_score'] = 0
    
    # liveViewership.csv の読み込み
    if csv_path and os.path.exists(csv_path):
        logging.info(f"Viewership CSVデータを統合します: {csv_path}")
        try:
            df_csv = pd.read_csv(csv_path)
            if 'ライブ配信の位置（秒）' in df_csv.columns:
                df_csv.rename(columns={'ライブ配信の位置（秒）': 'time_sec'}, inplace=True)
            elif 'time_sec' not in df_csv.columns:
                df_csv.rename(columns={df_csv.columns[0]: 'time_sec'}, inplace=True)

            df_csv['window'] = (df_csv['time_sec'] // window_sec) * window_sec
            agg_dict = {}
            if 'ライブ エンゲージメントの数' in df_csv.columns:
                agg_dict['ライブ エンゲージメントの数'] = 'sum'
            if 'リアクション' in df_csv.columns:
                agg_dict['リアクション'] = 'sum'
            
            retention_col = None
            for col in ['視聴者の維持率', '相対的な視聴者の維持率', '維持率', '同時接続数', 'ライブ同時視聴者数', '平均同時視聴者数']:
                if col in df_csv.columns:
                    retention_col = col
                    agg_dict[col] = 'mean'
                    break

            if agg_dict:
                df_csv_windowed = df_csv.groupby('window').agg(agg_dict).reset_index()
                
                if 'ライブ エンゲージメントの数' in df_csv_windowed.columns or 'リアクション' in df_csv_windowed.columns:
                    csv_sum = 0
                    if 'ライブ エンゲージメントの数' in df_csv_windowed.columns:
                        csv_sum += df_csv_windowed['ライブ エンゲージメントの数']
                    if 'リアクション' in df_csv_windowed.columns:
                        csv_sum += df_csv_windowed['リアクション']
                    df_csv_windowed['csv_score'] = csv_sum
                
                if retention_col:
                    df_csv_windowed['retention_score'] = df_csv_windowed[retention_col]

                df_scores.set_index('window', inplace=True)
                df_csv_windowed.set_index('window', inplace=True)
                if 'csv_score' in df_csv_windowed.columns:
                    df_scores['csv_score'] = df_csv_windowed['csv_score']
                if 'retention_score' in df_csv_windowed.columns:
                    df_scores['retention_score'] = df_csv_windowed['retention_score']
                df_scores.reset_index(inplace=True)
                df_scores.fillna(0, inplace=True)
        except Exception as e:
            logging.error(f"Viewership CSVの読み込み中にエラーが発生しました: {e}")

    # liveEngagements.csv の読み込み
    if engagements_csv_path and os.path.exists(engagements_csv_path):
        logging.info(f"Engagements CSVデータを統合します: {engagements_csv_path}")
        try:
            df_eng = pd.read_csv(engagements_csv_path)
            if 'ライブ配信の位置（秒）' in df_eng.columns and 'ライブ エンゲージメントの数' in df_eng.columns:
                df_eng['window'] = (df_eng['ライブ配信の位置（秒）'] // window_sec) * window_sec
                df_eng_windowed = df_eng.groupby('window')['ライブ エンゲージメントの数'].sum().reset_index()
                df_eng_windowed.rename(columns={'ライブ エンゲージメントの数': 'eng_score_add'}, inplace=True)
                
                df_scores = pd.merge(df_scores, df_eng_windowed, on='window', how='left').fillna(0)
                df_scores['csv_score'] += df_scores['eng_score_add']
                df_scores.drop(columns=['eng_score_add'], inplace=True)
        except Exception as e:
            logging.error(f"Engagements CSVの読み込み中にエラーが発生しました: {e}")

    # --- シグナル3: 音声の音量変化（ピーク回数） ---
    if use_audio and video_path and os.path.exists(video_path):
        df_audio = analyze_audio(video_path, window_sec)
        if not df_audio.empty:
            df_scores = pd.merge(df_scores, df_audio, on='window', how='left').fillna(0)
        else:
            df_scores['audio_score'] = 0
    else:
        df_scores['audio_score'] = 0

    # --- シグナル4: スパチャ・メンバーシップ ---
    df_sc = score_superchats(chat_data, window_sec)
    if not df_sc.empty:
        df_scores = pd.merge(df_scores, df_sc, on='window', how='left').fillna(0)
    else:
        df_scores['sc_score'] = 0

    # --- シグナル5: チャット密度の加速度 ---
    df_accel = score_chat_acceleration(chat_data, window_sec)
    if not df_accel.empty:
        df_scores = pd.merge(df_scores, df_accel, on='window', how='left').fillna(0)
    else:
        df_scores['chat_accel'] = 0

    # ============================================================
    # 正規化して重み付け合算
    # ============================================================
    df_scores.sort_values("window", inplace=True)
    df_scores.reset_index(drop=True, inplace=True)

    signal_map = {
        "chat":       "chat_score",
        "csv":        "csv_score",
        "retention":  "retention_score",
        "audio":      "audio_score",
        "superchat":  "sc_score",
        "chat_accel": "chat_accel",
    }

    logging.info("--- シグナル別統計 ---")
    df_scores['total_score'] = 0.0
    for signal_name, col_name in signal_map.items():
        if col_name in df_scores.columns:
            weight = signal_weights.get(signal_name, 0)
            normalized = normalize_series(df_scores[col_name])
            df_scores[f'norm_{col_name}'] = normalized
            df_scores['total_score'] += normalized * weight
            raw_max = df_scores[col_name].max()
            logging.info(f"  {signal_name:12s}: weight={weight:.2f}, max_raw={raw_max:.2f}")

    # 移動平均を算出してスパイクを滑らかにする
    df_scores['smoothed_score'] = df_scores['total_score'].rolling(window=ma_window, min_periods=1).mean()

    return df_scores

def determine_clip_intervals(df_scores, top_n=5,
                               buffer_before=DEFAULT_BUFFER_BEFORE,
                               buffer_after=DEFAULT_BUFFER_AFTER):
    """
    スコアからピークを特定し、区間をマージして出力リストを作成します。
    """
    logging.info("ピーク（スパイク）の検出を行っています...")
    if df_scores.empty:
        return []

    scores = df_scores['smoothed_score'].values

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    height_threshold = max(mean_score + std_score * 1.0, 0.1)

    peaks_indices, properties = find_peaks(scores, height=height_threshold, distance=3)

    peaks_info = []
    for idx in peaks_indices:
        window_time = df_scores.iloc[idx]['window']
        peak_score = df_scores.iloc[idx]['smoothed_score']
        peaks_info.append({"peak_time": window_time, "score": peak_score})

    logging.info(f"{len(peaks_info)} 個のピークが検出されました。(閾値: {height_threshold:.4f})")

    if not peaks_info:
        return []

    intervals = []
    for p in peaks_info:
        start = max(0, p["peak_time"] - buffer_before)
        end = p["peak_time"] + buffer_after
        intervals.append({"start": start, "end": end, "max_score": p["score"], "peaks": [p["peak_time"]]})

    intervals.sort(key=lambda x: x["start"])

    merged_intervals = []
    for current in intervals:
        if not merged_intervals:
            merged_intervals.append(current)
            continue
        prev = merged_intervals[-1]
        if current["start"] <= prev["end"]:
            prev["end"] = max(prev["end"], current["end"])
            prev["max_score"] = max(prev["max_score"], current["max_score"])
            prev["peaks"].extend(current["peaks"])
        else:
            merged_intervals.append(current)

    merged_intervals.sort(key=lambda x: x["max_score"], reverse=True)
    final_intervals = merged_intervals[:top_n]
    final_intervals.sort(key=lambda x: x["start"])

    for i, interval in enumerate(final_intervals):
        logging.info(f"特定された区間 {i+1}: {format_time(interval['start'])} ~ {format_time(interval['end'])} (スコア: {interval['max_score']:.4f})")

    return final_intervals
