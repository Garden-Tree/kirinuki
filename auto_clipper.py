import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import ffmpeg
from scipy.signal import find_peaks
from dotenv import load_dotenv

load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# Default Settings
# ============================================================
DEFAULT_WINDOW_SEC = 60
DEFAULT_MOVING_AVERAGE_WINDOW = 2  # e.g., 2 minutes
DEFAULT_BUFFER_BEFORE = 30
DEFAULT_BUFFER_AFTER = 40  # ちょっと余韻を残すために40秒
DEFAULT_TOP_N_CLIPS = 5

# シグナルの重み付け設定（合計を1.0にする必要はないが、相対比が重要）
SIGNAL_WEIGHTS = {
    "chat":       0.35,   # チャット解析（AI文脈 or キーワード）
    "csv":        0.15,   # CSV（エンゲージメント・リアクション）
    "audio":      0.20,   # 音声の音量変化
    "superchat":  0.15,   # スパチャ・メンバーシップ
    "chat_accel": 0.15,   # チャット密度の加速度（急増検出）
}

# キーワード重み付け（--use-ai 不使用時のフォールバック）
DEFAULT_KEYWORD_WEIGHTS = {
    "草": 2, "w": 1, "かわいい": 3, "！？": 2,
    "助かる": 2, "てぇてぇ": 3, "神": 3, "あ": 1
}


# ============================================================
# ユーティリティ
# ============================================================
def _format_time(seconds):
    """秒数を HH:MM:SS 形式の文字列に変換します。"""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _normalize_series(series):
    """
    pandas Series をパーセンタイルランク（0〜1）に正規化します。
    スケールが異なるシグナルを公平に比較するために使用します。
    """
    if series.max() == series.min():
        return pd.Series(0.0, index=series.index)
    return series.rank(pct=True)


# ============================================================
# Step 0: チャットログの取得（スパチャ・メンバーシップも含む）
# ============================================================
def fetch_chat_log(url, cache_path="chat_cache.json"):
    """
    指定されたURLからチャットログを取得します。
    通常のテキストメッセージに加え、スパチャ・メンバーシップイベントも抽出します。
    yt-dlp を使用して安定した取得を行います。
    """
    if os.path.exists(cache_path):
        logging.info(f"キャッシュされたチャットログを読み込みます: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    logging.info(f"チャットログのダウンロードを開始します (yt-dlp使用): {url}")
    import subprocess
    import tempfile

    chat_data = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_output = os.path.join(tmpdir, "chat.%(ext)s")
        cmd = [
            sys.executable,
            "-m", "yt_dlp",
            "--skip-download",
            "--write-sub",
            "--sub-lang", "live_chat",
            url,
            "-o", tmp_output
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            logging.error("yt-dlpの実行に失敗しました。URLや動画が存在するか確認してください。")
            return chat_data
        except FileNotFoundError:
            logging.error("yt-dlp コマンドが見つかりません。pip install yt-dlp を実行してください。")
            return chat_data

        json_file = os.path.join(tmpdir, "chat.live_chat.json")
        if not os.path.exists(json_file):
            logging.error("チャットログファイルが生成されませんでした。生放送アーカイブではない可能性があります。")
            return chat_data

        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    action = data.get("replayChatItemAction", {})
                    actions = action.get("actions", [])
                    video_offset_msec = action.get("videoOffsetTimeMsec", "0")
                    time_sec = int(video_offset_msec) / 1000.0

                    for act in actions:
                        item = act.get("addChatItemAction", {}).get("item", {})

                        # 通常のテキストメッセージ
                        if "liveChatTextMessageRenderer" in item:
                            renderer = item["liveChatTextMessageRenderer"]
                            runs = renderer.get("message", {}).get("runs", [])
                            text = "".join([r.get("text", "") for r in runs if "text" in r])
                            author = renderer.get("authorName", {}).get("simpleText", "")
                            if text:
                                chat_data.append({
                                    "time_sec": time_sec,
                                    "message": text,
                                    "author": author,
                                    "type": "text"
                                })

                        # スーパーチャット（有料メッセージ）
                        elif "liveChatPaidMessageRenderer" in item:
                            renderer = item["liveChatPaidMessageRenderer"]
                            amount = renderer.get("purchaseAmountText", {}).get("simpleText", "")
                            chat_data.append({
                                "time_sec": time_sec,
                                "message": f"[スパチャ: {amount}]",
                                "author": renderer.get("authorName", {}).get("simpleText", ""),
                                "type": "superchat"
                            })

                        # メンバーシップ加入
                        elif "liveChatMembershipItemRenderer" in item:
                            chat_data.append({
                                "time_sec": time_sec,
                                "message": "[メンバーシップ加入]",
                                "author": item["liveChatMembershipItemRenderer"].get("authorName", {}).get("simpleText", ""),
                                "type": "membership"
                            })

                        # メンバーシップギフト
                        elif "liveChatSponsorshipsGiftPurchaseAnnouncementRenderer" in item:
                            chat_data.append({
                                "time_sec": time_sec,
                                "message": "[メンバーシップギフト]",
                                "author": "",
                                "type": "superchat"
                            })

                except Exception:
                    continue

    # Save cache
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)

        type_counts = {}
        for msg in chat_data:
            t = msg.get("type", "text")
            type_counts[t] = type_counts.get(t, 0) + 1
        logging.info(f"チャットログをキャッシュしました: {cache_path} ({type_counts})")
    except Exception as e:
        logging.warning(f"キャッシュの保存に失敗しました: {e}")

    return chat_data


# ============================================================
# AIスコアキャッシュ
# ============================================================
def _save_ai_cache(results, ai_cache_path):
    """AIスコア結果をキャッシュファイルに保存します。"""
    cache_results = [{"window": r["window"], "chat_score": r.get("score", r.get("chat_score", 0))} for r in results]
    try:
        with open(ai_cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_results, f, ensure_ascii=False, indent=2)
        logging.info(f"AIスコアをキャッシュしました: {ai_cache_path} ({len(cache_results)}件)")
    except Exception as e:
        logging.warning(f"AIスコアキャッシュの保存に失敗: {e}")


# ============================================================
# Step 1a: AI文脈スコアリング
# ============================================================
def score_with_ai(chat_data, api_key, window_sec, ai_cache_path="ai_score_cache.json"):
    """
    Gemini API を用いてチャットの文脈からスコアリングを行います。
    結果は ai_cache_path にキャッシュされ、2回目以降はAPIを呼びません。
    429エラー時は指数バックオフでリトライします。
    """
    from google import genai
    import time as time_module

    if not chat_data:
        return pd.DataFrame()

    # キャッシュが存在すれば読み込み
    if os.path.exists(ai_cache_path):
        logging.info(f"キャッシュされたAIスコアを読み込みます: {ai_cache_path}")
        with open(ai_cache_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        df_ai = pd.DataFrame(results)
        if not df_ai.empty:
            df_ai['window'] = pd.to_numeric(df_ai['window'])
            df_ai['chat_score'] = pd.to_numeric(df_ai['chat_score'])
            df_ai = df_ai.groupby('window')['chat_score'].mean().reset_index()
        return df_ai

    logging.info("Gemini APIでの文脈解析を開始します...")
    client = genai.Client(api_key=api_key)

    # テキストメッセージのみフィルタ
    text_messages = [m for m in chat_data if m.get("type", "text") == "text"]
    df_chat = pd.DataFrame(text_messages)
    df_chat["window"] = (df_chat["time_sec"] // window_sec) * window_sec

    batch_size = 50
    windows = sorted(list(df_chat['window'].unique()))
    results = []

    total_batches = (len(windows) + batch_size - 1) // batch_size
    max_retries = 3
    base_wait = 30

    try:
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i+batch_size]
            batch_num = i // batch_size + 1
            logging.info(f"AI解析中... バッチ {batch_num}/{total_batches} ({_format_time(batch_windows[0])} ~ {_format_time(batch_windows[-1])})")

            prompt = "以下はVTuber配信のチャットログです。指定された区間(window)ごとに分かれています。\n"
            prompt += "それぞれの区間の盛り上がりを、視聴者の驚き・興奮・感動などの文脈を最重視して0〜10点でスコア付けし、JSON配列で出力してください。\n"
            prompt += "単なるシステムメッセージや定型挨拶は1〜2点、大きなリアクションや面白い展開へのツッコミは7〜10点にしてください。\n"
            prompt += "【出力形式（厳守）】\n"
            prompt += '[{"window": <window数値>, "score": <0から10の数値>}, ...]\n\n'
            prompt += "【チャットログ】\n"

            for w in batch_windows:
                prompt += f"--- Window: {w} ({_format_time(w)}) ---\n"
                messages = df_chat[df_chat['window'] == w]['message'].tolist()
                if not messages:
                    prompt += "(コメントなし)\n"
                else:
                    for msg in messages:
                        prompt += f"- {msg}\n"

            success = False
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=prompt,
                        config={'response_mime_type': 'application/json'}
                    )
                    batch_result = json.loads(response.text)
                    for item in batch_result:
                        if 'window' in item and 'score' in item:
                            results.append(item)
                    success = True
                    break
                except Exception as e:
                    error_str = str(e)
                    if '429' in error_str:
                        wait_time = base_wait * (2 ** attempt)
                        logging.warning(f"レート制限 (429) に到達しました。{wait_time}秒待機してリトライします... (試行 {attempt+1}/{max_retries})")
                        time_module.sleep(wait_time)
                    else:
                        logging.error(f"Gemini API実行エラー: {e}")
                        break

            if not success:
                logging.warning(f"バッチ {batch_num} はスキップされました（スコア0として処理）")
                for w in batch_windows:
                    results.append({"window": w, "score": 0})

            if batch_num < total_batches:
                time_module.sleep(4)

    except KeyboardInterrupt:
        logging.warning("中断されました。ここまでの結果を保存します...")
    finally:
        if results:
            _save_ai_cache(results, ai_cache_path)

    df_ai = pd.DataFrame(results)
    if not df_ai.empty:
        df_ai.rename(columns={"score": "chat_score"}, inplace=True)
        df_ai['window'] = pd.to_numeric(df_ai['window'])
        df_ai['chat_score'] = pd.to_numeric(df_ai['chat_score'])
        df_ai = df_ai.groupby('window')['chat_score'].mean().reset_index()
    else:
        df_ai = pd.DataFrame(columns=["window", "chat_score"])

    return df_ai


# ============================================================
# Step 1b: 音声解析（音量変化量）
# ============================================================
def analyze_audio(video_path, window_sec=DEFAULT_WINDOW_SEC):
    """
    動画から音声のRMS（音量）を短いサブウィンドウ毎に計算し、
    指定ウィンドウ内に「繰り返して大きいリアクション（音量ピーク）」が
    何度あったか（ピークの高さの合計）をスコアとして返します。
    """
    import subprocess
    import tempfile

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


# ============================================================
# Step 1c: スパチャ・メンバーシップスコア
# ============================================================
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


# ============================================================
# Step 1d: チャット密度の加速度
# ============================================================
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


# ============================================================
# Step 1: 統合スコアリング（全シグナルを正規化して合算）
# ============================================================
def analyze_and_score(chat_data, csv_path=None, video_path=None,
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
    if csv_path and os.path.exists(csv_path):
        logging.info(f"CSVデータを統合します: {csv_path}")
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
            for col in ['視聴者の維持率', '相対的な視聴者の維持率', '維持率', '同時接続数']:
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

                merge_cols = ['window']
                if 'csv_score' in df_csv_windowed.columns: merge_cols.append('csv_score')
                if 'retention_score' in df_csv_windowed.columns: merge_cols.append('retention_score')
                
                df_scores = pd.merge(df_scores, df_csv_windowed[merge_cols], on='window', how='left').fillna(0)
        except Exception as e:
            logging.error(f"CSVの読み込み中にエラーが発生しました: {e}")

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
            normalized = _normalize_series(df_scores[col_name])
            df_scores[f'norm_{col_name}'] = normalized
            df_scores['total_score'] += normalized * weight
            raw_max = df_scores[col_name].max()
            logging.info(f"  {signal_name:12s}: weight={weight:.2f}, max_raw={raw_max:.2f}")

    # 移動平均を算出してスパイクを滑らかにする
    df_scores['smoothed_score'] = df_scores['total_score'].rolling(window=ma_window, min_periods=1).mean()

    return df_scores


# ============================================================
# Step 2: 切り抜き区間の決定
# ============================================================
def determine_clip_intervals(df_scores, top_n=DEFAULT_TOP_N_CLIPS,
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
        logging.info(f"特定された区間 {i+1}: {_format_time(interval['start'])} ~ {_format_time(interval['end'])} (スコア: {interval['max_score']:.4f})")

    return final_intervals


# ============================================================
# Step 3: FFmpegによる動画の切り出し
# ============================================================
def extract_clips(video_path, intervals, output_dir):
    """
    FFmpegを用いて動画から指定区間を切り出します。無劣化(-c copy)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"出力ディレクトリを作成しました: {output_dir}")

    for i, interval in enumerate(intervals):
        start_sec = interval["start"]
        end_sec = interval["end"]
        duration = end_sec - start_sec
        clip_filename = f"clip_{str(i+1).zfill(2)}.mp4"
        output_path = os.path.join(output_dir, clip_filename)

        logging.info(f"切り出しを実行中...: {clip_filename} ({_format_time(start_sec)} ~ {_format_time(end_sec)})")

        try:
            (
                ffmpeg
                .input(video_path, ss=start_sec)
                .output(output_path, t=duration, c='copy')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logging.info(f"成功: {output_path}")
        except ffmpeg.Error as e:
            logging.error(f"FFmpegエラーが発生しました: {clip_filename}")
            try:
                logging.error(e.stderr.decode('utf-8'))
            except:
                pass


# ============================================================
# メインエントリポイント
# ============================================================
def load_config(config_path):
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"設定ファイルの読み込みに失敗しました: {e}")
    return {}

def main():
    parser = argparse.ArgumentParser(
        description="VTuber配信アーカイブ用 ハイライト自動抽出スクリプト",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="python auto_clipper.py --video <動画パス> --url <YouTube_URL> [オプション]"
    )
    parser.add_argument("--video", type=str, required=True, help="入力動画ファイルのパス (例: video.mp4)")
    parser.add_argument("--url", type=str, required=True, help="対象配信のYouTube URL (チャット取得用)")
    parser.add_argument("--csv", type=str, default=None, help="liveViewership.csv のパス (任意)")
    parser.add_argument("--output", type=str, default="output", help="出力ディレクトリのパス (デフォルト: output)")
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N_CLIPS, help="出力する上位N件のクリップ数")
    parser.add_argument("--cache", type=str, default="chat_cache.json", help="チャットログのキャッシュJSON。デフォルト: chat_cache.json")
    parser.add_argument("--use-ai", action="store_true", help="Gemini APIを利用して文脈からスコアリングを行う")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini APIキー。環境変数 GOOGLE_API_KEY での指定も可")
    parser.add_argument("--ai-cache", type=str, default="ai_score_cache.json", help="AIスコアのキャッシュJSON。デフォルト: ai_score_cache.json")
    parser.add_argument("--no-audio", action="store_true", help="音声の音量解析を無効にする")
    parser.add_argument("--config", type=str, default="config.json", help="設定ファイル (デフォルト: config.json)")

    args = parser.parse_args()
    
    config = load_config(args.config)

    video_path = args.video
    url = args.url
    csv_path = args.csv
    output_dir = args.output
    top_n = args.top_n if args.top_n != DEFAULT_TOP_N_CLIPS else config.get("top_n_clips", DEFAULT_TOP_N_CLIPS)
    cache_path = args.cache
    use_ai = args.use_ai
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or config.get("api_key")
    ai_cache_path = args.ai_cache

    if use_ai and not api_key:
        logging.error("--use-ai を指定する場合は、--api-key または環境変数 GOOGLE_API_KEY を設定してください。")
        sys.exit(1)

    if not os.path.exists(video_path):
        logging.error(f"指定された動画ファイルが存在しません: {video_path}")
        sys.exit(1)

    # Step 0: チャットログの取得
    chat_data = fetch_chat_log(url, cache_path=cache_path)

    window_sec = config.get("window_sec", DEFAULT_WINDOW_SEC)
    ma_window = config.get("moving_average_window", DEFAULT_MOVING_AVERAGE_WINDOW)
    buffer_before = config.get("buffer_before", DEFAULT_BUFFER_BEFORE)
    buffer_after = config.get("buffer_after", DEFAULT_BUFFER_AFTER)
    signal_weights = config.get("signal_weights", SIGNAL_WEIGHTS)
    keyword_weights = config.get("keyword_weights", DEFAULT_KEYWORD_WEIGHTS)

    # Step 1: 解析とスコアリング
    df_scores = analyze_and_score(
        chat_data=chat_data,
        csv_path=csv_path,
        video_path=video_path,
        window_sec=window_sec,
        keyword_weights=keyword_weights,
        ma_window=ma_window,
        use_ai=use_ai,
        api_key=api_key,
        ai_cache_path=ai_cache_path,
        use_audio=not args.no_audio,
        signal_weights=signal_weights
    )

    if df_scores.empty:
        logging.error("解析対象のデータがありませんでした。処理を終了します。")
        sys.exit(1)

    # Step 2: 切り抜き区間の決定
    intervals = determine_clip_intervals(
        df_scores=df_scores,
        top_n=top_n,
        buffer_before=buffer_before,
        buffer_after=buffer_after
    )

    if not intervals:
        logging.warning("切り抜き対象のピークが見つかりませんでした。")
        sys.exit(0)

    # Step 3: FFmpegによる動画の切り出し
    extract_clips(video_path=video_path, intervals=intervals, output_dir=output_dir)

    logging.info("全ての処理が完了しました！")

if __name__ == "__main__":
    main()
