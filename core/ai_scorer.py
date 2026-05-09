import os
import json
import logging
import pandas as pd
import time as time_module
from google import genai
from core.utils import format_time

def _save_ai_cache(results, ai_cache_path):
    """AIスコア結果をキャッシュファイルに保存します。"""
    cache_results = [{"window": r["window"], "chat_score": r.get("score", r.get("chat_score", 0))} for r in results]
    try:
        with open(ai_cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_results, f, ensure_ascii=False, indent=2)
        logging.info(f"AIスコアをキャッシュしました: {ai_cache_path} ({len(cache_results)}件)")
    except Exception as e:
        logging.warning(f"AIスコアキャッシュの保存に失敗: {e}")

def score_with_ai(chat_data, api_key, window_sec, ai_cache_path="ai_score_cache.json"):
    """
    Gemini API を用いてチャットの文脈からスコアリングを行います。
    結果は ai_cache_path にキャッシュされ、2回目以降はAPIを呼びません。
    429エラー時は指数バックオフでリトライします。
    """
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
            logging.info(f"AI解析中... バッチ {batch_num}/{total_batches} ({format_time(batch_windows[0])} ~ {format_time(batch_windows[-1])})")

            prompt = "以下はVTuber配信（ゲーム実況等）のチャットログです。指定された区間(window)ごとに分かれています。\n"
            prompt += "それぞれの区間の盛り上がりを、視聴者の「熱量（驚き・興奮・感動・考察など）」の文脈を最重視して0〜10点で厳しくスコア付けし、JSON配列で出力してください。\n"
            prompt += "【スコアの基準】\n"
            prompt += "・0〜2点: 挨拶、待機コメント、無関係な雑談、またはコメント無し\n"
            prompt += "・3〜5点: 通常のプレイに対する平坦なリアクション\n"
            prompt += "・6〜8点: 大きなリアクション、「草」などの笑い、ゲーム展開への具体的なツッコミや考察が多数発生している\n"
            prompt += "・9〜10点: 配信の最大の見せ場。視聴者が一斉に驚愕したり、深い感動や爆笑の渦に包まれている\n"
            prompt += "※注意: 短いスラング（wなど）が連発されていなくても、長文で具体的なゲームの考察や、配信者との濃密なやり取りが交わされている区間は「熱量が高い」と判断して高く評価してください。\n"
            prompt += "【出力形式（厳守）】\n"
            prompt += '[{"window": <window数値>, "score": <0から10の数値>}, ...]\n\n'
            prompt += "【チャットログ】\n"

            for w in batch_windows:
                prompt += f"--- Window: {w} ({format_time(w)}) ---\n"
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
                        model='gemini-2.5-flash',
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
