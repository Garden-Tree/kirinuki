import os
import sys
import logging
import argparse
import glob
from dotenv import load_dotenv

# Import from core package
from core.constants import (
    DEFAULT_WINDOW_SEC,
    DEFAULT_MOVING_AVERAGE_WINDOW,
    DEFAULT_BUFFER_BEFORE,
    DEFAULT_BUFFER_AFTER,
    DEFAULT_TOP_N_CLIPS,
    DEFAULT_SUBTITLE_DELAY,
    SIGNAL_WEIGHTS,
    DEFAULT_KEYWORD_WEIGHTS
)
from core.config_loader import load_config
from core.downloader import fetch_chat_log
from core.analyzer import analyze_and_score, determine_clip_intervals
from core.video_editor import extract_clips, generate_shorts
from core.transcriber import transcribe_clips

load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(
        description="VTuber配信アーカイブ用 ハイライト自動抽出スクリプト",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="python auto_clipper.py --video <動画パス> --url <YouTube_URL> [オプション]"
    )
    parser.add_argument("--video", type=str, required=True, help="入力動画ファイルのパス (例: video.mp4)")
    parser.add_argument("--url", type=str, required=True, help="対象配信のYouTube URL (チャット取得用)")
    parser.add_argument("--csv", type=str, default=None, help="liveViewership.csv のパス (任意・自動検出あり)")
    parser.add_argument("--engagements-csv", type=str, default=None, help="liveEngagements.csv のパス (任意・自動検出あり)")
    parser.add_argument("--output", type=str, default="output", help="出力ディレクトリのパス (デフォルト: output)")
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N_CLIPS, help="出力する上位N件のクリップ数")
    parser.add_argument("--cache", type=str, default="chat_cache.json", help="チャットログのキャッシュJSON。デフォルト: chat_cache.json")
    parser.add_argument("--use-ai", action="store_true", help="互換性のため残しています（現在はデフォルトで有効です）")
    parser.add_argument("--no-ai", action="store_true", help="Gemini APIを利用した文脈スコアリングを無効にする")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini APIキー。環境変数 GOOGLE_API_KEY での指定も可")
    parser.add_argument("--ai-cache", type=str, default="ai_score_cache.json", help="AIスコアのキャッシュJSON。デフォルト: ai_score_cache.json")
    parser.add_argument("--no-audio", action="store_true", help="音声の音量解析を無効にする")
    parser.add_argument("--config", type=str, default="config.json", help="設定ファイル (デフォルト: config.json)")
    parser.add_argument("--transcribe", action="store_true", help="Whisperを使用して切り抜き動画の文字起こし（SRT字幕・テキスト生成）を行う")
    parser.add_argument("--whisper-model", type=str, default="turbo", help="Whisperのモデルサイズ (tiny, base, small, medium, large, turbo)。デフォルト: turbo")
    parser.add_argument("--shorts", action="store_true", help="切り抜き動画を縦型（1080x1920）にし、字幕を焼き付けたショート動画を生成する")
    parser.add_argument("--subtitle-delay", type=float, default=1.0, help="字幕の表示タイミングを遅らせる秒数（デフォルト: 1.0）")
    parser.add_argument("--top-image", type=str, default=None, help="ショート動画の上部黒帯に表示する静止画のパス")

    args = parser.parse_args()
    
    config = load_config(args.config)

    video_path = args.video
    url = args.url
    
    csv_path = args.csv
    if not csv_path:
        candidates = glob.glob("liveViewership*.csv")
        if candidates:
            csv_path = candidates[0]
            logging.info(f"自動検出: {csv_path} を使用します")
            
    eng_csv_path = args.engagements_csv
    if not eng_csv_path:
        candidates = glob.glob("liveEngagements*.csv")
        if candidates:
            eng_csv_path = candidates[0]
            logging.info(f"自動検出: {eng_csv_path} を使用します")

    output_dir = args.output
    top_n = args.top_n if args.top_n != DEFAULT_TOP_N_CLIPS else config.get("top_n_clips", DEFAULT_TOP_N_CLIPS)
    cache_path = args.cache
    use_ai = not args.no_ai
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or config.get("api_key")
    ai_cache_path = args.ai_cache

    if use_ai and not api_key:
        logging.warning("AIスコアリングが有効ですが、APIキーが設定されていません。.envファイル等を確認してください。AIスコアリングをスキップします。")
        use_ai = False

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
    subtitle_delay = args.subtitle_delay if args.subtitle_delay != DEFAULT_SUBTITLE_DELAY else config.get("subtitle_delay", DEFAULT_SUBTITLE_DELAY)

    # Step 1: 解析とスコアリング
    df_scores = analyze_and_score(
        chat_data=chat_data,
        csv_path=csv_path,
        engagements_csv_path=eng_csv_path,
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
    clip_paths = extract_clips(video_path=video_path, intervals=intervals, output_dir=output_dir)

    # Step 4: Whisperによる文字起こし (オプション)
    if args.transcribe and clip_paths:
        transcribe_clips(clip_paths, output_dir, model_name=args.whisper_model, subtitle_delay=subtitle_delay)
        
    # Step 5: ショート動画化 (オプション)
    if args.shorts and clip_paths:
        generate_shorts(clip_paths, output_dir, top_image=args.top_image)

    logging.info("全ての処理が完了しました！")

if __name__ == "__main__":
    main()
