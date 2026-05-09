import os
import logging
from core.utils import format_time

def transcribe_clips(clip_paths, output_dir, model_name="turbo", subtitle_delay=1.0):
    try:
        import whisper
    except ImportError:
        logging.error("Whisperライブラリがインストールされていません。'pip install openai-whisper' を実行してください。")
        return

    logging.info(f"Whisperモデル ('{model_name}') をロードしています。初回はダウンロードに時間がかかる場合があります...")
    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        logging.error(f"Whisperモデルのロードに失敗しました: {e}")
        return

    for clip_path in clip_paths:
        if not os.path.exists(clip_path):
            continue
            
        logging.info(f"文字起こしを開始します: {os.path.basename(clip_path)}")
        try:
            # 日本語を指定して文字起こし
            result = model.transcribe(clip_path, language="ja")
            
            base_name = os.path.splitext(os.path.basename(clip_path))[0]
            srt_path = os.path.join(output_dir, f"{base_name}.srt")
            txt_path = os.path.join(output_dir, f"{base_name}.txt")
            
            with open(srt_path, "w", encoding="utf-8") as f_srt, \
                 open(txt_path, "w", encoding="utf-8") as f_txt:
                
                for i, segment in enumerate(result["segments"], start=1):
                    def srt_format_time(seconds):
                        seconds = max(0.0, float(seconds) + subtitle_delay)
                        h = int(seconds / 3600)
                        m = int((seconds % 3600) / 60)
                        s = int(seconds % 60)
                        ms = int((seconds - int(seconds)) * 1000)
                        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                    
                    start_str = srt_format_time(segment["start"])
                    end_str = srt_format_time(segment["end"])
                    text = segment["text"].strip()
                    
                    f_srt.write(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")
                    f_txt.write(f"{text}\n")
                    
            logging.info(f"文字起こし完了: {srt_path}, {txt_path}")
        except Exception as e:
            logging.error(f"文字起こし中にエラーが発生しました ({clip_path}): {e}")
