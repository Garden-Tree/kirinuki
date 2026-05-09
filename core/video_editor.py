import os
import logging
import ffmpeg
from core.utils import format_time

def extract_clips(video_path, intervals, output_dir):
    """
    FFmpegを用いて動画から指定区間を切り出します。無劣化(-c copy)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"出力ディレクトリを作成しました: {output_dir}")

    clip_paths = []
    for i, interval in enumerate(intervals):
        start_sec = interval["start"]
        end_sec = interval["end"]
        duration = end_sec - start_sec
        clip_filename = f"clip_{str(i+1).zfill(2)}.mp4"
        output_path = os.path.join(output_dir, clip_filename)

        logging.info(f"切り出しを実行中...: {clip_filename} ({format_time(start_sec)} ~ {format_time(end_sec)})")

        try:
            (
                ffmpeg
                .input(video_path, ss=start_sec)
                .output(output_path, t=duration, c='copy')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logging.info(f"成功: {output_path}")
            clip_paths.append(output_path)
        except ffmpeg.Error as e:
            logging.error(f"FFmpegエラーが発生しました: {clip_filename}")
            try:
                logging.error(e.stderr.decode('utf-8'))
            except:
                pass
    return clip_paths

def generate_shorts(clip_paths, output_dir, top_image=None):
    logging.info("ショート動画（縦型・字幕付き）の生成を開始します...")
    
    for clip_path in clip_paths:
        if not os.path.exists(clip_path):
            continue
            
        base_name = os.path.splitext(os.path.basename(clip_path))[0]
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        shorts_path = os.path.join(output_dir, f"{base_name}_shorts.mp4")
        
        has_srt = os.path.exists(srt_path)
        logging.info(f"ショート動画生成中: {os.path.basename(shorts_path)}")
        
        try:
            # フィルターのパスエスケープ問題を回避するため、実行ディレクトリをoutput_dirに変更して相対パスで処理する
            input_file = f"{base_name}.mp4"
            output_file = f"{base_name}_shorts.mp4"
            srt_file = f"{base_name}.srt"
            
            stream = ffmpeg.input(input_file)
            audio = stream.audio
            video = stream.video
            
            # 幅を1080にリサイズ（アスペクト比維持、高さは偶数）
            video = ffmpeg.filter(video, 'scale', 1080, -2)
            # 1080x1920の黒背景に中央配置
            video = ffmpeg.filter(video, 'pad', 1080, 1920, '(ow-iw)/2', '(oh-ih)/2', color='black')
            
            if top_image and os.path.exists(top_image):
                top_image_abs = os.path.abspath(top_image)
                image_input = ffmpeg.input(top_image_abs, loop=1)
                # 画像の幅を1080に、高さは最大656（アスペクト比維持）にリサイズ
                image_scaled = ffmpeg.filter(image_input, 'scale', 1080, 656, force_original_aspect_ratio='decrease')
                # 画面上部（y=(656-h)/2）の中央に配置
                video = ffmpeg.filter([video, image_scaled], 'overlay', x='(main_w-overlay_w)/2', y='(656-overlay_h)/2', shortest=1)

            if has_srt:
                # 字幕の焼き付け
                # ※SRTを焼き付ける際のFFmpegの内部解像度（PlayResY=288）に合わせてMarginVを調整
                video = ffmpeg.filter(video, 'subtitles', srt_file, force_style='FontSize=22,Alignment=2,MarginV=45,Outline=1.5,Shadow=1')
            
            original_cwd = os.getcwd()
            os.chdir(output_dir)
            try:
                (
                    ffmpeg
                    .output(video, audio, output_file, vcodec='libx264', acodec='copy')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            finally:
                os.chdir(original_cwd)
            logging.info(f"ショート生成完了: {shorts_path}")
        except ffmpeg.Error as e:
            logging.error(f"ショート生成中にFFmpegエラーが発生しました ({clip_path}):")
            try:
                logging.error(e.stderr.decode('utf-8'))
            except:
                pass
        except Exception as e:
            logging.error(f"予期せぬエラー: {e}")
