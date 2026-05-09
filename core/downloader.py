import os
import sys
import json
import logging
import subprocess
import tempfile

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
