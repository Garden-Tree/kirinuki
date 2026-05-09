# Default Settings
DEFAULT_WINDOW_SEC = 60
DEFAULT_MOVING_AVERAGE_WINDOW = 2  # e.g., 2 minutes
DEFAULT_BUFFER_BEFORE = 30
DEFAULT_BUFFER_AFTER = 40  # ちょっと余韻を残すために40秒
DEFAULT_TOP_N_CLIPS = 5

# シグナルの重み付け設定
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
