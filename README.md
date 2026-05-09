# Auto Clipper (ハイライト自動抽出スクリプト)

VTuber等の配信アーカイブ動画から、チャットの盛り上がり、音量の変化、エンゲージメントデータ（CSV）などを総合的に解析し、自動的にハイライト（切り抜き）動画を生成するスクリプトです。

## 特徴
- **チャット解析**: コメントの加速度（急増）や特定のキーワード、AI（Gemini）を用いた文脈解析により、盛り上がりを検知します。
- **音声解析**: 動画内の急激な音量増加（繰り返される大きなリアクション）を検知します。単純な音量ではなく、ピークの回数と大きさを評価します。
- **データ連携**: YouTubeの `liveViewership.csv` などを読み込み、視聴者の維持率やエンゲージメント数をスコアに反映します。
- **設定の外部化**: 各シグナルの重み付けやAPIキーは `config.json` で簡単に管理・調整が可能です。

## 前提条件・インストール
- Python 3.8 以上
- FFmpeg（システムにインストールされ、パスが通っている必要があります）

```bash
# 仮想環境の作成とアクティベート（推奨）
python -m venv venv
venv\Scripts\activate  # Windowsの場合

# 必要なライブラリのインストール
pip install pandas numpy scipy ffmpeg-python yt-dlp google-genai python-dotenv
```

## 使い方

基本の実行コマンドは以下の通りです。動画ファイルと配信のURL（チャットログ取得用）が必須となります。

```bash
python auto_clipper.py --video "archive.mp4" --url "https://www.youtube.com/watch?v=XXXXXXX"
```

### 主なコマンドライン引数

- `--video` **(必須)**: 入力動画ファイルのパス（例: `video.mp4`）
- `--url` **(必須)**: 対象配信のYouTube URL
- `--csv`: `liveViewership.csv` のパス（指定しない場合は同フォルダから自動検出）
- `--engagements-csv`: `liveEngagements.csv` のパス（指定しない場合は同フォルダから自動検出）
- `--output`: 切り抜き動画の保存先ディレクトリ（デフォルト: `output`）
- `--top_n`: 出力する切り抜き動画の数（`config.json`での指定より優先されます）
- `--config`: 設定ファイルのパス（デフォルト: `config.json`）
- `--no-ai`: AIを利用したチャットの高度なスコアリングを無効にします（現在はデフォルトでONになっています）
- `--api-key`: Gemini APIキー（コマンドライン引数、環境変数、または `.env` ファイルにて指定可能）
- `--transcribe`: 出力されたハイライト動画に対し、Whisperを用いて文字起こし（字幕 `.srt` およびテキスト `.txt`）を自動生成します
- `--whisper-model`: Whisperのモデルサイズを指定します（例: `tiny`, `base`, `small`, `medium`, `large`, `turbo`）。デフォルトは最新の高精度かつ高速な `turbo` です
- `--subtitle-delay`: 字幕の表示タイミングを遅らせる秒数を指定します（デフォルト: `1.0`）。抽出された動画の音声ズレを補正するために使用します
- `--shorts`: 抽出した動画をYouTube Shorts等の縦型（1080x1920・黒背景）に変換し、文字起こしが有効な場合は字幕もハードサブとして焼き付けます
- `--top-image`: `--shorts` 実行時に、画面上部の黒帯部分に配置する静止画（PNGやJPG）のパスを指定します

## 設定ファイルとAPIキーについて

当スクリプトでは、セキュリティ（APIキーの漏洩防止）と利便性を両立させるため、以下の2つのファイルを使用します。

1. **`.env` (秘密情報)**: Gemini APIキーなどの公開してはいけない情報を記述します。（`.gitignore` で除外されています）
2. **`config.json` (公開設定)**: 抽出ロジックの重み付けやチューニングパラメータを記述します。（Gitで履歴管理・共有が可能）

### 1. APIキーの設定 (.env)
リポジトリに含まれる `.env.example` をコピーまたはリネームして `.env` というファイルを作成し、ご自身のAPIキーを記載してください。
```bash
GOOGLE_API_KEY=あなたの_GEMINI_API_KEY
```

### 2. パラメータの設定 (config.json)

スコアの計算方法や抽出のパラメータは `config.json` で調整できます。スクリプト実行時、同じディレクトリにある `config.json` が自動的に読み込まれます。

```json
{
  "window_sec": 60,
  "moving_average_window": 2,
  "buffer_before": 30,
  "buffer_after": 40,
  "top_n_clips": 5,
  "signal_weights": {
    "chat": 0.25,        // チャット（AI文脈・キーワード）のスコア重み
    "csv": 0.15,         // CSV内のエンゲージメント・リアクション数の重み
    "retention": 0.35,   // CSV内の視聴者維持率・同接の重み
    "audio": 0.20,       // 音量ピーク（大きなリアクション）の重み
    "superchat": 0.0,    // スパチャ・メンバーシップの重み（0.0で無効化）
    "chat_accel": 0.30   // チャット速度の加速度（急増具合）の重み
  },
  "keyword_weights": {
    "草": 2, "w": 1, "かわいい": 3, "！？": 2,
    "助かる": 2, "てぇてぇ": 3, "神": 3, "あ": 1
  }
}
```

### チューニングのヒント
- **スパチャをハイライト基準に入れたい場合**: `superchat` の数値を `0.15` などに変更してください。
- **切り抜きの長さを変えたい場合**: `buffer_before`（ピーク前の秒数）と `buffer_after`（ピーク後の秒数）を調整してください。
