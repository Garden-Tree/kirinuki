import pandas as pd

def format_time(seconds):
    """秒数を HH:MM:SS 形式の文字列に変換します。"""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def normalize_series(series):
    """
    pandas Series をパーセンタイルランク（0〜1）に正規化します。
    スケールが異なるシグナルを公平に比較するために使用します。
    """
    if series.max() == series.min():
        return pd.Series(0.0, index=series.index)
    return series.rank(pct=True)
