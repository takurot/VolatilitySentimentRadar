import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import json
import random
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import concurrent.futures

# yfinance で株価・ビットコイン・先物などを取得
import yfinance as yf

# PRAW (Python Reddit API Wrapper) を使ったRedditデータ取得
import praw

# 自然言語処理で簡易的にセンチメント分析
from textblob import TextBlob

########################################
# 1. 各種データの取得
########################################

def fetch_stock_data(ticker="^GSPC", period="1mo", interval="1d"):
    """
    yfinance を使って株価データを取得
    :param ticker: ティッカーシンボル (S&P 500: ^GSPC, など)
    :param period: データの期間(例: "1mo", "3mo", "1y" ...)
    :param interval: 取得頻度(例: "1d", "1h", "15m" ...)
    :return: pandas.DataFrame
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    df["Returns"] = df["Close"].pct_change()
    return df


def fetch_bitcoin_data(ticker="BTC-USD", period="1mo", interval="1d"):
    """
    yfinance を使ってビットコイン価格データを取得
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    df["Returns"] = df["Close"].pct_change()
    return df


def fetch_futures_data(ticker="ES=F", period="1mo", interval="1d"):
    """
    yfinance を使って先物価格データを取得 (例: S&P 500 E-mini 先物)
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    df["Returns"] = df["Close"].pct_change()
    return df


def fetch_all_futures_data(period="1mo", interval="1d"):
    """
    主要な先物市場のデータを取得
    """
    futures_symbols = {
        "ES=F": "S&P 500 E-mini",
        "NQ=F": "NASDAQ 100 E-mini",
        "YM=F": "Dow E-mini",
        "RTY=F": "Russell 2000 E-mini",
        "GC=F": "金先物",
        "SI=F": "銀先物",
        "CL=F": "原油先物",
        "ZB=F": "米国債先物"
    }
    
    futures_data = {}
    for symbol, name in futures_symbols.items():
        try:
            df = fetch_futures_data(symbol, period, interval)
            futures_data[symbol] = {
                "name": name,
                "data": df
            }
        except Exception as e:
            print(f"警告: {name}（{symbol}）の取得に失敗: {str(e)}")
            continue
    
    return futures_data


########################################
# 2. Redditデータの取得とセンチメント分析
########################################

def load_reddit_config():
    """
    config.jsonからRedditの設定を読み込む
    """
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("エラー: config.jsonが見つかりません")
        return None
    except json.JSONDecodeError:
        print("エラー: config.jsonの形式が不正です")
        return None


def fetch_all_reddit_posts(
    subreddits=[
        "stocks",
        "investing",
        "wallstreetbets",
        "StockMarket",
        "pennystocks",
        "options",
        "Daytrading",
        "thetagang"
    ],
    mode="hot",
    limit_per_subreddit=50
):
    """
    複数のサブレディットから投稿を取得
    """
    all_posts = []
    config = load_reddit_config()
    if not config:
        return []

    # Reddit API認証
    reddit = praw.Reddit(
        client_id=config.get('reddit_client_id'),
        client_secret=config.get('reddit_client_secret'),
        user_agent=config.get('reddit_user_agent'),
        username=config.get('reddit_username'),
        password=config.get('reddit_password')
    )

    for subreddit_name in subreddits:
        try:
            subreddit_posts = fetch_reddit_posts_internal(reddit, subreddit_name, mode, limit_per_subreddit)
            all_posts.extend(subreddit_posts)
        except Exception as e:
            print(f"警告: {subreddit_name}からの取得中にエラーが発生しました: {str(e)}")
            continue

    return all_posts

def fetch_reddit_posts_internal(reddit, subreddit_name, mode="hot", limit=50):
    """
    単一のサブレディットから投稿を取得（内部関数）
    """
    subreddit = reddit.subreddit(subreddit_name)

    if mode == "hot":
        submissions = subreddit.hot(limit=limit)
    elif mode == "new":
        submissions = subreddit.new(limit=limit)
    else:
        submissions = subreddit.top(limit=limit)

    data = []
    for submission in submissions:
        text = submission.title + " " + (submission.selftext if submission.selftext else "")
        sentiment_score = TextBlob(text).sentiment.polarity
        data.append({
            "subreddit": subreddit_name,
            "title": submission.title,
            "text": text,
            "sentiment": sentiment_score
        })

    return data


def analyze_reddit_sentiment(reddit_data):
    """
    取得したReddit投稿のセンチメントを簡易的に統計情報化
    :param reddit_data: list of dict
    :return: dict (平均スコア、スコアの分布など)
    """
    sentiments = [d["sentiment"] for d in reddit_data]
    if len(sentiments) == 0:
        return {
            "mean_sentiment": 0,
            "median_sentiment": 0,
            "positive_ratio": 0,
            "negative_ratio": 0
        }
    
    positive_ratio = sum(s > 0 for s in sentiments) / len(sentiments)
    negative_ratio = sum(s < 0 for s in sentiments) / len(sentiments)

    return {
        "mean_sentiment": np.mean(sentiments),
        "median_sentiment": np.median(sentiments),
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio
    }


########################################
# 3. 相場変動の検知ロジック (詳細版)
########################################

def analyze_market_trends(df, periods):
    """
    複数の期間での市場トレンドを分析
    """
    results = {}
    for period in periods:
        data = df.tail(period)
        results[period] = {
            "volatility": data["Returns"].std() * 100,
            "return": data["Returns"].sum() * 100,
            "direction": "上昇" if data["Returns"].sum() > 0 else "下落",
            "max_drawdown": calculate_max_drawdown(data["Close"])
        }
    return results

def calculate_max_drawdown(prices):
    """最大ドローダウンを計算"""
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak * 100
    return drawdown.min().iloc[0]  # float()の代わりに.iloc[0]を使用

def detect_market_signals(stock_df, btc_df, futures_data, reddit_sentiment):
    """
    改善案を含めたシグナル検出
    """
    signals = {}
    
    # 1. 期間設定の見直し
    analysis_periods = {
        "very_short_term": 3,    # 3日
        "short_term": 5,         # 1週間
        "medium_term": 20,       # 1ヶ月
        "long_term": 60,         # 3ヶ月
        "very_long_term": 120    # 6ヶ月
    }

    # 各市場の分析を実行
    signals["stock"] = analyze_market_trends(stock_df, analysis_periods.values())
    signals["btc"] = analyze_market_trends(btc_df, analysis_periods.values())
    signals["futures"] = {}
    for symbol, data in futures_data.items():
        signals["futures"][symbol] = analyze_market_trends(data["data"], analysis_periods.values())

    # 2. ボラティリティの動的閾値設定（より厳格に）
    vol_lookback = 252  # 1年
    if len(stock_df) >= vol_lookback:
        historical_vol = stock_df["Returns"].rolling(vol_lookback).std() * 100
        vol_threshold = historical_vol.mean() + 2 * historical_vol.std()  # 2標準偏差に変更
    else:
        vol_threshold = 2.5  # デフォルト値を引き上げ

    bull_signals = 0
    bear_signals = 0

    # 強気と弱気の判定基準を分離
    bull_conditions = {
        "return_threshold": 2.0,      # リターンの閾値
        "consec_days": 3,             # 連続上昇/下落日数
        "sentiment_threshold": 0.6,    # センチメントの閾値
        "futures_consensus": 0.6       # 先物市場のコンセンサス
    }
    
    bear_conditions = {
        "return_threshold": 2.5,      # より厳格なリターンの閾値
        "consec_days": 4,             # より長い連続下落日数
        "sentiment_threshold": 0.7,    # より厳格なセンチメントの閾値
        "futures_consensus": 0.7,      # より強い市場コンセンサス
        "drawdown_threshold": -5.0     # ドローダウンの閾値
    }

    for period, days in analysis_periods.items():
        # 強気シグナルの判定
        if signals["stock"][days]["return"] > bull_conditions["return_threshold"] * np.sqrt(days/20):
            bull_signals += 1.5
            
        # 弱気シグナルの判定
        if signals["stock"][days]["return"] < -bear_conditions["return_threshold"] * np.sqrt(days/20):
            bear_signals += 1.5

        if len(stock_df) >= 200:
            ma20 = stock_df['Close'].rolling(20).mean()
            ma50 = stock_df['Close'].rolling(50).mean()
            ma200 = stock_df['Close'].rolling(200).mean()
            
            current_price = stock_df['Close'].iloc[-1].item()
            
            # ゴールデンクロス（強気）
            if (ma20.iloc[-1].item() > ma50.iloc[-1].item() and 
                ma20.iloc[-2].item() < ma50.iloc[-2].item()):
                bull_signals += 1.0
            
            # デッドクロス（弱気）
            if (ma20.iloc[-1].item() < ma50.iloc[-1].item() and 
                ma20.iloc[-2].item() > ma50.iloc[-2].item()):
                bear_signals += 1.0

            # 長期移動平均との関係
            if current_price > ma200.iloc[-1].item():
                bull_signals += 0.5
            if current_price < ma200.iloc[-1].item():
                bear_signals += 0.5

        # 連続上昇/下落日数のチェック
        consecutive_up_days = 0
        consecutive_down_days = 0
        for i in range(min(days, len(stock_df))):
            if stock_df['Returns'].iloc[-(i+1)] > 0:
                consecutive_up_days += 1
            else:
                break
        
        for i in range(min(days, len(stock_df))):
            if stock_df['Returns'].iloc[-(i+1)] < 0:
                consecutive_down_days += 1
            else:
                break

        if consecutive_up_days >= bull_conditions["consec_days"]:
            bull_signals += 0.5
        if consecutive_down_days >= bear_conditions["consec_days"]:
            bear_signals += 0.5

        # 先物市場のコンセンサス
        try:
            futures_up_count = 0
            futures_down_count = 0
            valid_futures = 0
            
            for symbol, data in futures_data.items():
                if len(data["data"]) >= days:
                    returns = data["data"]["Returns"].tail(days)
                    if not returns.empty and not np.all(returns == returns.iloc[0]):
                        valid_futures += 1
                        if returns.sum() > bull_conditions["return_threshold"]:
                            futures_up_count += 1
                        if returns.sum() < -bear_conditions["return_threshold"]:
                            futures_down_count += 1
            
            if valid_futures > 0:
                if futures_up_count >= valid_futures * bull_conditions["futures_consensus"]:
                    bull_signals += 1.0
                if futures_down_count >= valid_futures * bear_conditions["futures_consensus"]:
                    bear_signals += 1.0
        except Exception:
            pass

    # センチメントによる判定
    if reddit_sentiment["positive_ratio"] > bull_conditions["sentiment_threshold"]:
        if reddit_sentiment["mean_sentiment"] > 0.2:
            bull_signals += 1.0
    if reddit_sentiment["negative_ratio"] > bear_conditions["sentiment_threshold"]:
        if reddit_sentiment["mean_sentiment"] < -0.2:
            bear_signals += 1.0

    # 動的シグナル閾値の調整
    market_volatility = stock_df["Returns"].tail(20).std() * 100
    bull_threshold = 3.0
    bear_threshold = 3.5  # 弱気の基本閾値を調整
    
    if market_volatility > vol_threshold:
        bull_threshold = 4.0
        bear_threshold = 4.5
    elif market_volatility < vol_threshold * 0.5:
        bull_threshold = 2.5
        bear_threshold = 3.0

    # 最終判定
    signals["bull_signal"] = (bull_signals >= bull_threshold)
    signals["bear_signal"] = (bear_signals >= bear_threshold)
    
    return signals


########################################
# 4. メインの実行フロー
########################################

def backtest_signals(ticker="^GSPC", start_date="2020-01-01", end_date=None):
    """
    指定期間でのバックテストを実行
    """
    # データ取得
    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    btc = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
    
    # 結果保存用
    backtest_results = []
    
    # 20営業日（約1ヶ月）ごとにシグナルをチェック
    for i in range(20, len(stock.index), 20):
        current_date = stock.index[i]
        
        # 現時点までのデータでシグナルを計算
        stock_window = stock.iloc[:i].copy()
        btc_window = btc.iloc[:i].copy()
        
        # リターンを計算
        stock_window["Returns"] = stock_window["Close"].pct_change()
        btc_window["Returns"] = btc_window["Close"].pct_change()
        
        # 先物データ（簡略化のため株価の代用可）
        futures_data = {"ES=F": {"name": "S&P 500 E-mini", "data": stock_window}}
        
        # センチメントデータ（過去データの取得は困難なため、ダミー値を使用）
        dummy_sentiment = {
            "mean_sentiment": 0,
            "median_sentiment": 0,
            "positive_ratio": 0.5,
            "negative_ratio": 0.5
        }
        
        # シグナル検出
        signals = detect_market_signals(stock_window, btc_window, futures_data, dummy_sentiment)
        
        # 次の20営業日のリターンを計算（シグナルの成績評価用）
        try:
            if i + 20 < len(stock.index):
                next_close = stock["Close"].iloc[i + 20]
                current_close = stock["Close"].iloc[i]
                forward_return = (next_close - current_close) / current_close * 100
            else:
                forward_return = None
        except Exception as e:
            print(f"警告: インデックス {i} でのリターン計算でエラー: {str(e)}")
            forward_return = None
        
        # 結果を記録
        backtest_results.append({
            "date": current_date,
            "bull_signal": signals["bull_signal"],
            "bear_signal": signals["bear_signal"],
            "forward_return": forward_return
        })
    
    return pd.DataFrame(backtest_results)

def analyze_backtest_results(results):
    """
    バックテスト結果の分析
    """
    # シグナルごとの成績を取得
    bull_mask = results["bull_signal"]
    bear_mask = results["bear_signal"]
    no_signal_mask = ~(bull_mask | bear_mask)
    
    # 各シグナルのforward_returnを取得し、要素ごとにfloatに変換
    bull_signals = pd.Series([x.iloc[0] for x in results[bull_mask]["forward_return"].dropna().map(pd.Series)])
    bear_signals = pd.Series([x.iloc[0] for x in results[bear_mask]["forward_return"].dropna().map(pd.Series)])
    no_signals = pd.Series([x.iloc[0] for x in results[no_signal_mask]["forward_return"].dropna().map(pd.Series)])
    
    print("=== バックテスト結果 ===")
    print("\n強気シグナル発生回数:", len(bull_signals))
    if len(bull_signals) > 0:
        mean_return = bull_signals.mean()
        win_rate = (bull_signals > 0).sum() / len(bull_signals) * 100
        print(f"平均リターン: {mean_return:.2f}%")
        print(f"勝率: {win_rate:.1f}%")
    
    print("\n弱気シグナル発生回数:", len(bear_signals))
    if len(bear_signals) > 0:
        mean_return = bear_signals.mean()
        win_rate = (bear_signals < 0).sum() / len(bear_signals) * 100
        print(f"平均リターン: {mean_return:.2f}%")
        print(f"勝率: {win_rate:.1f}%")
    
    print("\nシグナルなしの期間:", len(no_signals))
    if len(no_signals) > 0:
        mean_return = no_signals.mean()
        print(f"平均リターン: {mean_return:.2f}%")

def run_simulation(params, stock_df, btc_df, futures_data, reddit_sentiment):
    """単一シミュレーションを実行（グローバルスコープに移動）"""
    bull_conditions = {
        "return_threshold": params["bull_return"],
        "consec_days": int(params["bull_consec"]),
        "sentiment_threshold": params["bull_sentiment"],
        "futures_consensus": params["bull_futures"]
    }
    
    bear_conditions = {
        "return_threshold": params["bear_return"],
        "consec_days": int(params["bear_consec"]),
        "sentiment_threshold": params["bear_sentiment"],
        "futures_consensus": params["bear_futures"],
        "drawdown_threshold": params["bear_drawdown"]
    }

    bull_threshold = params["bull_threshold"]
    bear_threshold = params["bear_threshold"]

    # シグナル検出を実行
    signals = detect_market_signals_with_params(
        stock_df, btc_df, futures_data, reddit_sentiment,
        bull_conditions, bear_conditions,
        bull_threshold, bear_threshold
    )

    # バックテスト結果を取得
    backtest_results = backtest_signals("^GSPC", "2015-01-01")
    
    # 評価指標を計算
    bull_score = evaluate_signals(backtest_results, "bull")
    bear_score = evaluate_signals(backtest_results, "bear")
    
    return {
        "params": params,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "total_score": bull_score + bear_score
    }

def generate_random_params():
    """ランダムなパラメータセットを生成（グローバルスコープに移動）"""
    return {
        # 強気パラメータ
        "bull_return": random.uniform(1.5, 3.0),
        "bull_consec": random.randint(2, 5),
        "bull_sentiment": random.uniform(0.5, 0.8),
        "bull_futures": random.uniform(0.5, 0.8),
        "bull_threshold": random.uniform(2.5, 4.0),
        
        # 弱気パラメータ
        "bear_return": random.uniform(2.0, 3.5),
        "bear_consec": random.randint(3, 6),
        "bear_sentiment": random.uniform(0.6, 0.9),
        "bear_futures": random.uniform(0.6, 0.9),
        "bear_drawdown": random.uniform(-7.0, -3.0),
        "bear_threshold": random.uniform(3.0, 4.5)
    }

def evaluate_signals(results, signal_type):
    """シグナルの性能を評価（NoneTypeのエラー処理を追加）"""
    try:
        if signal_type == "bull":
            mask = results["bull_signal"]
            returns = results[mask]["forward_return"].apply(
                lambda x: float(x.iloc[0]) if isinstance(x, pd.Series) else (
                    float(x) if x is not None else 0.0
                )
            )
            win_rate = (returns > 0).mean() if len(returns) > 0 else 0
            avg_return = returns.mean() if len(returns) > 0 else 0
        else:
            mask = results["bear_signal"]
            returns = results[mask]["forward_return"].apply(
                lambda x: float(x.iloc[0]) if isinstance(x, pd.Series) else (
                    float(x) if x is not None else 0.0
                )
            )
            win_rate = (returns < 0).mean() if len(returns) > 0 else 0
            avg_return = -returns.mean() if len(returns) > 0 else 0

        # スコアの計算（勝率とリターンの組み合わせ）
        score = (win_rate * 0.7 + (avg_return / 2) * 0.3) * 100
        return score
    except Exception as e:
        print(f"評価中にエラーが発生: {str(e)}")
        return 0.0  # エラーの場合は0を返す

def optimize_parameters(stock_df, btc_df, futures_data, reddit_sentiment, n_simulations=1000):
    """モンテカルロシミュレーションでパラメータを最適化（進捗バー付き）"""
    best_score = -float('inf')
    best_params = None
    
    print("\nパラメータ最適化を開始...")
    with ProcessPoolExecutor() as executor:
        # シミュレーション用のパラメータリストを事前に生成
        param_list = [generate_random_params() for _ in range(n_simulations)]
        
        # tqdmで進捗を表示しながら実行
        futures = [
            executor.submit(
                run_simulation,
                params,
                stock_df, btc_df, futures_data, reddit_sentiment
            ) 
            for params in param_list
        ]
        
        # as_completedを使用して完了したものから結果を取得
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=n_simulations,
            desc="シミュレーション進捗",
            ncols=100
        ):
            try:
                result = future.result()
                if result["total_score"] > best_score:
                    best_score = result["total_score"]
                    best_params = result["params"]
            except Exception as e:
                print(f"\nシミュレーション中にエラーが発生: {str(e)}")
                continue

    return best_params, best_score

def detect_market_signals_with_params(stock_df, btc_df, futures_data, reddit_sentiment,
                                    bull_conditions, bear_conditions,
                                    bull_threshold, bear_threshold):
    """
    パラメータ化されたシグナル検出関数
    （既存のdetect_market_signals関数の内容をパラメータ化）
    """
    # 既存の関数の内容をここに移動し、
    # bull_conditions, bear_conditions, bull_threshold, bear_thresholdを使用
    pass

if __name__ == "__main__":
    # 例: S&P 500, BTC-USD, S&P 500 先物
    stock_df = fetch_stock_data("^GSPC")
    btc_df = fetch_bitcoin_data("BTC-USD")
    futures_data = fetch_all_futures_data()

    # Redditデータを取得（複数のサブレディットから）
    reddit_posts = fetch_all_reddit_posts(
        mode="hot",
        limit_per_subreddit=50
    )

    # センチメント集計
    reddit_sentiment = analyze_reddit_sentiment(reddit_posts)

    # 簡易シグナル検出
    signals = detect_market_signals(stock_df, btc_df, futures_data, reddit_sentiment)

    print("=== 市場分析結果 (詳細版) ===")
    
    print("\n--- 株式市場 ---")
    for period, analysis in signals["stock"].items():
        print(f"\n{period}日間の分析:")
        print(f"  トレンド: {analysis['direction']}")
        print(f"  リターン: {analysis['return']:.2f}%")
        print(f"  ボラティリティ: {analysis['volatility']:.2f}%")
        print(f"  最大ドローダウン: {analysis['max_drawdown']:.2f}%")

    print("\n--- 仮想通貨市場 ---")
    for period, analysis in signals["btc"].items():
        print(f"\n{period}日間の分析:")
        print(f"  トレンド: {analysis['direction']}")
        print(f"  リターン: {analysis['return']:.2f}%")
        print(f"  ボラティリティ: {analysis['volatility']:.2f}%")
        print(f"  最大ドローダウン: {analysis['max_drawdown']:.2f}%")

    print("\n--- 先物市場 ---")
    for symbol, data in signals['futures'].items():
        print(f"\n{symbol}:")
        for period, analysis in data.items():
            print(f"  {period}日間の分析:")
            print(f"    トレンド: {analysis['direction']}")
            print(f"    リターン: {analysis['return']:.2f}%")
            print(f"    ボラティリティ: {analysis['volatility']:.2f}%")
            print(f"    最大ドローダウン: {analysis['max_drawdown']:.2f}%")

    if signals["bull_signal"]:
        print("[シグナル] 暴騰の可能性 (強気シグナル検出)")
    elif signals["bear_signal"]:
        print("[シグナル] 暴落の可能性 (弱気シグナル検出)")
    else:
        print("[シグナル] 特に大きなシグナルは検出されませんでした")

    # バックテストの実行
    print("\n=== バックテストの実行 ===")
    backtest_results = backtest_signals("^GSPC", "2022-01-01")
    analyze_backtest_results(backtest_results)

    # パラメータの最適化を実行
    print("\n=== パラメータ最適化の実行 ===")
    with tqdm(total=3, desc="データ準備", ncols=100) as pbar:
        # データの準備
        stock_df = fetch_stock_data("^GSPC")
        pbar.update(1)
        
        btc_df = fetch_bitcoin_data("BTC-USD")
        pbar.update(1)
        
        futures_data = fetch_all_futures_data()
        pbar.update(1)

    # Redditデータを取得
    print("\nRedditデータの取得中...")
    reddit_posts = fetch_all_reddit_posts(
        mode="hot",
        limit_per_subreddit=50
    )

    # センチメント集計
    print("\nセンチメント分析中...")
    reddit_sentiment = analyze_reddit_sentiment(reddit_posts)

    # 最適化の実行
    best_params, best_score = optimize_parameters(
        stock_df, btc_df, futures_data, reddit_sentiment,
        n_simulations=3000
    )
    
    print("\n=== 最適化結果 ===")
    print("\n最適なパラメータ:")
    for key, value in best_params.items():
        print(f"{key}: {value:.3f}")
    print(f"\n総合スコア: {best_score:.2f}")

