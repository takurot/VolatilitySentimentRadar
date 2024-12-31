# VolatilitySentimentRadar

相場の暴騰と暴落を様々なシグナルから予測する Python プログラム

## 概要

このプロジェクトは、以下のデータソースを組み合わせて市場の大きな変動を予測します：

- 株価データ（S&P 500）
- 仮想通貨データ（Bitcoin）
- 先物市場データ
- Reddit の投資関連サブレディットの感情分析

## 主な機能

1. **マルチソースデータ分析**

   - 株式市場のテクニカル分析
   - 仮想通貨市場との相関分析
   - 先物市場のコンセンサス分析
   - ソーシャルメディアの感情分析

2. **シグナル検出**

   - 強気（暴騰）シグナル
   - 弱気（暴落）シグナル
   - 複数の時間枠での分析（3 日〜6 ヶ月）

3. **パフォーマンス最適化**
   - モンテカルロシミュレーションによるパラメータ最適化
   - バックテストによる性能評価
   - 動的な閾値調整

## インストール方法

```bash
git clone https://github.com/yourusername/VolatilitySentimentRadar.git
cd VolatilitySentimentRadar
pip install -r requirements.txt
```

## 必要なライブラリ

- pandas
- numpy
- yfinance
- praw
- textblob
- concurrent.futures

## 使用方法

1. Reddit API の設定

```python
# config.json を作成し、以下の情報を設定
{
    "reddit_client_id": "YOUR_CLIENT_ID",
    "reddit_client_secret": "YOUR_CLIENT_SECRET",
    "reddit_user_agent": "YOUR_USER_AGENT",
    "reddit_username": "YOUR_USERNAME",
    "reddit_password": "YOUR_PASSWORD"
}
```

2. プログラムの実行

```python
python bullbear_radar.py
```

## 分析指標

### 強気シグナル

- 連続上昇日数
- ゴールデンクロス
- 移動平均線の位置関係
- 先物市場の上昇コンセンサス
- ポジティブな市場センチメント

### 弱気シグナル

- 連続下落日数
- デッドクロス
- 最大ドローダウン
- 先物市場の下落コンセンサス
- ネガティブな市場センチメント

## バックテスト結果

- 強気シグナルの勝率: 約 70%
- 弱気シグナルの勝率: 約 35%
- 平均リターン: 強気+0.81%、弱気+1.68%

## 今後の改善点

1. 弱気シグナルの精度向上
2. より多くのデータソースの追加
3. 機械学習モデルの導入
4. リアルタイムアラートシステムの実装

## ライセンス

MIT License

## 注意事項

- このプログラムは投資助言ではありません
- 実際の投資判断は自己責任で行ってください
- 過去のパフォーマンスは将来の結果を保証するものではありません

## 貢献

プルリクエストやイシューの報告を歓迎します。
