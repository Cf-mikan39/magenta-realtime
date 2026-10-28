# Projection実モデル検証（AGENTS5、2026-10-27）

この記録は実モデル・キャッシュ・MLP・W&B・RVQの接続確認です。
対応CSVが未用意のため、既存画像と音楽をファイル名順で組み合わせた20組を使いました。
**画像と音楽の意味的対応は未確認であり、学習・Retrieval・RVQの品質評価に使えません。**
データセットはダウンロードしていません。実験出力はGit管理外の
`outputs/projection/agents5_validation/`に保存しています。

## 専用環境とモデル

MRT2 `.venv`は変更せず、外側の`.venv-projection`内にモデル依存を揃えました。
Python 3.9.10 / PyTorch 2.1.0+cu118 / torchvision 0.16.0+cu118 /
torchaudio 2.1.0+cu118 / transformers 4.40.2 / timm 0.9.16 / W&B 0.19.11です。
依存の全一覧は`environment.freeze.txt`にあります。

- CLIP: `models/CLIP/clip-vit-base-patch32/`、公式HF revision
  `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。重み605,247,071 bytesを取得・照合済み。
- ImageBind: ユーザー提供`models/ImageBind/imagebind_huge.pth`、4,803,584,173 bytes。
  SHA-256: `d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157`。
- 公式ImageBind source revision: `53680b02d7e37b19b124fa37bae4b6c98c38f5be`。
- 公式指定PyTorchVideo revision: `6cdc929315aab1b5674b6dcf73b16ec99147735f`。

`raw.json`と`l2.json`の`models`からモデルパスを指定できます。
モデルの自動ダウンロードはCLIには追加していません。

## 画像特徴の実測

GPUはRTX 3090（24 GiB）、5画像、それぞれbatch=1で3回です。
時間には画像の読込・前処理・GPU推論・CPU転送を含み、GPU同期後に計測しています。
warm平均は各画像の2・3回目、計10回の平均です。

| 項目 | CLIP ViT-B/32 | ImageBind Huge |
|---|---:|---:|
| 出力shape（1画像） | (512,) | (1024,) |
| dtype | float32 | float32 |
| ノルム | 約1 | 約1 |
| 画像ごと3回のbitwise一致 | 全5画像で一致 | 全5画像で一致 |
| 異なる画像間cosine範囲 | 0.3300–0.6567 | 0.1696–0.5053 |
| warm平均 | 約23.3 ms | 約46.2 ms |
| モデルロード・資産ハッシュ等 | 約23.6秒 | 約104.4秒 |
| PyTorch peak allocated | 約346.7 MiB | 約4678.1 MiB |

計測値はこの環境・5画像の値です。CUDA context/driverのメモリはallocatorの値に含みません。
初回推論には初期化コストがあり、warm値だけで起動遅延を見積もらないでください。
このCUDA環境ではnvrtcに関するCuDNN workaround警告が出ますが、処理は完了しました。
全時間、cosine行列、画像ハッシュ、再現性、peak reservedも以下に保存しました。

- `clip_benchmark/benchmark.json`、`images.csv`、`features.npy`、`cosine.npy`
- `imagebind_benchmark/benchmark.json`、`images.csv`、`features.npy`、`cosine.npy`

## 実素材による小規模学習

素材は`test_scripts/*.jpg`の既存20枚と、既存
`dataset/Tunefine/selected_copy/*.mp3`の先頭20曲（各冒頭10秒）です。
出力内の`prepare_wiring_pairs.py`、`unverified_pairs.csv`、`data_notice.json`で選び方を記録しました。
実際の対応関係を表すCSVではありません。

分割はSeed 42、独立20組のTrain 16 / Validation 2 / Test 2です。
画像キャッシュはCLIP 512次元・ImageBind 1024次元、実MusicCoCa教師はRaw 768次元です。
Audioキャッシュを両Encoder・Raw/L2で共用し、L2変換は学習時に適用しました。

MLP hidden=64、GELU、dropout=.1、4 Epoch、batch=4、AdamW、LR=.001、CPU、4 threadsです。
RawはMSE=1・cosine=.1、出力正規化なし、L2はcosine=1・出力L2という既存設定例を使いました。
このためRawとL2のLoss数値を直接比較する条件ではありません。

| 条件 | 最初のTrain Loss | 最後のTrain Loss | Best Val Loss | W&B |
|---|---:|---:|---:|---|
| CLIP Raw | 1.111681 | 1.077415 | 1.097851 | 無効 |
| CLIP L2 | 0.975415 | 0.841217 | 0.918599 | offline |
| ImageBind Raw | 1.122906 | 1.080501 | 1.103498 | offline |
| ImageBind L2 | 1.022620 | 0.855075 | 0.942207 | offline |

Best Epochは全条件4でした。この減少は接続確認の結果であり、汎化性能の根拠にはしません。
`run_small_training.py`、`training_validation.json`、各`*_train/`のCSV/JSON・
`best.pt`・`last.pt`・`checkpoints/epoch_*.pt`に記録しました。
各`*_evaluation/`には予測・教師・Mean Teacher Baselineの評価とRetrieval指標があります。
Testは2件だけなのでRecall@5等の数値に研究上の意味を持たせないでください。

## 実W&B offlineと再開

3条件を実W&B 0.19.11でoffline記録しました。CSVのTrain/Val Loss等は、
`wandb_metrics.jsonl`に加えてSDKの`run-*.wandb`を読み戻し、一致を確認しています。
全条件でEpochとSDK Stepは1,2,3,4です。
SDKファイルへの確認方法は`check_offline_records.py`、結果は`offline_record_validation.json`です。
素材・CSV・チェックポイントの添付はなく、configにローカルパスがないことを確認しました。

CLIP L2のEpoch 2から新しい出力先`clip_l2_resumed/`へ再開しました。
offlineでは親Runを参照する別Runになり、記録したEpochは3,4だけです。
最終重みは全tensorで連続学習と一致し、Validation履歴も完全一致しました。
onlineでの同一Run再開、未ログイン・未インストール、通信失敗、無効時にSDKをimportしない動作は
ネットワーク不要のMockテストで確認しました。

`evaluation_offline/`にEmbedding・Retrieval指標、`rvq_offline/`に全12段・先頭6段と
段別Token一致率を、明示的な`log-results`で記録しました。
今回はonline認証・送信・syncを実行していません。オンラインのダッシュボード表示は未検証です。

## 1枚画像推論とRVQ

CLIP RawとImageBind L2の`best.pt`を読み込み直し、実画像`sample01.jpg`を推論しました。
どちらも768次元float32を出力し、実MusicCoCa RVQから12段のTokenを取得しました。
`clip_infer/`、`imagebind_infer/`、対応する`*_infer_rvq/tokens.json`が結果です。
CLIP RawのTest 2件では予測・教師・平均BaselineのRVQ比較も`clip_raw_rvq/`へ保存しました。
ここでのToken生成成功は、リアルタイム音楽の聴取品質を保証するものではありません。

## 回帰検証と未検証項目

Projection 15件、W&B 9件、Audioキャッシュ1件、既存Audio基準実験13件、計38件が通りました。
W&B失敗後のローカル保存、Raw/L2設定、監視ON/OFFによる重みの一致、再開、任意Artifactの
内容制限もテストしています。既存MRT2、VLM、Web UI、Audio基準実験コードは変更していません。

今後は対応品質を確認したCSVを用意し、同じ分割・Loss条件で学習と独立評価を行う必要があります。
大規模学習、GPU MLP学習、online実送信、実アカウントでのArtifact送信、
音楽の聴取評価とWeb UI統合は今回の対象外です。
