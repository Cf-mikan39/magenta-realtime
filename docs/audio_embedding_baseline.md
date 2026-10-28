# MusicCoCa Audio Embeddingの基準実験

`scripts/musiccoca_audio_baseline.py`は、AGENTS3.mdの基準実験用コードです。
既存Webサーバーとは通信せず、MRT2、VLM、UI、モデルの重みを変更しません。
Projectionの実装・学習、CLIP/ImageBindの導入は含みません。
モデル・パッケージのダウンロード処理もありません。

## 実行

以下は`magenta_rt2/magenta-realtime`で実行します。
`/path/to/...`は説明用です。実在する入力と、新規の出力先に置き換えてください。
入力音楽はまだ指定されていないため、実音楽の評価結果はありません。

```bash
cd /disk/fs1/bigtmp/kai.ogata.r5/current/magenta_rt2/magenta-realtime

# ディレクトリ内の音声を再帰探索。既定では最大10ファイル。
../.venv/bin/python -m scripts.musiccoca_audio_baseline \
  --audio-dir /path/to/music \
  --output-dir outputs/audio_baseline/run01

# 複数ファイルの直接指定。最初の60秒だけ処理する例。
../.venv/bin/python -m scripts.musiccoca_audio_baseline \
  /path/to/music/a.wav /path/to/music/b.flac \
  --duration-seconds 60 \
  --ratio 1:1 --ratio 2:1 --ratio 1:2 \
  --final-normalization both \
  --output-dir outputs/audio_baseline/run02

# 改行区切りの一覧。相対パスは一覧ファイルのある場所を基準に解釈。
../.venv/bin/python -m scripts.musiccoca_audio_baseline \
  --file-list /path/to/music_files.txt \
  --output-dir outputs/audio_baseline/run03
```

出力先が既に存在する場合は、空でもエラーにします。既存結果を上書きしません。
入力なし、存在しないファイル、空区間などを成功扱いしません。
ファイル単位のエラーは`errors.json`に残し、他のファイルを続行します。
一部失敗時は終了コード1と`manifest.json`の`partial`、全件失敗時は`failed`で通知します。

主なオプション:

| オプション | 意味 |
|---|---|
| `--max-files 5` | 最大5曲。既定10、0は上限なし |
| `--start-seconds 30 --duration-seconds 60` | 30秒地点から最大60秒。省略時は全曲 |
| `--hop-seconds 10` | 10秒窓のhop。窓長自体は既存MusicCoCaの10秒 |
| `--no-pad-end` | 不完全な末尾窓を除外。既定はゼロ埋めして利用 |
| `--prompt 'calm ambient'` | Text指定。繰り返し可能。既定の3プロンプトを置換 |
| `--no-text` | Audio/RVQだけ実施。Text Encoder／Mapperをロードしない |
| `--ratio 2:1` | Text:Audio。繰り返し可能。既定は1:1、2:1 |
| `--final-normalization both` | 合成後の正規化なし／L2を独立比較。既定`none` |
| `--mix-windows` | 曲平均に加えて窓ごとのText合成も比較 |
| `--mapper-seed 0` | Text→Audio MapperのノイズSeed。既定0 |
| `--resource-dir /existing/musiccoca` | ローカルの既存モデル資産を指定 |

ディレクトリ探索はwav/flac/ogg/aif/aiff/mp3が対象です。実際のデコード可否は
既存の`soundfile`環境に依存します。直接指定・一覧指定では拡張子で制限しません。
パスの重複を除去し、採用順は`settings.json`／`manifest.json`へ保存します。

## 処理と計測

音声読込は`Waveform.from_file()`、Embeddingは次の既存APIを使います。

```python
windows = model.embed(
    waveform, pool_across_time=False,
    hop_length=10.0, pad_end=True, mono_strategy="average")
pooled = np.mean(windows, axis=0)
```

この平均はMusicCoCaの`pool_across_time=True`と同じ処理です。
Encoderの二重実行を避けています。窓を正規化してから平均する処理には置き換えません。
リサンプリング、モノラル化、スペクトログラム化は既存API内で実施します。
ゼロ埋めした末尾窓も、既存仕様と同じく他の窓と等しい重みで平均されます。

すべての窓と曲平均について、raw/L2のEmbeddingと12段RVQを保存します。
shapeは単一ベクトル`(768,)`、窓列`(N,768)`、dtypeは`float32`です。
窓ごとの時間区間、末尾のゼロ埋め秒数、曲平均とのcosineも記録します。
ノルム0のL2正規化は未定義なのでエラーにし、cosineの分母が0ならJSONの`null`とします。

Textは既存Web版と同じ`embed(text, use_mapper=True, seed=...)`で取得します。
既定プロンプトは`lofi, soft piano`、`energetic rock`、`calm ambient`です。
Textに含まれるカンマは1プロンプト内の文字列です。既存VLMのタグ別分割とは別の
固定Text基準条件として扱います。

合成には既存の`blend_style_embeddings()`を利用します。
`T`をMapper後Text、`A`を生Audio、比率を`a:b`とした場合、比較条件は以下です。

```text
生Audio合成:       Z = (a*T + b*A) / (a+b)
単位Audio合成:     Z = (a*T + b*normalize(A)) / (a+b)
合成後L2の追加条件: normalize(Z)
```

合成前の入力ノルム、係数を掛けたノルム、合成後ノルム、各入力とのcosineを記録します。
係数を掛けたノルムはベクトル長の寄与の指標で、知覚上の音楽の寄与率ではありません。
Text合成は既定では曲平均のみです。窓ごとのraw/L2のRVQ比較は常に行います。

計時はファイル読込、Audio Embedding、Text Embedding、各RVQを分けます。
Audio計時は内部のモノラル化・リサンプリング・前処理・全窓Encoderを含みます。
初回はlazy loadも含むcold timingで、ウォームアップ後の純粋推論時間とは異なります。
モデルのハッシュ計算はEmbeddingの計時区間に含めません。

## 出力

| ファイル | 内容 |
|---|---|
| `settings.json` | 開始時設定、入力順、モデル資産のパス・サイズ・SHA-256、バージョン |
| `manifest.json` | 上記に終了状態、件数、所要時間を追加 |
| `sources.csv` | 入力パス・SHA-256、長さ、SR、チャンネル、使用区間、取得時間、解析完了状態 |
| `embeddings/a001_*.windows_raw.npy` | 全窓の生Audio教師、`(N,768)` |
| `embeddings/a001_*.pooled_raw.npy` | 生Audio教師の単純平均、`(768,)` |
| `embeddings/*_l2.npy` | 対応するL2版。生教師を置き換えず別保存 |
| `embeddings/t*.npy` | Mapper後の固定Text条件 |
| `embeddings/*.mix_*.npy` | 合成後の連続ベクトル |
| `conditions.csv` / `.json` | 各条件のEmbedding参照、ノルム、統計、12トークン、合成情報 |
| `token_comparisons.csv` | raw対L2、比率変更、合成後L2の一致率・変化位置 |
| `dimension_stats.csv` | 曲平均／窓／Textの各グループで次元別mean・std。stdはddof=0 |
| `pooled_cosine.csv` / `.npy` | 曲間cosine。npyの行列順はsources.csv順 |
| `token_diversity.csv` | 条件グループ・段ごとの頻度、ユニーク数、entropy(bits)、12段／先頭6段の系列種類数 |
| `errors.json` | デコード等の失敗。成功時は空配列 |
| `generations.csv`, `generated/*.wav` | 明示的に生成した場合のみ |

CSVのリスト・辞書セルはJSON文字列です。窓ごとの`embedding_file`は窓列npyを指し、
`embedding_row`が0始まりの行番号です。曲平均／合成ベクトルでは行番号はnullです。
トークンの変化位置は1始まりです。

一致率は12段全体、先頭6段、設定に応じた有効段を別々に記録します。
既定の末尾6段マスクでは先頭6段が有効です。全12段マスク時の有効段一致率はnullです。
`generation_tokens_before_offset`はマスク適用後・内部の+7オフセット適用前です。
Token分布は曲平均と窓を分離し、窓数が多い曲が曲平均条件の統計を支配しないようにします。
少数曲のentropyを一般的な品質指標としては扱わないでください。
空の集計（1曲だけの曲間cosineなど）のCSVは作成しません。

## 任意のMRT2生成

通常実行ではJAXをインポートしません。`--generate`を付けた場合だけ、
既存ローカルチェックポイントから別のMRT2インスタンスをロードします。
GPUプロセスが動作中、または稼働状況を確認できない場合は生成を開始せず、
Embedding/RVQ結果を保持して終了コード1にします。サーバーを停止する処理はありません。
実行中サーバーと同じGPUでの同時生成は行わず、空いている時間帯に実行してください。

```bash
# 1曲の曲平均raw/L2だけを各5秒生成。GPUが空いているときに明示実行。
../.venv/bin/python -m scripts.musiccoca_audio_baseline \
  /path/to/music/a.wav --duration-seconds 30 \
  --generate --generate-seconds 5 \
  --generate-condition '*.pooled.raw' \
  --generate-condition '*.pooled.l2' \
  --output-dir outputs/audio_baseline/generation01
```

Text合成も選ぶには、例えば`--generate-condition '*.pooled.mix_t001.r00.*'`を追加します。
フィルタ省略時は**全曲の曲平均raw/L2と全Text合成条件**を生成するため、
最初は1曲と条件フィルタで絞ってください。

既存API`mrt.generate(style=vector)`は同じRVQを実行します。
その直前に`tokenize_style(vector)`と保存済み12トークンの一致を確認します。
離散トークンを連続Embeddingの引数へ渡すような変更や、生成処理の差し替えは行いません。

全条件を`state=None`で開始し、MIDI全128音とDrumsをマスクします。
既定はWeb版と同じCFG(1.6/2.4/4.0)、temperature=1.1、top-k=50、
末尾6段マスク、num_cfgs=2、float32デコーダー出力です。
生成秒数は40msフレーム単位で切り上げ、実際の長さも保存します。
WAVは条件ID別の48kHz stereo float形式で保存し、条件別の音量正規化は行いません。

**Seedの制約:** 現在のMRT2 `_compile()`は内部random Seed=0、params Seed=42が固定です。
各条件で初期状態を作り直すことで同じ乱数条件にします。`--mapper-seed`はText Mapper用であり、
MRT2生成Seedを変更するオプションではありません。両者を別々に記録します。
モデルのコンパイル時間、条件別生成時間、チェックポイントSHA-256も保存します。
GPUメモリの事前確保はこの実験プロセス内で無効にします。

## 次のProjection学習への判断材料

現時点では実音楽による比較結果がないため、以下は実験前の設計方針です。

1. **生の窓Embeddingを必ず保存する。** 正規化・pooling・Lossの方針を後から変更できます。
   画像と曲全体の対応が妥当なら曲平均も使えますが、時間対応があるデータでは
   対応区間の教師を優先します。曲平均と窓の教師を同一条件として混ぜないでください。
2. **生Audioの再現を目標にする場合:** cosineだけでは長さが決まらないため、
   まずMSEを基準にし、必要に応じてcosine項を加えます。
   `MSE(z,y) + lambda_cos*(1-cos(z,y))`は方向とスケールを同時に拘束します。
   MSEがあればノルムLossは必須ではありません。尺度差が大きい場合は損失値と勾配を見て
   係数を調整し、必要な場合だけ`(norm(z)-norm(y))^2`等のノルム項を追加します。
3. **単位Audioを出力目標にする場合:** `z=normalize(Projection(x))`とし、
   `y_unit=normalize(y)`とのcosine Lossを使えば、推論出力の長さは構造で固定できます。
   単位ベクトル同士の二乗距離はcosineと等価なため、同じ目的でMSEを重ねる必要はありません。
   ゼロ近傍対策と、正規化した教師がraw教師と同じRVQになるとは限らない点に注意します。
4. **通常Textとの合成:** 既存Mapper後Textは単位長なので、単位Audioとの合成は
   係数と入力ベクトル長を分離しやすくなります。ただし音楽的な最適性は未評価です。
   合成後のL2正規化は入力間の相対係数を直す処理ではありません。
   raw Audioをそのまま混ぜた場合のAudio側の長さによる影響は、合成後L2でも残ります。
5. **判断順:** Audio教師そのもののraw/L2生成、Text合成時の聴取比較、
   RVQの有効段の変化を確認して、教師ターゲットと推論時のスケール方針を揃えます。
   Tokenが変わることやcosineが高いことだけで生成品質を判断しないでください。

今回のスクリプトはLossやProjectionを実装せず、この判断に必要な基準データだけを作ります。

## コードの検証

```bash
../.venv/bin/python -m unittest discover -s tests \
  -p test_musiccoca_audio_baseline.py -v
```

単体テストでは既知のベクトルを返すテスト用モデルを使い、教師の保持、合成、
統計、トークン比較、上書き拒否、エラー記録、生成条件のリセットを確認します。
これは実MusicCoCaによる音楽評価やGPU生成の成功を意味しません。

実装時には13件の単体テストに加え、既存MusicCoCaをCPUで使用し、
21秒・16kHzのモノラル正弦波と12秒・44.1kHzのステレオ正弦波で接続を確認しました。
71条件、91組のToken比較を保存でき、JAXがロードされないことも確認しました。
曲平均相当のrawベクトルのノルムは約27.19／26.32、raw対L2の先頭6段一致率は
約16.7%／50.0%でした。これは**合成波形による実装確認**であり、実音楽の評価結果ではありません。
実音楽の分布・聴取評価と、実MRT2によるGPU生成は未実行です。
