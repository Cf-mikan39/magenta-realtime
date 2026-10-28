# CLIP / ImageBind → MusicCoCa Audio Projection

`AGENTS4.md`の方式C用オフラインパイプラインです。
学習するのは小規模MLPだけで、Image Encoder・MusicCoCa・MRT2は更新しません。
既存VLM、Web UI、連続生成、前回の基準実験スクリプトは変更していません。
MUImage等のアクセス権・ダウンロード・固有スキーマには依存しません。

## 構成と環境

| ファイル | 役割 |
|---|---|
| `projection/dataset.py` | CSV、ID、区間、リーク防止分割 |
| `projection/encoders.py` | 固定CLIP / 公式ImageBind、ローカル重み限定 |
| `projection/cache.py` | 画像特徴とRaw Audio教師の重複排除・保存・検証 |
| `projection/model.py` | 2層MLP、学習出力の正規化、Loss |
| `projection/config.py`、`configs/*.json` | 設定検証、Raw/L2の設定例 |
| `projection/training.py` | 学習、Validation、平均教師Baseline、保存、再開 |
| `projection/tracking.py` | 任意のW&B監視、Run再開、数値評価の明示的出力 |
| `projection/benchmark.py` | 実画像Encoderの再現性・時間・VRAM計測 |
| `projection/inference.py` | Webに依存しない画像／特徴ベクトル推論API |
| `projection/evaluation.py` | Embedding・Retrieval・平均Baseline評価 |
| `projection/rvq.py` | 既存MRT2環境でのRVQ後処理。PyTorch不要 |
| `projection/synthetic.py` | 人工特徴だけのCPU接続確認 |
| `projection/__main__.py` | 各工程のCLI。必要な依存だけを遅延import |

全コマンドは`magenta_rt2/magenta-realtime`から実行します。

```bash
MRT_PY=../.venv/bin/python
PROJ_PY=../.venv-projection/bin/python
```

`cache-audio`と`rvq`は既存のMRT2環境で実行します。
画像特徴抽出、学習、推論、Embedding評価は独立したPyTorch環境で実行します。
学習時に画像Encoder、MusicCoCa、MRT2をロードしません。
`projection`のimport、CLIのhelp、Audio/RVQ工程ではPyTorch/JAXをimportしません。

外側の`.venv-projection`はPython 3.9.10、PyTorch 2.1.0+cu118、
torchvision 0.16.0+cu118、torchaudio 2.1.0+cu118、NumPy 1.26.4、Pillow 10.4.0です。
AGENTS5対応で、以前の別環境への`.pth`参照を無効化し、PyTorchと必要な依存を
専用環境内へコピーしました。出所は`.venv-projection/local_torch_provenance.json`に記録しました。
既存MRT2環境も既存musicgen環境も変更していません。
transformers 4.40.2、timm 0.9.16、W&B 0.19.11、公式ImageBindも導入済みです。

別マシンに新規環境を構成する場合:

```bash
# インストール済みPythonを指定。MRT2の .venv は対象にしない。
uv venv --python /path/to/python3.9 --no-python-downloads ../.venv-projection
uv pip install --python "$PROJ_PY" -r projection/requirements-models.txt \
  --extra-index-url https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match
git clone https://github.com/facebookresearch/ImageBind.git ../models/ImageBind/source
git -C ../models/ImageBind/source checkout 53680b02d7e37b19b124fa37bae4b6c98c38f5be
uv pip install --python "$PROJ_PY" --no-deps \
  'git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f' \
  ../models/ImageBind/source
```

ImageBindは[公式Repository](https://github.com/facebookresearch/ImageBind)の
セットアップ手順で別途導入し、Git revisionと依存関係を記録してください。
torchvision / torchaudioは使用するPyTorchに合うビルドを選びます。
公式ソースのこのrevisionは`timm.layers`を使用するため、古いtimm 0.6系は使いません。
この依存セットはCUDA 11.8用です。異なるCUDA/CPUビルドを選ぶ場合は別環境で検証してください。

CPUが既定です。`--device cuda:0`を明示した場合は、他プロセスがGPU使用中なら開始しません。
プロセスを終了させたり、リアルタイムサーバーの状態を変えたりする処理はありません。
GPUの空き確認と開始は不可分ではないため、他のジョブの起動と重ならない時間帯で実行してください。

## 対応する画像モデル

| | CLIP | ImageBind |
|---|---|---|
| モデル | `openai/clip-vit-base-patch32` | 公式`imagebind_huge` |
| 特徴 | vision projection後512次元をL2正規化 | 公式vision出力1024次元、L2 |
| 入力 | RGB、224px、resize/center crop | RGB、短辺224px、bicubic resize、224px center crop |
| 正規化 | ローカル`CLIPImageProcessor`設定を使用 | mean=(.48145466,.4578275,.40821073)、std=(.26862954,.26130258,.27577711) |
| 実装 | `transformers.CLIPVisionModelWithProjection` | `imagebind.models.imagebind_model`と公式vision前処理 |
| ローカル重み | HFスナップショットのディレクトリ | `imagebind_huge.pth` |
| 重み容量 | 公式`pytorch_model.bin`は605,247,071 bytes（約605 MB） | 公式pthは4,803,584,173 bytes（約4.80 GB） |

容量は重み単体です。実行時の計測方法は後述の`benchmark-encoder`、実測値は
[実モデル検証記録](projection_real_model_validation.md)を参照してください。
配布先: [CLIP](https://huggingface.co/openai/clip-vit-base-patch32/tree/main)、
[ImageBind](https://github.com/facebookresearch/ImageBind#imagebind-model)。
[CLIP API](https://huggingface.co/docs/transformers/model_doc/clip)、
[ImageBind前処理](https://github.com/facebookresearch/ImageBind/blob/main/imagebind/data.py)も参照してください。

実験コードにダウンロード機能はありません。ユーザーが別途用意した重みだけを指定します。
CLIPは`local_files_only=True`、ImageBindは`pretrained=False`で構築してローカルstate_dictを読み込みます。
両モデルとも`eval()`、パラメータ固定、`inference_mode()`で抽出します。
重みSHA-256、前処理、特徴次元、実装・パッケージ版をキャッシュとチェックポイントへ保存します。
ImageBindはモデル・前処理ソースのSHA-256も記録します。

今回のローカル設定は`projection/configs/raw.json`と`l2.json`の`models`で指定できます。
`models.clip`は`models/CLIP/clip-vit-base-patch32`、`models.imagebind`は
ユーザー提供の`models/ImageBind/imagebind_huge.pth`を指します。
JSON内のモデルパスは**設定ファイルのディレクトリ基準**、CLIの`--weights`は作業ディレクトリ基準です。
`models.musiccoca`を指定する場合も同様です。未指定なら既存MRT2の資産配置を使います。
CLIPは公式HF revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`の
`config.json`、`preprocessor_config.json`、`pytorch_model.bin`だけを取得・ハッシュ照合しました。
取得履歴はモデルディレクトリの`download_manifest.json`にあります。

```bash
"$PROJ_PY" -m projection benchmark-encoder --image-encoder clip \
  --config projection/configs/raw.json --image-dir ../test_scripts \
  --max-images 5 --repeats 3 --device cuda:0 --output-dir outputs/projection/clip_probe01
# ImageBindは --image-encoder imagebind に置換して別の出力先を指定。
```

`benchmark.json`にshape/dtype/norm、画像間cosine行列、同一画像の繰り返し誤差、
各回の時間、モデルロード時間、PyTorch allocatorのピークVRAMを記録します。
初回推論と以後の時間を分けて保持します。VRAMはCUDAコンテキスト等を含む総使用量ではありません。

CLIPのバックボーン形状はViT-B/32に限定して検証します。
前処理ファイルの値は記録された設定を使い、勝手な画像拡張は加えません。
CLIPとImageBindで入力次元だけが異なる共通MLPを使用します。

## CSVとデータ分割

```csv
image_id,image_path,audio_id,audio_path,split,start_seconds,duration_seconds,group_id,pooling,window_index
img001,images/001.jpg,mus001,audio/001.wav,train,0,30,original_song001,mean,
img002,images/002.jpg,mus002,audio/002.wav,val,10,20,original_song002,window,0
img003,images/003.jpg,mus003,audio/003.wav,test,0,,original_song003,mean,
```

必須列は`image_id,image_path,audio_id,audio_path`です。
パスはCSVのあるディレクトリ基準です。`pair_id`省略時は行順で生成します。
同じ画像ID・音声IDは複数行に登場できますが、一つのIDは同じファイルを指す必要があります。
`start_seconds`は既定0、`duration_seconds`空欄は残り全部を使用します。
`pooling`、`window_index`空欄はAudioキャッシュ作成時のオプションを使用します。

以下のいずれかを共有する行を、推移的につながるグループとして扱います。

- 同じ画像ID、画像パス、または画像のバイト内容
- 同じ音声ID、音声パス、または音声のバイト内容
- `group_id`（`source_id`も別名として受付）、`image_source_id`、`audio_source_id`

同じ楽曲・元動画の別ファイルへの切り出しは、バイト内容だけでは検出できません。
必ず元楽曲／動画を表すsource/group IDを付けてください。
画像の変形コピーなども同様に元画像のsource IDで結びます。
明示splitがグループ内で衝突した場合はエラーにします。
一部のsplitだけ指定されているグループはその指定を引き継ぎ、未指定グループをSeed付きで分割します。
全未指定時は既定80/10/10の**グループ数**を目安に分割し、可能なら3分割を確保します。
グループサイズが不均等なら行数比率は一致しません。
学習には独立したtrain/val、Test評価にはtestも必要です。

分割を先に固定し、CLIP/ImageBindで同じCSVを使用してください。

```bash
"$MRT_PY" -m projection split \
  --csv /path/to/pairs.csv --seed 42 --output-dir outputs/projection/split01
```

元画像・音声が両方読める状態で実行します。出力`pairs.csv`は絶対パスと確定splitを保持します。
ImageBindで選定されたArt2Mus等を使う場合、ペア選択がImageBindに有利である可能性を記録し、
CLIPとの比較でデータ選択バイアスとEncoderの効果を混同しないでください。

## 事前計算とキャッシュ

```bash
"$PROJ_PY" -m projection cache-images \
  --csv outputs/projection/split01/pairs.csv \
  --image-encoder clip --weights /path/to/local/clip-vit-base-patch32 \
  --batch-size 16 --device cpu --output-dir outputs/projection/clip_cache01

"$PROJ_PY" -m projection cache-images \
  --csv outputs/projection/split01/pairs.csv \
  --image-encoder imagebind --weights /path/to/imagebind_huge.pth \
  --batch-size 4 --device cpu --output-dir outputs/projection/imagebind_cache01

"$MRT_PY" -m projection cache-audio \
  --csv outputs/projection/split01/pairs.csv \
  --pooling mean --hop-seconds 10 \
  --output-dir outputs/projection/audio_cache01
```

Audioは既存`Waveform.from_file()`と`MusicCoCa.embed(pool_across_time=False)`を使います。
内部のモノラル化、16kHz化、10秒窓、前処理を変更しません。
Raw窓列`(N,768)`と、その単純平均`(768,)`をfloat32で保存します。
末尾窓は既定ゼロ埋め・平均に同じ重みで含めます。除外する場合は`--no-pad-end`です。
L2版は学習時にRawから計算し、保存したRaw教師を変更しません。
窓教師は`--pooling window --window-index 0`またはCSVの行別指定で選びます。
対応時刻は区間の`start_seconds`と窓index×hopで定まり、選択をキャッシュに保持します。

画像は同じバイト内容、音声は同じバイト内容と指定区間の組を一度だけエンコードします。
同じ区間の平均と窓を別ペアが使っても、Encoderは再実行しません。
完全に同じ時間範囲でも指定値の表現が異なる場合や、重なる別区間は別キャッシュです。

各キャッシュには`manifest.json`、確定`pairs.csv`、`features/*.npy`があります。
元ID・パス、区間、pooling、モデル資産SHA-256、前処理、shape/dtype、schema版、
ペア集合とキャッシュのfingerprintを保存します。
読み込み時にfingerprintと各配列のSHA-256を確認し、異なるペア・split・モデルの混用を拒否します。
キャッシュは追記・上書きしません。不完全な作成で`manifest.json`がないものは利用できません。
別実験の教師窓・hop・poolingは別キャッシュとして管理してください。

## MLP、Raw/L2、Loss

```text
512D(CLIP) / 1024D(ImageBind)
  → Linear(hidden_dim) → [optional LayerNorm] → GELU/ReLU/SiLU
  → Dropout → Linear(768D)
```

既定hidden=1024、GELU、dropout=.1です。最適設定を意味しません。
Rawの例は`projection/configs/raw.json`、L2の例は`projection/configs/l2.json`です。

| 設定 | Raw既定 | L2既定 |
|---|---|---|
| `target_normalization` | `raw` | `l2` |
| `training_output_normalization` | `none` | `l2` |
| MSE係数 | 1.0 | 0.0 |
| Cosine Loss係数 | 0.1 | 1.0 |
| ノルム二乗誤差係数 | 0.0 | 0.0 |

Lossはサンプルごとに次を計算し、バッチで平均します。

```text
w_mse * mean_dimension((pred-target)^2)
  + w_cos * (1-cosine(pred,target))
  + w_norm * (||pred||-||target||)^2
```

教師変換、学習出力の正規化、Loss係数は独立して変更できます。
設定ファイルの明示値は、教師条件だけをCLIで変えても自動的には変わりません。
Raw/L2の同一Loss比較では、3つのLoss係数と学習出力正規化も明示してください。

RawではMSEがスケールも拘束するため、ノルム項は必須ではありません。
L2で予測出力を構造的に正規化する場合はcosineだけでも長さが規定されます。
L2教師の出力へ後から名前だけ`raw`を付ける処理や、元のノルムを復元する処理はありません。
ゼロ近傍の学習正規化にはepsを使い、推論・評価でゼロに近いベクトルを正規化する場合はエラーにします。

## 学習・Validation・再開

```bash
"$PROJ_PY" -m projection train \
  --image-cache outputs/projection/clip_cache01 \
  --audio-cache outputs/projection/audio_cache01 \
  --config projection/configs/raw.json \
  --epochs 50 --batch-size 64 --learning-rate 0.0001 --seed 42 \
  --device cpu --output-dir outputs/projection/clip_raw01

"$PROJ_PY" -m projection train \
  --image-cache outputs/projection/clip_cache01 \
  --audio-cache outputs/projection/audio_cache01 \
  --config projection/configs/l2.json \
  --output-dir outputs/projection/clip_l2_01
```

同じAudioキャッシュと分割でImageBindキャッシュに置換すればEncoder比較ができます。

## W&Bの追加監視

既定は`wandb.enabled=false`、`log_model=false`です。無効時はW&Bをimportせず、
ログイン・ネットワーク接続は不要です。ローカルCSV/JSONと全チェックポイントは常に保存します。
`requirements-models.txt`にW&Bを含めていますが、W&Bだけ追加するなら専用環境に
`uv pip install --python "$PROJ_PY" wandb==0.19.11`を実行します。

Raw/L2の既存JSONにある以下の項目を編集すれば有効になります。

```json
"wandb": {
  "enabled": true,
  "project": "mrt2-image-audio-projection",
  "entity": null,
  "name": "clip_raw_trial01",
  "group": "paired-data-v1",
  "tags": ["trial01"],
  "mode": "offline",
  "log_model": false,
  "resume": "auto"
}
```

`entity`はTeam/User名で、nullなら標準認証の既定値です。
`name`をnullにするとEncoder・Raw/L2・Run IDで命名し、`group`をnullにすると
データ集合fingerprintから比較グループを作ります。Encoder名とRaw/L2は自動でtagsに加えます。
モデル構造、Loss係数、学習条件、分割件数、キャッシュfingerprint、モデル版・重みSHAも記録します。
ローカルモデルパス、素材のパスやID、画像・音声・データセット本体、APIキーは送信しません。
ユーザーが設定するname/group/tagsには公開してよい実験名を指定してください。

CLIの例（JSONの設定より優先）:

```bash
"$PROJ_PY" -m projection train --image-cache outputs/projection/clip_cache01 \
  --audio-cache outputs/projection/audio_cache01 --config projection/configs/raw.json \
  --wandb --wandb-mode offline --wandb-group paired-data-v1 \
  --wandb-name clip_raw_trial01 --output-dir outputs/projection/clip_raw_wandb01
```

`--no-wandb`で強制無効にできます。`--wandb-project/entity/name/group/tag`、
`--wandb-log-model/--no-wandb-log-model`も使えます。
onlineを使用する際は事前に`../.venv-projection/bin/wandb login`を実行するか、
標準の`WANDB_API_KEY`環境変数を用意して`--wandb-mode online`を指定します。
認証情報をJSONやソースへ記述しないでください。

`train/loss,mse,cosine_loss,norm_squared_error,norm_rmse,cosine_similarity`と
対応する`val/*`、`learning_rate`、`best/val_loss`、`best/epoch`をEpochごとに記録します。
`norm_rmse`は同一Epoch内のノルム二乗誤差平均の平方根です。
W&Bの`epoch`を横軸としてTrainingとValidationを重ねられます。
Runs画面で`image_encoder`・`target_normalization`・`loss`・`projection`を表示し、
同じgroupのCLIP/ImageBind・Raw/L2を絞り込んで比較します。
比較条件はconfigに残りますが、条件の異なる実験を同一条件の優劣として扱わないでください。

未インストール、未ログイン、通信・SDKエラーは`wandb_status.json`に明示します。
W&B失敗後もローカル学習を継続し、終了時の`summary.json`にはローカル学習と
W&Bそれぞれの状態を保存します。監視に失敗した場合のCLI終了コードは2です。
`wandb_metrics.jsonl`はSDKへの記録を試みた数値のローカル控えです。
これはアップロード完了を保証するものではありません。

再開時はCheckpointに保持したRun ID・条件signature・Epochを確認します。
onlineでは同じProject/Entity/条件のRunを`resume=must`で再開し、
リモートStepがCheckpointより進んでいた場合は別Runに分けてEpoch重複を避けます。
`--wandb-resume new`では明示的に別Runにできます。
offlineではSDKが同一Run再開をサポートしないため、元Runを`parent_run_id`で参照する
別Runに残りのEpochだけを記録し、同じgroupにまとめます。
学習自体の再開・乱数復元は従来どおりで、監視のON/OFFが重みを変えないよう乱数を隔離しています。
W&B SDKがofflineのresumeを無視する動作は
[公式SDKソース](https://github.com/wandb/wandb/blob/v0.19.11/wandb/sdk/wandb_init.py)でも確認できます。

offlineのデータは出力先の`wandb/offline-run-*`に保存します。オンライン認証は不要です。
必要になった時だけ、ユーザーの判断で`wandb sync /path/to/offline-run-*`を実行できます。
今回の検証ではsyncもonline送信も実行していません。
`log_model=true`を明示した場合だけ、MLP重み・構造・公開モデル条件のみを含む
`wandb_projection_weights.pt`をArtifactへ追加します。optimizer/RNG、素材パスを含む
既存`best.pt`そのものや、画像・音声をアップロードする処理はありません。

Embedding・Retrieval・RVQの後処理結果は、完了済み学習Runを書き換えず、
明示コマンドで関連付けた別の評価Runに記録できます。

```bash
"$PROJ_PY" -m projection log-results --input-dir outputs/projection/evaluation01 \
  --training-dir outputs/projection/clip_raw01 --config projection/configs/raw.json \
  --wandb-mode offline --output-dir outputs/projection/evaluation_wandb01
# RVQ結果も --input-dir にその出力先を指定して同様に記録。
```

このコマンドは既定offline、有効です。`metrics.json`の有限な数値だけを記録し、
予測ベクトルや素材を読み込んで送信しません。評価指標は`evaluation/*`に入ります。
入力次元はキャッシュから決定し、モデル構成へ保存します。
CLIでは`--hidden-dim`、`--activation`、`--dropout`、`--layer-norm`、
`--optimizer adamw|adam|sgd`、`--weight-decay`、`--patience`等を指定できます。
patience=0はearly stopping無効です。Validation Lossが最小のモデルを選び、Testは使いません。

保存物は`config.json`、`provenance.json`、`initial_validation.json`、
`history.csv/json`、`summary.json`、各epochのチェックポイント、`last.pt`、`best.pt`です。
チェックポイントにはモデル、入力次元、全設定、Encoder/教師の契約、キャッシュfingerprint、
train教師平均Baseline、Seed、epochなどを含みます。
`epoch_*.pt`と`last.pt`はoptimizer・乱数状態・履歴も含み、再開できます。
`best.pt`は推論用で、optimizer再開には使いません。
毎epochを別ファイルにし、既存実験のファイルを上書きしません。

```bash
# 新しい出力先へ再開。epochsは追加回数でなく総epoch数。
"$PROJ_PY" -m projection train \
  --image-cache outputs/projection/clip_cache01 \
  --audio-cache outputs/projection/audio_cache01 \
  --resume outputs/projection/clip_raw01/last.pt --epochs 80 \
  --output-dir outputs/projection/clip_raw01_resume
```

正確なepoch境界再開のため、総epochs以外の設定・キャッシュ・CPU/CUDA種別の変更を拒否します。
CPU検証では中断なし学習と再開学習のパラメータ・Validation履歴を比較します。
異なるGPU、PyTorch版やプラットフォームをまたぐビット単位の再現性は保証しません。
`history.csv`は今回実行したepoch、`history.json`は再開元を含む全履歴です。

## 評価とRVQ

```bash
"$PROJ_PY" -m projection evaluate \
  --checkpoint outputs/projection/clip_raw01/best.pt \
  --image-cache outputs/projection/clip_cache01 \
  --audio-cache outputs/projection/audio_cache01 \
  --split test --output-normalization as_trained \
  --output-dir outputs/projection/eval_raw01

"$MRT_PY" -m projection rvq \
  --input-dir outputs/projection/eval_raw01 \
  --output-dir outputs/projection/rvq_raw01
```

`--split val`でValidationの詳細、`--split train`で学習集合を評価できます。
`--output-normalization l2`は学習済み出力をさらにL2にする独立した推論条件です。
`as_trained`は学習時の出力正規化をそのまま使います。
評価時は教師にも同じ有効出力正規化を適用し、RVQ入力条件を揃えます。
生の教師は別の`raw_teachers.npy`として保持します。
変更後の評価値は元の学習Lossと同一とは限らず、設定を別条件として報告してください。

Embedding評価では次を保存します。

- Cosine、MSE、予測／教師ノルム、絶対ノルム誤差。Rawでも方向のcosineを別計測
- サンプル別CSV、グループごとの各次元mean/variance、平均次元分散
- Train教師だけから得た定数平均Baselineの同じ指標
- 観測された同一画像の複数Audioペアを正解候補として扱うRecall@1 / @5

Retrievalは評価split内の重複を除いたAudio条件をgalleryにし、画像ごとに平均します。
未対応ペアが意味的に不正解とは限らず、galleryが5以下ならRecall@5は自明です。
平均分散が小さいことも画像の時間的安定性の証明にはならず、Collapseを調べる材料です。
Semantic Bottleneck・時間的安定性のVLM比較そのものは今回まだ実施しません。

評価フォルダには`metrics.json`、`samples.csv`、`dimension_stats.csv`、
`predictions.npy`、`teachers.npy`、`raw_teachers.npy`、`baseline.npy`、`evaluation.json`を保存します。

`rvq`工程は同じ量子化モデルに、評価済みの予測・教師・Baselineをそのまま入力します。
追加の正規化はしません。全12段、先頭6段、段別一致率と、頻度・entropy・系列種類数を記録します。
整数Token IDの数値差は距離として使いません。一致率が低いだけで音楽品質低下とは判断しません。
モデル資産とエクスポート配列のSHA-256を検証し、`tokens.json`、`*_tokens.npy`、
`comparisons.csv`、`token_distributions.csv`、`metrics.json`を保存します。

## 任意画像からの推論とAPI

```bash
"$PROJ_PY" -m projection infer \
  --checkpoint outputs/projection/clip_raw01/best.pt \
  --image /path/to/image.jpg --output-normalization as_trained \
  --output-dir outputs/projection/inference01

"$MRT_PY" -m projection rvq \
  --input-dir outputs/projection/inference01 \
  --output-dir outputs/projection/inference01_rvq
```

`prediction.npy`はfloat32 `(768,)`で、`inference.json`にノルム、Encoder、
Projectionチェックポイント、正規化、入力画像、所要時間などを保存します。
実画像重みの配置を移した場合は`--weights /new/local/path`で指定できます。
同一重み・前処理かを照合し、不一致ならエラーにします。

```python
from projection.inference import ImageStylePredictor

predictor = ImageStylePredictor("outputs/projection/clip_raw01/best.pt", device="cpu")
style = predictor.predict_image("image.jpg", output_normalization="as_trained")
# style: np.ndarray, float32, shape=(768,), Audio相当の連続値。Mapperを通さない。
```

特徴量入力の`ProjectionPredictor.predict_features(features, encoder_metadata)`も利用できます。
入力次元だけが合う別Encoderの特徴を誤用しないよう、モデル・前処理の契約を必ず照合します。

将来のWeb統合では`visual_session.py`の画像ワーカーからこのAPIを呼び、
`visual_conditioning.py`の`set_visual(style[None, :])`へ渡せます。
GPU/CPU推論はイベントループ外のワーカーで行い、既存generation/sequence管理で古い結果を破棄します。
`style_encoding.py`は従来のText/VLM用として維持し、画像ベクトルを再度Text Encoder/Mapperへ通しません。
UIの方式選択、タグ表示とベクトル表示、再接続時の画像条件保持、Solo時の扱いは次段階で追加します。
通常Textとの合成では入力ノルムの違いに注意し、今回の推論正規化条件を引き継いでください。

## 重み・実データなしのCPU動作確認

```bash
"$PROJ_PY" -m projection synthetic-cache --output-dir outputs/projection/synthetic01
"$PROJ_PY" -m projection train \
  --image-cache outputs/projection/synthetic01/images \
  --audio-cache outputs/projection/synthetic01/audio \
  --hidden-dim 24 --epochs 8 --batch-size 6 --learning-rate 0.01 \
  --output-dir outputs/projection/synthetic_train01
"$PROJ_PY" -m projection evaluate \
  --checkpoint outputs/projection/synthetic_train01/best.pt \
  --image-cache outputs/projection/synthetic01/images \
  --audio-cache outputs/projection/synthetic01/audio \
  --output-dir outputs/projection/synthetic_eval01
"$PROJ_PY" -m projection infer \
  --checkpoint outputs/projection/synthetic_train01/best.pt \
  --feature-file outputs/projection/synthetic01/example_feature.npy \
  --feature-metadata outputs/projection/synthetic01/example_encoder.json \
  --output-dir outputs/projection/synthetic_infer01
```

すべての出力に`synthetic=true`を保持します。人工特徴はCLIP/ImageBindの特徴ではなく、
教師もMusicCoCaではありません。コードの接続を検証するだけで、実画像・音楽の学習成功を意味しません。
実MusicCoCaの教師キャッシュ・RVQ接続は別工程でCPU検証できます。

```bash
CUDA_VISIBLE_DEVICES='' "$PROJ_PY" -B -m unittest discover -s tests -p test_projection.py -v
CUDA_VISIBLE_DEVICES='' "$PROJ_PY" -B -m unittest discover -s tests -p test_projection_tracking.py -v
"$MRT_PY" -B -m unittest discover -s tests -p test_projection_audio.py -v
"$MRT_PY" -B -m unittest discover -s tests -p test_musiccoca_audio_baseline.py -v
```

AGENTS4時点の検証では、Projection側15件、Audioキャッシュ1件、既存基準実験13件が通りました。
CLIP/ImageBindのアダプターテストはテスト用モデルを使用し、ローカル限定読込・固定・次元・正規化を確認しました。
人工特徴によるCPU学習では、再開後の重みとValidation履歴が、中断しない学習と一致しました。
加えて実MusicCoCaで12秒の合成正弦波3本を教師キャッシュに保存し、
人工特徴で学習した予測・教師を実RVQへ渡せることも確認しました。
Audio/RVQ工程ではPyTorchとJAXがimportされないことを確認しています。
これらは実画像・音楽ペアでの学習成功や聴取品質の検証を意味しません。

## AGENTS5で追加した実モデル・小規模検証

CLIP/ImageBindの実重みを専用環境でロードし、5画像×3回のGPU推論を検証しました。
両方でfloat32・L2ノルム約1、同一画像の繰り返し出力の一致を確認しました。
既存の実画像20枚と実音楽20曲から、両Encoderの画像キャッシュとMusicCoCaのRaw教師を作成し、
CLIP/ImageBind×Raw/L2の4条件、各4 EpochのCPU学習・Validation・評価を実行しました。
ただしユーザーの対応CSVが未用意のため、この20組は**意味的対応を確認していない接続確認専用**です。
Lossの減少、Retrieval、RVQ一致率から画像と音楽の対応品質を結論しないでください。

W&B無効1条件、有効offline3条件と再開1条件を実SDKで確認しました。
offlineの実バイナリ履歴を読み戻し、CSVの全Loss・Epochと一致し、素材添付がないことも確認しています。
Epoch 2からの再開はEpoch 3・4だけを別Runに記録し、連続学習と重み・Validation履歴が一致しました。
両Encoderの1枚画像推論から768次元float32と実RVQの12段Tokenを取得しました。
関連テストは合計38件（Projection 15、W&B 9、Audioキャッシュ1、基準実験13）が通りました。

条件・実測値・出力ファイルと未検証項目は
[実モデル検証記録](projection_real_model_validation.md)を参照してください。
online認証・ダッシュボードへの実送信、GPUでのMLP学習、
本物の対応CSVによる汎化性能・音楽の聴取品質・Web UI統合は今回検証していません。
