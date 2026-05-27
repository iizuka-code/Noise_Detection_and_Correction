# Dust Mask Repair 引き継ぎ書

作成日: 2026-05-21  
対象リポジトリ: `C:\Users\windo\OneDrive\ドキュメント\Codex_projects\ConvertCodex\dust-mask-repair`

更新メモ:

- 2026-05-21: 強補正用 `aggressive` method、JPEG入力対応、ローカルHTMLテストUIを追加。
- 2026-05-23: `NLP_davinci` の `red_highlight_v1` を基に、赤照明検査画像から白黒マスクを生成する最小検出コアとCLIを追加。
- 2026-05-23: ローカルHTMLテストUIに赤照明画像モードを追加し、`/api/repair-red` でマスク生成から補修まで実行できるようにした。
- 2026-05-23: 実画像に近い小型合成fixtureで、赤照明検出、端グロー抑制、生成マスク補修、マスク外不変性の統合回帰テストを追加。
- 2026-05-23: 明示オプション `include_long_scratches` による細長い赤スクラッチ検出を追加。通常モードでは従来通り大きい/長い成分を除外する。
- 2026-05-23: `dust-mask-benchmark` を追加し、赤照明検出からマスク補修までの処理時間・生成マスク画素数・tracemallocピークをJSONで記録できるようにした。
- 2026-05-23: 補修時のデバッグ画像生成を `debug_dir` または `collect_debug_images=True` の時だけに変更し、通常処理とベンチマーク時のメモリ/時間を削減。
- 2026-05-27: 本体統合用に `repair_image_from_red_highlight()` を追加。decode済みRGB/RGBA配列と赤照明RGB配列を渡すだけで、白黒マスク生成から補修まで実行できる。

## 1. このリポジトリの位置づけ

このリポジトリは、写真反転ソフト本体とは別の独立プロジェクトとして作成した「マスク指定型 埃・塵補修エンジン」です。

入力は次の2つです。

1. 通常のフィルムスキャン画像
2. すでに別工程で生成済みの埃位置マスクPNG

このプロジェクトは埃検出を行いません。レーザー照射画像などから埃・塵の位置を赤く浮かび上がらせ、その結果をマスクPNGとして出力する工程は、すでに別システムで完了している前提です。

重要な設計方針は「ノイズリダクションではなく、マスクで指定された局所欠陥のスポット補修」として扱うことです。画像全体にぼかし、ノイズ低減、シャープ化、色補正をかける処理は入れていません。

## 2. 現在のリポジトリ状態

- `dust-mask-repair` ディレクトリ内で `git init` 済みです。
- ベースライン、赤照明検出CLI、赤照明Web UI統合はローカルコミットとして作成済みです。
- 作業前後は `git status --short` で未コミット差分を確認してください。
- 既存の親プロジェクト `ConvertCodex` 側の `README.md` や `AGENTS.md` は変更していません。
- テスト生成物は `test_outputs/` に出ます。`.gitignore` 済みです。
- 初回のpytest実行時に、この環境の一時ディレクトリ権限問題で `.pytest_tmp/` と `pytest-cache-files-*` が生成されました。これらも `.gitignore` 済みです。削除しようとしましたが、Windows側でアクセス拒否されました。

## 3. ファイル構成

```text
dust-mask-repair/
  .gitignore
  AGENTS.md
  HANDOFF.md
  README.md
  pyproject.toml
  examples/
    README.md
  web/
    index.html
  src/
    dust_mask_repair/
      __init__.py
      benchmark.py
      cli.py
      config.py
      io.py
      mask.py
      metrics.py
      red_highlight.py
      red_highlight_cli.py
      repair.py
      server.py
      workflow.py
  tests/
    test_benchmark.py
    test_cli.py
    test_invariance.py
    test_mask_loading.py
    test_red_highlight.py
    test_red_highlight_regression.py
    test_repair.py
    test_server.py
    test_workflow.py
```

## 4. 依存関係

`pyproject.toml` 上のランタイム依存は最小限です。

- `numpy>=1.24`
  - 画像配列、マスク処理、補修カーネル、メトリクス計算に使用。
- `Pillow>=10.0`
  - TIFF fallback と一般的な画像サポートに使用。

オプション依存:

- `dev`: `pytest>=8.0`
- `tiff`: `tifffile>=2024.0`

実装時のローカル確認環境:

- Python: 3.12.10
- NumPy: 2.4.4
- Pillow: 11.3.0
- pytest: 8.4.2

`imageio` はローカルに未導入だったため使っていません。16bit RGB PNG をPillow任せにすると8bit化のリスクがあるため、PNGについては内部実装のreader/writerを用意しました。

## 5. インストールと実行コマンド

開発用インストール:

```powershell
py -3.12 -m pip install -e .[dev]
```

通常のテスト:

```powershell
py -3.12 -m pytest -q -p no:cacheprovider
```

基本的な構文・ビルド確認:

```powershell
py -3.12 -m compileall src
```

この環境では pytest のキャッシュ・一時ディレクトリ作成が権限エラーになることがあったため、`AGENTS.md` では `-p no:cacheprovider` を付けています。テスト側も `tmp_path` fixture 依存を避け、`test_outputs/` に固定出力します。

## 6. CLI仕様

エントリポイント:

```toml
[project.scripts]
dust-mask-repair = "dust_mask_repair.cli:main"
dust-mask-repair-web = "dust_mask_repair.server:main"
dust-mask-detect-red = "dust_mask_repair.red_highlight_cli:main"
dust-mask-repair-red = "dust_mask_repair.red_highlight_cli:repair_main"
dust-mask-benchmark = "dust_mask_repair.benchmark:main"
```

実行例:

```powershell
dust-mask-repair `
  --image input.png `
  --mask dust_mask.png `
  --output repaired.png `
  --method hybrid `
  --mask-channel auto `
  --threshold 0.5 `
  --dilate-radius 2 `
  --feather-radius 2 `
  --strength 1.0 `
  --max-component-area 5000 `
  --debug-dir debug_output
```

CLI引数:

- `--image`: 必須。入力RGB/RGBA PNG/TIFF。
- `--mask`: 必須。埃マスク。ヘルプ上はPNG前提だが、実装上は `read_image()` 経由なのでTIFFも読める可能性がある。ただしマスク仕様としてはPNGを前提に扱う。
- `--output`: 必須。出力PNG/TIFF。
- `--method`: `median`, `inpaint`, `denoise`, `hybrid`, `aggressive`。既定値は `hybrid`。
- `--mask-channel`: `auto`, `grayscale`, `alpha`, `red`, `max_rgb`。既定値は `auto`。
- `--threshold`: float。既定値 `0.5`。
- `--dilate-radius`: int。既定値 `2`。
- `--feather-radius`: int。既定値 `2`。
- `--strength`: float。既定値 `1.0`。
- `--min-component-area`: int。既定値 `1`。
- `--max-component-area`: int。既定値 `5000`。CLIでは `None` 指定不可。APIでは `None` 可。
- `--padding`: int。既定値 `16`。
- `--debug-dir`: 任意。指定時にデバッグ画像と `metrics.json` を出力。

CLIは処理後に `metrics` をJSONとして標準出力します。

### 6.1 赤ハイライト検出CLI

赤照明検査画像から白黒マスクを作るCLIを追加しています。

```powershell
dust-mask-detect-red --source red_lit_scan.png --output-dir red_mask_output
```

出力:

- `mask.png`
- `<stem>_red_highlight_mask.png`
- `input_preview.png`
- `preview_mask.png`
- `overlay_preview.png`
- `overlay.png`
- `component_features.json`
- `manifest.json`

赤照明マスク生成と補修を続けて行うCLI:

```powershell
dust-mask-repair-red --image normal_scan.png --red-image red_lit_scan.png --output repaired.png
```

`--image` と `--red-image` は同一寸法である必要があります。自動位置合わせ、回転、クロップ合わせ、RAW decode は未実装です。

長い赤スクラッチは誤検出リスクが違うため、既定では従来通り除外されます。検出する場合は明示的に有効化します。

```powershell
dust-mask-detect-red `
  --source red_lit_scan.png `
  --output-dir red_mask_output `
  --include-long-scratches `
  --min-scratch-aspect 5.0 `
  --max-scratch-width 48 `
  --max-scratch-dim 720 `
  --max-scratch-area 9000
```

`dust-mask-repair-red` でも同じ赤検出オプションを指定できます。

### 6.2 ベンチマークCLI

赤照明検出から生成マスク補修までを、決定的な合成fixtureで測るCLIを追加しています。

```powershell
dust-mask-benchmark `
  --width 1280 `
  --height 853 `
  --iterations 3 `
  --warmup 1 `
  --output-json benchmark_results/red_highlight_1280.json
```

出力JSONには以下を含みます。

- `detect_ms`, `repair_ms`, `total_ms`
- `component_count`
- `final_mask_pixels`
- `changed_pixel_count`
- `max_abs_diff_outside_mask`
- `peak_traced_memory_bytes`

`peak_traced_memory_bytes` は Python `tracemalloc` による計測であり、一部のnative allocationは含まれない可能性があります。性能改善前後の相対比較用の基準として扱ってください。

## 6.3 ローカルHTMLテストUI

`src/dust_mask_repair/server.py` と `web/index.html` を追加しています。

起動コマンド:

```powershell
$env:PYTHONPATH="src"
py -3.12 -m dust_mask_repair.server --host 127.0.0.1 --port 8765
```

ブラウザで開くURL:

```text
http://127.0.0.1:8765/
```

HTML UIでは、入力モードを選択できます。

- `mask PNG`: 補正対象ファイルと既存マスクファイルを選択し、従来通り `/api/repair` で補修します。
- `red highlight`: 補正対象ファイルと同一寸法の赤照明検査画像を選択し、`/api/repair-red` で白黒マスク生成から補修まで実行します。

実行後はbefore/afterをスライダー比較し、mask、diff、metricsも同じ画面で確認できます。赤照明モードでは、画面上のmask表示に生成済み白黒マスクを使います。出力は `web_outputs/<run_id>/` に保存されます。

赤照明モードでは `include long scratches` を有効にすると、細長い赤スクラッチ用の別上限（aspect、width、dim、area）を使って検出します。既定ではOFFです。

## 7. ライブラリAPI

公開APIは `src/dust_mask_repair/__init__.py` で次をexportしています。

```python
from dust_mask_repair import RepairConfig, RepairResult, repair_image
```

赤照明検査画像からマスクを作る追加APIもexportしています。

```python
from dust_mask_repair import RedHighlightConfig, detect_red_highlight_mask
```

`detect_red_highlight_mask(red_lit_image, RedHighlightConfig())` は、赤照明で浮いた埃・塵・短い欠陥を黒地の白マスク `uint8` として返します。通常スキャン画像から埃を検出する経路ではありません。長い赤スクラッチは `RedHighlightConfig(include_long_scratches=True)` のときだけ、`min_scratch_aspect` / `max_scratch_width` / `max_scratch_dim` / `max_scratch_area` の別上限で保持します。

本体統合用の連結APIもexportしています。

```python
from dust_mask_repair import RedHighlightRepairResult, repair_image_from_red_highlight
```

`repair_image_from_red_highlight(normal_rgb_or_rgba, red_lit_rgb, red_config, repair_config)` は、decode済みの通常画像配列と赤照明画像配列を受け取り、寸法一致を確認した上で、赤照明マスク生成と補修を連続実行します。RAW/DNG decodeはこのリポジトリでは行いません。

利用例:

```python
from dust_mask_repair import RepairConfig, repair_image

config = RepairConfig(
    method="hybrid",
    mask_channel="auto",
    threshold=0.5,
    dilate_radius=2,
    feather_radius=2,
    strength=1.0,
    min_component_area=1,
    max_component_area=5000,
    padding=16,
)

result = repair_image(image, mask, config)
```

赤照明画像から直接補修する場合:

```python
from dust_mask_repair import RedHighlightConfig, RepairConfig, repair_image_from_red_highlight

workflow_result = repair_image_from_red_highlight(
    normal_rgb_or_rgba,
    red_lit_rgb,
    red_config=RedHighlightConfig(),
    repair_config=RepairConfig(mask_channel="grayscale"),
)

generated_mask = workflow_result.generated_mask
repaired = workflow_result.repaired_image
```

### 7.1 `RepairConfig`

定義場所: `src/dust_mask_repair/config.py`

フィールドと既定値:

```python
method: str = "hybrid"
mask_channel: str = "auto"
threshold: float = 0.5
dilate_radius: int = 2
feather_radius: int = 2
strength: float = 1.0
min_component_area: int = 1
max_component_area: int | None = 5000
padding: int = 16
debug_dir: str | Path | None = None
collect_debug_images: bool = False
```

`validate()` で以下をチェックします。

- `method` は `median`, `inpaint`, `denoise`, `hybrid`, `aggressive` のいずれか。
- `mask_channel` は `auto`, `grayscale`, `alpha`, `red`, `max_rgb` のいずれか。
- `threshold` は `0.0..1.0`。
- `dilate_radius` と `feather_radius` は0以上。
- `strength` は `0.0..1.0`。
- `min_component_area` は0以上。
- `max_component_area` が `None` でない場合は `min_component_area` 以上。
- `padding` は0以上。
- `collect_debug_images=True` の場合、`debug_dir` がなくても `RepairResult.debug_images` を生成。

### 7.2 `RepairResult`

定義場所: `src/dust_mask_repair/repair.py`

フィールド:

```python
repaired_image: np.ndarray
binary_mask: np.ndarray
soft_mask: np.ndarray
changed_bbox_list: list[tuple[int, int, int, int]]
metrics: dict[str, Any]
debug_images: dict[str, np.ndarray]
debug_paths: dict[str, str]
```

`binary_mask` は threshold、成分フィルタ、dilate 後のboolマスクです。  
`soft_mask` は feather 後のfloat32マスクです。  
`changed_bbox_list` は `soft_mask > 0.0` の連結成分bboxです。
`debug_images` は `debug_dir` 指定時、または `collect_debug_images=True` の時だけ生成されます。

## 8. 処理パイプライン

中心関数は `repair_image(image, mask, config)` です。

処理順:

1. `RepairConfig.validate()` を実行。
2. `image` を `np.asarray()` で受け取る。
3. 画像形状を検証。
   - `image.ndim == 3`
   - channels は `3` または `4`
   - RGBまたはRGBAのみ受ける。
4. `normalize_mask(mask, cfg.mask_channel)` でマスクを `0.0..1.0` の `float32` に正規化。
5. 画像とマスクの幅・高さが一致しない場合は `ValueError`。
6. `threshold_mask()` で二値化。
7. `filter_components()` で連結成分を面積フィルタ。
   - `min_component_area` 未満を除外。
   - `max_component_area` 超過を除外。
   - 除外数はmetricsに入る。
8. `dilate_mask()` で二値マスクを拡張。
9. `feather_mask()` でsoft maskを作成。
10. `soft_mask > 0.0` から `changed_bbox_list` を作成。
11. `strength == 0.0` または `binary_mask` が空なら、入力画像の完全コピーを返す。
12. 画像を `as_float32()` で `0.0..1.0` float32へ変換。
13. `binary_mask` の連結成分ごとにROIを切る。
14. ROIには `padding` を加える。
15. `_repair_roi()` でRGBチャンネルのみ補修候補を生成。
16. `alpha = soft_mask * strength` で合成係数を作る。
17. RGBのみ `original * (1 - alpha) + repair_candidate * alpha` で合成。
18. `alpha <= 0.0` の画素は、float段階でもdtype復元後でも元画像で強制上書き。
19. `restore_dtype()` で元dtypeへ戻す。
20. metrics、必要ならdebug_images/debug_pathsを返す。

非常に重要な点:

- マスク外不変性は最終段階で強制しています。
- `feather_radius > 0` の場合、元の二値マスクの外側でも `soft_mask > 0` の境界領域は変更可能範囲になります。
- metricsの `outside_mask` は「元マスク外」ではなく「最終soft mask外」です。
- RGBA入力の場合も補修対象はRGBのみです。alphaチャンネルは保持されます。

## 9. 補修methodの詳細

実装場所: `src/dust_mask_repair/repair.py`

### 9.1 `median`

関数: `_median_repair()`

- ROI内で `mask == False` の画素をcontextとして使う。
- context画素のRGB中央値を計算。
- `mask == True` の画素を中央値で置換。
- 小さな埃や平坦部には強いが、エッジやテクスチャ復元は弱い。

### 9.2 `inpaint`

関数: `_diffusion_inpaint()`

- OpenCVは使っていない。
- Telea/Navier-Stokesではない。
- 既知画素から未知画素へ8近傍平均を反復伝播するdeterministicな局所fill。
- `max_iterations = max(4, roi_height + roi_width)`。
- 各反復で、周囲にfilled画素を持つunknown画素を平均値で埋める。
- 最後までunknownが残った場合、filled画素の中央値で埋める。
- 16bit入力を8bitに落とさないため、float32正規化後に処理する。

### 9.3 `denoise`

関数: `_masked_denoise()`

- ROI全体に半径1のbox blur候補を作る。
- ただし反映するのは `mask == True` の画素だけ。
- グローバルなぼかしではない。
- 埃を消すというより、マスク内の軽い平滑化用。

### 9.4 `hybrid`

関数: `_repair_roi()` 内の `method == "hybrid"` 分岐。

- 成分面積 `area <= 256` なら `_diffusion_inpaint()`。
- それより大きい kept component は `_median_repair()`。
- その後、半径1のbox blurを使い、マスク内だけ `85% repaired + 15% smoothed` にする。
- 境界の自然さは最終合成のsoft maskにも依存する。

### 9.5 `aggressive`

関数: `_aggressive_repair()`

- `hybrid` より強く補正を見せるためのmethod。
- diffusion fill後、周辺リングの中央値を45%混ぜる。
- その後、マスク内だけ半径1のbox blurを3回適用。
- 各平滑化後にcontext画素を元ROIで戻すため、ROI内のマスク外画素も保護される。
- 最終合成でも `soft_mask * strength` の範囲外は元画像で強制復元する。
- 白黒マスクの効果確認、レビュー、強めの埃消しに使う想定。

## 10. マスク処理の詳細

実装場所: `src/dust_mask_repair/mask.py`

### 10.1 `normalize_mask()`

戻り値:

```python
NormalizedMask(values: np.ndarray, channel_used: str)
```

対応チャンネル:

- `auto`
- `grayscale`
- `alpha`
- `red`
- `max_rgb`

正規化ルール:

- `uint8`: `value / 255.0`
- `uint16`: `value / 65535.0`
- float: `0.0..1.0` にclip
- その他integer: dtypeの最大値で割る

`grayscale` 指定時:

- 2Dならそのまま正規化。
- RGB/RGBAならRGB各チャンネルを先に正規化し、その後 Rec.709 係数で輝度化。
- 係数は `0.2126 R + 0.7152 G + 0.0722 B`。

`auto` 判定:

1. 2Dなら `grayscale`。
2. 2chまたは4chならalphaを確認。
   - alphaに信号があり、かつ「alpha全面1.0でRGBにも信号あり」ではない場合は `alpha`。
3. RGBが3チャンネル同一なら `grayscale`。
4. redの最大値がgreen/blue最大値の1.5倍以上なら `red`。
5. それ以外は `max_rgb`。

注意:

- RGBAの赤黒マスクでalphaが全面255の場合、alphaではなくRGB側を見る設計。
- `max_rgb` を2Dに指定した場合、実装上は `grayscale` として扱う。

### 10.2 `threshold_mask()`

`mask >= threshold` でbool化します。

### 10.3 `connected_components()`

- 8近傍の連結成分ラベリング。
- Pythonのstackベース実装。
- 戻り値は `(labels, components)`。
- `Component` は `label`, `area`, `bbox` を持つ。
- `bbox` は `(x0, y0, x1, y1)` の半開区間。

### 10.4 `filter_components()`

- threshold済みマスクに対して実行。
- `min_area` 未満、`max_area` 超過を除外。
- kept, removed_small, removed_large を返す。
- 現状、警告ログは出していない。除外結果はmetricsで確認する。

### 10.5 `dilate_mask()`

- 半径 `radius` の円形近似offsetで膨張。
- `dy * dy + dx * dx <= radius * radius` を満たすoffsetのみ使う。
- `radius <= 0` または空マスクならコピーを返す。

### 10.6 `feather_mask()`

- `binary_mask` をcoreとする。
- `radius <= 0` なら `binary.astype(float32)`。
- radiusありの場合:
  - coreを `dilate_mask(core, radius)` でexpandedにする。
  - core floatにbox blurをかける。
  - expanded外は0にする。
  - core内は必ず1.0に戻す。
- つまりsoft maskの変更可能範囲は、coreからfeather半径分に限定される。

## 11. 画像I/Oの詳細

実装場所: `src/dust_mask_repair/io.py`

### 11.1 `read_image()` / `write_image()`

- 拡張子でPNG/TIFFを分岐。
- PNGは内部実装。
- TIFFは `tifffile` があれば `tifffile`、なければPillow。
- 戻り値 `ImageData`:

```python
pixels: np.ndarray
bit_depth: int
color_mode: str
path: Path | None
metadata: dict[str, Any]
```

### 11.2 PNG reader

内部PNG readerの対応範囲:

- PNG signature確認あり。
- `IHDR`, `IDAT`, `IEND` を処理。
- bit depthは8または16のみ。
- color type:
  - `0`: grayscale
  - `2`: RGB
  - `4`: grayscale + alpha
  - `6`: RGBA
- interlaceは非対応。`interlace != 0` ならエラー。
- filter typeは `0..4` 対応。
- 16bitはPNGのbig-endianを `np.uint16` に変換。
- palette PNGなどは非対応。

### 11.3 PNG writer

内部PNG writerの対応範囲:

- `uint8` または `uint16` のみ。
- 2D grayscale、HWC 1ch、3ch RGB、4ch RGBA。
- filter typeは常に0。
- metadataやICCは書かない。

### 11.4 TIFF

- `tifffile` があればread/writeとも `tifffile`。
- `tifffile` がなければPillow。
- `uint16` のRGB/RGBA TIFFを書こうとして `tifffile` がない場合は `ValueError`。
- Pillow fallbackでは16bit RGB TIFFの完全保持は保証しない。

### 11.5 JPEG

- JPEG入力はPillowでRGB `uint8` として読み込む。
- JPEG出力もPillowで対応しているが、lossyなので保存・検証用途ではPNG/TIFF推奨。

### 11.6 dtype変換

`as_float32()`:

- `uint8`: `/ 255.0`
- `uint16`: `/ 65535.0`
- float: `0.0..1.0` にclip

`restore_dtype()`:

- `uint8`: `round(value * 255)`
- `uint16`: `round(value * 65535)`
- float: clip後に元dtypeへcast

## 12. metrics

実装場所: `src/dust_mask_repair/metrics.py`

基本metrics:

- `changed_pixel_count`
- `changed_bbox_count`
- `max_abs_diff_outside_mask`
- `mean_abs_diff_inside_mask`
- `mean_abs_diff_outside_mask`
- `processing_time_ms`

`repair.py` 側で追加されるmask metrics:

- `mask_channel_used`
- `kept_component_count`
- `removed_small_component_count`
- `removed_large_component_count`

差分計算は `as_float32()` 後の `0.0..1.0` 空間で行います。  
outside/insideの判定は `soft_mask > 0.0` です。

## 13. debug-dir出力

`RepairConfig.debug_dir` または CLI `--debug-dir` を指定した場合、以下を保存します。

- `normalized_mask.png`
- `binary_mask.png`
- `soft_mask.png`
- `repaired_preview.png`
- `diff_visualization.png`
- `metrics.json`

`result.debug_paths` に保存先パスが入ります。

注意:

- `debug_images` は `debug_dir` 指定時、または `collect_debug_images=True` の時だけ作成します。大画像の通常処理では余分なデバッグ配列を持たない設計です。
- `repaired_preview.png` は `repaired_image` そのものを書きます。入力が16bit PNGなら16bit PNGとして保存されます。
- `diff_visualization.png` は赤に差分、緑にsoft maskを入れたuint8 RGB可視化です。

## 14. テスト構成

テストは32件あります。

### 14.1 `tests/test_benchmark.py`

- `run_benchmark()` が赤照明検出と補修のsummaryを返すこと。
- CLI入口がJSONファイルを書き、`include_long_scratches` 設定を反映すること。

### 14.2 `tests/test_mask_loading.py`

1. `test_grayscale_mask_loads_and_normalizes`
   - grayscale PNGを内部writerで保存し、readerで読み、`grayscale` として正規化できること。
2. `test_alpha_mask_loads_and_normalizes`
   - RGBA maskのalphaを読めること。
   - `auto` がalphaを選ぶこと。
3. `test_red_channel_mask_loads_and_auto_detects`
   - 赤黒RGB maskのredを読めること。
   - `auto` がredを選ぶこと。
4. `test_rgb_grayscale_channel_normalizes_before_luma`
   - RGBを `grayscale` 指定した時、輝度化前に正規化され、128/255付近になること。

### 14.3 `tests/test_repair.py`

5. `test_empty_mask_returns_exact_input`
   - 空マスクなら出力が入力と完全一致。
6. `test_strength_zero_returns_exact_input`
   - maskがあっても `strength=0.0` なら完全一致。
7. `test_white_dust_is_repaired_only_in_mask`
   - 白い埃を合成した画像で、マスク内だけ補修されること。
8. `test_black_dust_is_repaired_only_in_mask`
   - 黒い埃を合成した画像で、マスク内だけ補修されること。

### 14.4 `tests/test_invariance.py`

9. `test_pixels_outside_soft_mask_are_unchanged`
   - `soft_mask <= 0.0` の画素が完全一致。
   - `max_abs_diff_outside_mask == 0.0`。
10. `test_unmasked_edge_content_is_unchanged`
   - マスク外の細線・エッジが変化しないこと。
11. `test_size_mismatch_raises_clear_error`
   - 画像とマスクのサイズ不一致で明示的に `ValueError`。
12. `test_uint16_png_roundtrip_and_repair_preserve_uint16`
   - 16bit RGB PNGのroundtripで `uint16` を維持。
   - 補修結果も `uint16`。
   - 255超の値が保持されること。

### 14.5 `tests/test_cli.py`

13. `test_cli_writes_output_and_debug_dir`
   - `python -m dust_mask_repair.cli` 経由でCLIを実行。
   - 出力画像を生成。
   - `debug-dir` の必須ファイル6種類を生成。
   - 出力の一部画素が入力と一致すること。

追加テスト:

14. `test_aggressive_method_preserves_pixels_outside_mask`
   - `aggressive` methodでもsoft mask外が完全一致すること。
15. `test_jpeg_input_can_be_read_for_cli_workflows`
   - JPEG入力を読み込めること。
16. `test_debug_images_are_opt_in_without_debug_dir`
   - `debug_dir` なしの通常補修では `debug_images` を作らず、`collect_debug_images=True` でのみ保持すること。

### 14.6 `tests/test_red_highlight.py`

- 赤照明検査画像から局所的な赤い埃・塵を検出できること。
- 画像端の広い赤グローを抑制できること。
- 通常モードでは長い赤スクラッチが従来のサイズ制限で除外されること。
- `include_long_scratches=True` では細長い赤スクラッチが保持されること。
- `tight` / `wide` のmask edge modeで境界サイズが変わること。
- sourceサイズとpreviewサイズが異なる場合も、最終マスクがsource寸法で返ること。
- CLIの検出出力、manifest、補修連結経路が動くこと。

### 14.7 `tests/test_red_highlight_regression.py`

- 粒状感、階調、エッジを含む小型合成フィルムスキャンfixtureで、赤照明検出から補修まで通す。
- 同fixture内に検出対象の赤い埃・短い傷と、無視すべき画像端の赤グローを同居させる。
- 生成マスクが対象欠陥を一定以上覆い、端グローをほぼ拾わないことを確認する。
- 補修後も `soft_mask <= 0.0` の画素が完全一致し、対象欠陥領域の暗い汚れが改善することを確認する。

### 14.8 `tests/test_server.py`

- `/api/repair-red` が通常画像と赤照明画像を受け取り、生成マスクと補修画像URLを返すこと。
- 通常画像と赤照明画像の寸法が異なる場合、400エラーで明示的に拒否すること。

### 14.9 `tests/test_workflow.py`

- `repair_image_from_red_highlight()` がdecode済みRGB配列を受け取り、赤照明マスク生成から補修まで返すこと。
- 通常画像と赤照明画像の寸法が異なる場合、明示的な `ValueError` を返すこと。

## 15. 最終検証結果

直近の確認結果:

```powershell
py -3.12 -m pytest -q -p no:cacheprovider
```

結果:

```text
32 passed
```

構文・ビルド確認:

```powershell
py -3.12 -m compileall src
```

結果:

```text
Listing 'src'...
Listing 'src\\dust_mask_repair'...
```

`compileall` は成功しています。

直近のベンチマーク確認:

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m dust_mask_repair.benchmark --width 640 --height 426 --iterations 3 --warmup 1 --detection-long-edge 640 --include-long-scratches --output-json test_outputs\benchmark\red_highlight_640.json
```

結果 summary:

```text
detect_ms median: 1087.830
repair_ms median: 655.025
total_ms median: 1741.270
peak_traced_memory_bytes median: 37780341
final_mask_pixels: 4820
changed_pixel_count: 5857
max_abs_diff_outside_mask_max: 0.0
```

参考: `debug_images` 常時生成時の同条件では、repair中央値 `694.302 ms`、total中央値 `1826.264 ms`、peak traced memory中央値 `37779788 bytes` でした。今回の変更ではrepair/totalは改善し、peak traced memoryは赤検出側の大きな中間配列が支配しているため横ばいです。

## 16. 品質上の最重要条件

このプロジェクトで最も重要なのは、マスク外画素不変性です。

守るべき条件:

- 空マスクなら出力は入力と完全一致。
- `strength=0.0` なら出力は入力と完全一致。
- `soft_mask <= 0.0` の画素は入力と完全一致。
- 画像全体へのblur、denoise、sharpen、color correctionは禁止。
- 補修候補をROI内で作ることは許可。ただし最終反映はsoft mask範囲だけ。
- 自動リサイズ・自動位置合わせはMVPでは禁止。
- 生成AI、拡散モデル、GAN、大規模MLモデルはMVPでは禁止。

現在の実装では、合成後に `outside = alpha <= 0.0` を使い、float段階とdtype復元後の両方で元画像を上書きしています。これによりsoft mask外の完全一致を担保しています。

## 17. 既知の制限と注意点

### 17.1 アルゴリズム品質

- `inpaint` は本格的な画像補完ではなく、8近傍平均の局所diffusion fillです。
- エッジ方向、テクスチャ、粒状性を明示的に推定していません。
- 長いスクラッチ検出は `include_long_scratches` を明示した場合のみ有効です。修復品質は周辺ROIの局所fillに依存するため、広い傷や太い傷はまだ弱いです。
- `hybrid` の閾値 `area <= 256` は暫定値です。
- `aggressive` はレビュー用の強補正として追加したため、自然さよりも効きの見えやすさを優先している。
- 2026-05-22: `aggressive` で白地に黒いシミが出る誤補正を抑えるため、局所統計ベースのguardを追加。補修候補が元画素より周辺統計から遠ざかる場合、または明るくきれいな領域を大きく暗くする場合は元画素を優先する。

### 17.2 性能

- 連結成分はPython stackベースです。巨大で密なマスクでは遅くなる可能性があります。
- dilationは半径内offsetを全て走査します。
- box blurは単純な二重ループで、積分画像やseparable filterではありません。
- `dust-mask-benchmark` で赤照明検出から補修までの処理時間を測れます。大きな実画像に近い条件では、width/height/iterationsを上げて再測定してください。

### 17.3 メモリ

- `debug_images` は必要時のみ作られます。
- 大きな16bit画像では `original`, `image_float`, `repair_candidate`, `output_float` が同時に存在します。`debug_dir` または `collect_debug_images=True` を使う場合は、さらにデバッグ配列のメモリが必要です。

### 17.4 I/O

- PNG metadata、ICC profile、gamma chunkなどは保持しません。
- PNG writerはfilter type 0のみです。圧縮効率は最適ではありません。
- interlaced PNG、palette PNGは非対応です。
- TIFFの16bit RGB/RGBAは `tifffile` なしでは完全保証できません。
- `read_image()` はfloat画像も扱える補助関数がありますが、ファイルI/Oは主にuint8/uint16前提です。

### 17.5 マスク semantics

- `feather_radius > 0` のとき、元のbinary mask外でもsoft maskの境界部分は変更対象になります。
- testsの「マスク外不変」は `soft_mask <= 0.0` を基準にしています。
- 「元の検出マスク外を1ピクセルも変えない」仕様に変えるなら、featherの設計を変更する必要があります。

### 17.6 CLI/API差分

- APIでは `max_component_area=None` が可能。
- CLIでは `--max-component-area` はintのみで、None指定は未実装。
- CLI helpではmaskをPNGと書いていますが、実装は `read_image()` 経由のためTIFF maskも通る可能性があります。ただし仕様としてはPNG前提です。

### 17.7 生成物

- `test_outputs/` はテスト実行で生成されます。
- `web_outputs/` はHTML UI実行で生成されます。
- `.pytest_tmp/` と `pytest-cache-files-*` は権限問題由来の残骸です。`.gitignore` 済みですが、環境によっては手動削除に管理者権限等が必要かもしれません。

## 18. 将来の改善案

優先度高:

1. 実フィルムスキャン画像サイズでベンチマークを継続取得し、最適化前後の比較値を残す。
2. `feather_radius > 0` のテストを追加し、変更可能範囲が `dilate + feather` に収まることを検証。
3. 大きすぎる成分を除外した時の警告またはstructured diagnosticsを追加。
4. `max_component_area=None` をCLIで指定できる表現を追加するか、明示的に禁止としてREADMEに書く。

優先度中:

1. optional OpenCV経路を追加し、8bit化が必要かどうかをREADMEに明記。
2. `tifffile` extraを使った16bit TIFF roundtripテスト。
3. ROIごとのdebug crop、before/after比較の保存。
4. edge-awareな補修方法を追加。
5. 粒状性を維持するためのマスク内限定grain/noise re-injection。
6. ruffやmypyなどの品質ゲートを追加。

優先度低:

1. PNG writerのfilter最適化。
2. palette PNGやinterlaced PNG対応。
3. メタデータ/ICC保持。
4. JSON/YAML設定ファイル読み込み。
5. 太い傷や広い傷向けの専用repair methodを追加。

## 19. 写真反転ソフト本体へ統合するときの注意

統合時に必ず確認すること:

- 本体側でRAW/DNGをdecodeし、このリポジトリにはdecode済みRGB/RGBA配列を渡す。
- 赤照明画像との連結は `repair_image_from_red_highlight()` を基本入口にする。
- 通常スキャン画像と埃マスクに同じ幾何変換を適用する。
- crop、rotate、resize、perspective補正後にマスク座標がずれないか確認する。
- 補修を「反転前RGB」に入れるか「反転後RGB」に入れるか比較する。
- RAW/DNG pipelineのscene-linear段に入れるのか、表示/出力RGB段に入れるのかを明確にする。
- 本体側でマスク外不変性の統合テストを追加する。
- 色変換、トーン、反転処理とは責務を分ける。
- この補修はグローバルdenoiseではないので、UI名や設定名でも「Dust/Spot Repair」系にする。

現時点のMVPはRGB/RGBA arrayを受ける設計です。本体のRAW decode直後のsensor配列やCFA配列を直接補修する設計にはなっていません。

## 20. 次に作業する人への実務メモ

作業前:

```powershell
cd C:\Users\windo\OneDrive\ドキュメント\Codex_projects\ConvertCodex\dust-mask-repair
git status --short
py -3.12 -m pytest -q -p no:cacheprovider
py -3.12 -m compileall src
```

新しい補修アルゴリズムを入れる場合:

- `RepairConfig.method` の許可リストを更新。
- CLI choicesを更新。
- READMEのmethod説明を更新。
- 少なくとも以下のテストを追加。
  - empty mask exact
  - strength=0 exact
  - outside soft mask exact
  - 8bit処理
  - 16bit処理
  - debug-dirが壊れないこと

I/Oを変更する場合:

- 16bit RGB PNG roundtripを必ず維持。
- TIFF対応を強化する場合は `tifffile` あり/なし両方の挙動を明記。
- metadata/ICCを扱うなら、READMEの既知の限界を更新。

マスク処理を変更する場合:

- `auto` 判定の既存テストを壊さない。
- featherのsupport範囲をテストで固定する。
- `soft_mask <= 0.0` の完全一致を維持する。

性能改善をする場合:

- まずベンチマークを追加。
- 変更前後で `max_abs_diff_outside_mask == 0.0` を確認。
- 高速化のためにグローバル処理へ逃げない。

## 21. 現時点の完了条件への対応

- ライブラリAPI: 実装済み。
- CLI: 実装済み。
- ローカルHTMLテストUI: 実装済み。
- テスト: 32件、pass確認済み。
- README: 実装済み。
- AGENTS.md: 実装済み。
- Debug output: 実装済み。
- 8/16bit PNG: 内部I/Oで対応。
- 16bit TIFF: optional `tifffile` 前提。Pillow fallbackの限界あり。
- マスク外不変性: `soft_mask <= 0.0` 基準でテスト済み。
- ベンチマーク: `dust-mask-benchmark` 実装済み。性能改善前後の比較用にJSON出力可能。
- Lint/Format: 専用ツール未設定。`compileall` のみ確認済み。

## 22. 最重要の引き継ぎポイント

このプロジェクトを進めるときは、補修品質より先に「どこを変えてよいか」を守ること。  
埃・塵はマスクで指定された局所欠陥であり、画像全体をなめらかにする処理ではありません。

実装上の防衛線は次の3つです。

1. 補修候補はROI内で作る。
2. 合成は `soft_mask * strength` で行う。
3. `alpha <= 0.0` の画素を最後に元画像で強制復元する。

この3点を崩す変更は、必ずテストと設計説明を伴って行ってください。
