# Dust Mask Repair 技術書

最終更新: 2026-06-21

## 1. 本書の位置づけ

本書は `dust-mask-repair` の現行実装を、設計・処理パイプライン・入出力・運用・検証の観点からまとめた技術書である。

このプロジェクトは、フィルムスキャンやカメラスキャン画像に含まれる埃、塵、短い傷などを、マスクで指定された局所領域だけ補修するためのサブシステムである。単体のPythonパッケージとして動作し、GUI、CLI、ローカルWeb UI、Python APIを持つ。

親プロジェクトであるフィルムネガ反転アプリケーションに対しては、RAW現像やネガ反転そのものを置き換えるものではなく、補修対象が明確な局所欠陥を扱う補助モジュールとして位置づける。

## 2. プロジェクトの目的

目的は、補正対象画像とマスク作成用画像を入力し、生成されたマスクに従って補正対象画像のマスク部分だけを補修することである。

主な利用シナリオは次の通り。

- 黒地または暗色地の検査画像から、白く写る浮き埃を検出する。
- 赤照明検査画像から、赤く強調された埃や短い傷を検出する。
- 既存の白黒マスクPNGまたはカスタムXMPマスクを読み込み、通常画像に適用する。
- 補修方式として `linear` または `kl` を使い、GUIで簡単に実行する。
- ARWなどのRAWファイルを、単体検証用にRGBへレンダリングして処理する。

最重要の品質条件は、マスク外画素を変更しないことである。補修品質よりも先に、変更範囲の制御が守られる必要がある。

## 3. 非目標

このプロジェクトは次を目標にしない。

- Lightroom、Capture One、darktableなどのRAW現像ソフト全体の再実装
- RAW現像パイプライン全体の再発明
- 画像内容に基づく完全自動の位置合わせ、回転補正、パース補正
- グローバルなノイズ除去、シャープ化、色補正
- 生成AI、GAN、拡散モデル、大規模MLモデルによるinpainting
- ARWなどのRAWファイルへ補修結果を直接書き戻すこと
- ICCプロファイルや全メタデータの完全保持

RAW入力は、単体検証と簡易GUIの利便性のために `rawpy` でRGBへレンダリングする経路である。親アプリ本体がRAW現像を持つ場合は、本体側で現像したRGB/RGBAバッファをAPIへ渡す設計が望ましい。

## 4. 全体構成

主要ファイルは次の通り。

| ファイル | 役割 |
| --- | --- |
| `src/dust_mask_repair/io.py` | PNG/JPEG/TIFF/RAWの読み書き。RAWは optional `rawpy` |
| `src/dust_mask_repair/mask.py` | マスク正規化、二値化、連結成分、膨張、feather |
| `src/dust_mask_repair/repair.py` | マスクガイド補修エンジン |
| `src/dust_mask_repair/white_dust.py` | 黒/暗色地の白埃検出 |
| `src/dust_mask_repair/red_highlight.py` | 赤照明検査画像からの赤ハイライト検出 |
| `src/dust_mask_repair/workflow.py` | 検出から補修までをまとめるPython API |
| `src/dust_mask_repair/gui.py` | Tkinterベースの簡易デスクトップGUI |
| `src/dust_mask_repair/server.py` | ローカルWeb UI/APIサーバ |
| `src/dust_mask_repair/cli.py` | 既存マスクからの補修CLI |
| `src/dust_mask_repair/white_dust_cli.py` | 白埃検出/補修CLI |
| `src/dust_mask_repair/red_highlight_cli.py` | 赤ハイライト検出/補修CLI |
| `src/dust_mask_repair/xmp.py` | カスタムXMPマスクsidecar |
| `src/dust_mask_repair/adobe_xmp.py` | Adobe native mask adapter境界の検証 |
| `KLComplementary2_0_GUI.pyw` | checkoutから直接起動するWindows GUIランチャー |

実行エントリポイントは `pyproject.toml` で定義されている。

| コマンド | 入口 |
| --- | --- |
| `dust-mask-repair` | `dust_mask_repair.cli:main` |
| `dust-mask-repair-gui` | `dust_mask_repair.gui:main` |
| `dust-mask-repair-web` | `dust_mask_repair.server:main` |
| `dust-mask-detect-white` | `dust_mask_repair.white_dust_cli:main` |
| `dust-mask-repair-white` | `dust_mask_repair.white_dust_cli:repair_main` |
| `dust-mask-detect-red` | `dust_mask_repair.red_highlight_cli:main` |
| `dust-mask-repair-red` | `dust_mask_repair.red_highlight_cli:repair_main` |
| `dust-mask-benchmark` | `dust_mask_repair.benchmark:main` |

## 5. 利用形態

### 5.1 簡易GUI

checkoutから直接起動する場合:

```powershell
cd C:\Users\windo\OneDrive\ドキュメント\Codex_projects\ConvertCodex\dust-mask-repair
py -3.12 KLComplementary2_0_GUI.pyw
```

インストール済みパッケージとして起動する場合:

```powershell
dust-mask-repair-gui
```

GUIでは次を選択する。

| UI項目 | 内容 |
| --- | --- |
| `補正対象写真` | 補修したい通常画像。PNG/JPEG/TIFF/ARWなど |
| `マスク作成用写真` | 黒地/暗色地の検査画像。PNG/JPEG/TIFF/ARWなど |
| `補正方式` | `kl`, `linear`, `defect_aware` |
| `検出 long edge` | 白埃検出用の縮小長辺。既定値は `1024` |
| `検出感度` | 白埃検出のしきい値感度 |
| `周辺補正 px` | 検出マスクを補修前に膨張する半径。GUI既定値は `1` |
| `境界なじませ px` | 拡張後の補修境界をsoft mask化する半径。GUI既定値は `2` |
| `色なじませ強度` | 補修候補のRGB平均/分散を周辺contextへ寄せる強度。`0.0..1.0`、GUI既定値は `0.65` |
| `粒状感強度` | 補修範囲へ周辺の高周波粒状感を再注入する強度。GUI既定値は `0.45` |
| `出力フォルダ` | タイムスタンプ付き実行フォルダを作る親フォルダ |
| `edge-guidedテスト` | 写真選択なしで合成斜めエッジ微小欠陥を生成し、`defect_aware` edge-guidedの可視確認とPASS/FAIL評価を行う |

GUI出力は実行ごとに次のようなフォルダへ保存される。

```text
gui_outputs/<target_stem>_<method>_<yyyymmdd_hhmmss>/
```

出力ファイル:

| ファイル | 内容 |
| --- | --- |
| `target_preview.png` | 補正対象写真の表示用プレビュー |
| `generated_mask.png` | 検査画像から生成した白黒マスク |
| `repair_mask_expanded.png` | `周辺補正 px` 適用後の実補修対象マスク |
| `inspection_overlay.png` | 検査画像上にマスクを重ねた確認画像 |
| `white_dust_score.png` | 検出スコアマップ |
| `processing_status.json` | 現在または最終処理状態 |
| `repaired.*` | 補修後画像。ARW入力時はPNG |
| `repair_metrics.json` | 補修メトリクス |
| `repair_result.json` | 実行記録 |
| `error.txt` | 実行フォルダ内の詳細エラー。失敗時 |
| `last_error.txt` | 出力親フォルダ直下の直近エラー。失敗時 |

`repair_result.json` は次の項目を含む。

```json
{
  "実行日": "...",
  "version": "KLComplementary 2.0 / dust-mask-repair ...",
  "対象写真": "...",
  "マスク作成用写真": "...",
  "補正方法": "kl",
  "正答率": null,
  "補正後画像": "...",
  "生成マスク": "...",
  "補正対象マスク": "...",
  "補正設定": {"mask_expand_radius": 1, "feather_radius": 2, "color_match_strength": 0.65, "grain_strength": 0.45}
}
```

通常のGUI補修ワークフローには模範解答画像がないため、`正答率` は `null` である。過去の9x9/100x100補完実験のように中央欠損の正解画像と比較する設計とは異なる。

`edge-guidedテスト` は例外的に合成した模範解答を持つ。出力フォルダには `edge_guided_clean_answer.png`, `edge_guided_damaged_input.png`, `edge_guided_mask.png`, `edge_guided_repaired_defect_aware.png`, `edge_guided_disabled_comparison.png`, `edge_guided_comparison.png`, `edge_guided_error_heatmap.png`, `edge_guided_test_metrics.json`, `edge_guided_test_result.json`, `debug/defect_components.json` を保存する。`edge_guided_test_result.json` の `正答率` はmask内MAEであり、edge-guided有効時、edge-guided無効時、破損入力の3値を比較する。

### 5.2 CLI

既存マスクから補修する。

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

白埃検査画像からマスクだけ作る。

```powershell
dust-mask-detect-white `
  --source dark_base_inspection.png `
  --output-dir white_mask_output `
  --background-mode dark `
  --detection-long-edge 1024
```

白埃検出から補修まで一括実行する。

```powershell
dust-mask-repair-white `
  --image normal_scan.png `
  --source dark_base_inspection.png `
  --output repaired.png `
  --mask-output generated_mask.png `
  --method kl
```

### 5.3 ローカルWeb UI

サーバ起動:

```powershell
py -3.12 -m dust_mask_repair.server --host 127.0.0.1 --port 8765
```

通常補修/赤ハイライト補修:

```text
http://127.0.0.1:8765/
```

白埃検出/白埃補修:

```text
http://127.0.0.1:8765/white_dust.html
```

Web UIはデバッグや比較確認のためのローカルテスト面である。大量のRAW処理や最終ワークフローは、現在は簡易GUIまたはPython API経由で扱う方が分かりやすい。

### 5.4 Python API

ホストアプリ側でRAW/DNG現像済みのRGB/RGBA配列を持つ場合、Python APIが最も自然である。

```python
from dust_mask_repair import RepairConfig, WhiteDustConfig, repair_image_from_white_dust

result = repair_image_from_white_dust(
    normal_rgb_or_rgba,
    dark_base_inspection_rgb,
    white_config=WhiteDustConfig(visual_artifacts=False),
    repair_config=RepairConfig(method="kl", mask_channel="grayscale"),
)

generated_mask = result.generated_mask
repaired_image = result.repaired_image
```

## 6. 依存関係

必須依存:

| 依存 | 用途 |
| --- | --- |
| `numpy` | 配列処理、補修カーネル |
| `Pillow` | JPEG/TIFF fallback、GUIプレビュー、resize |

任意依存:

| extra | 依存 | 用途 |
| --- | --- | --- |
| `dev` | `pytest` | テスト |
| `tiff` | `tifffile` | 16bit RGB/RGBA TIFF読み書き |
| `raw` | `rawpy` | ARW/DNG/RW2/FFFなどのRAW decode |

ARWを扱う場合:

```powershell
py -3.12 -m pip install -e .[raw]
```

RAW supportは `rawpy` / LibRaw に依存する。未対応カメラや壊れたRAWは明示的なdecodeエラーになる。

## 7. 入出力仕様

### 7.1 画像入力

対応する代表的な入力:

- PNG: 8bit/16bit
- JPEG: 8bit RGB
- TIFF: Pillowまたは`tifffile`
- RAW: ARW/DNG/RW2/FFF/CR2/CR3/NEF/ORF/RAFなど

`io.py` の `read_image()` は `ImageData` を返す。

```python
@dataclass(frozen=True)
class ImageData:
    pixels: np.ndarray
    bit_depth: int
    color_mode: str
    path: Path | None
    metadata: dict[str, Any]
```

RAW読み込みは `read_raw()` 経由で、`rawpy.postprocess()` によりsRGB RGB配列へ変換する。

現在のRAW設定:

- `use_camera_wb=True`
- `output_color=rawpy.ColorSpace.sRGB`
- `no_auto_bright=True`
- `output_bps=8` または `16`
- `half_size` は呼び出し側で指定

GUIでは、補正対象RAWは16bit、検査RAWは8bitフル解像度でdecodeする。検査RAWを8bitにする理由は、マスク座標を維持しながら白埃検出側のメモリを下げるためである。

### 7.2 画像出力

ARWなどのRAWへは書き戻さない。RAW入力時の補修結果は表示可能なPNGとして保存する。

| 入力 | GUI補修出力 |
| --- | --- |
| JPEG | JPEG |
| PNG | PNG |
| TIFF | PNG |
| RAW/ARW | PNG |

TIFFをGUIでPNG出力にしているのは、16bit RGB TIFF書き込みに任意依存が必要であり、GUIの安定性を優先しているためである。

## 8. マスク仕様

マスクは基本的に入力画像と同じ幅・高さである必要がある。

通常の `repair_image()` では、画像とマスクの寸法が異なる場合はエラーにする。これはマスク外不変性を守るための基本ルールである。

`mask_channel` は次を受け付ける。

| 値 | 意味 |
| --- | --- |
| `auto` | チャンネル内容から自動判定 |
| `grayscale` | 輝度として扱う |
| `alpha` | alphaチャンネル |
| `red` | redチャンネル |
| `max_rgb` | RGB最大値 |

`auto` は、2Dならgrayscale、alphaが有効ならalpha、RGB同値ならgrayscale、redが支配的ならred、それ以外はmax_rgbを選ぶ。

## 9. 白埃検出

白埃検出は `white_dust.py` が担当する。

対象は、黒地または暗色地の検査画像に明るく写る浮き埃である。旧来の茶色地検査画像に対応するため、`background_mode="brown"` も持つ。

主要設定:

| 設定 | 意味 |
| --- | --- |
| `detection_long_edge` | 検出用縮小画像の長辺 |
| `local_radius` | 局所背景評価半径 |
| `background_mode` | `dark` / `brown` / `any` |
| `mask_edge_mode` | `tight` / `normal` / `wide` |
| `threshold_sensitivity` | 検出感度 |
| `whiteness_min` | 白さの下限 |
| `min_area` / `max_area` | 成分面積制限 |
| `max_dim` / `max_thickness` | 形状制限 |
| `focus_margin_x/y` | 端の除外比率 |

検出結果は `WhiteDustSourceResult` で表される。

```python
@dataclass(frozen=True)
class WhiteDustSourceResult:
    mask: np.ndarray
    preview_mask: np.ndarray
    overlay: np.ndarray
    overlay_preview: np.ndarray
    score_map: np.ndarray
    components: list[dict[str, Any]]
    manifest: dict[str, Any]
```

GUIでは `visual_artifacts=False` で検出を行い、必要なoverlayはGUI側で軽量に作る。これにより、大きなARWでのメモリ使用を抑えている。

## 10. 赤ハイライト検出

赤ハイライト検出は `red_highlight.py` が担当する。

赤照明や赤く強調された検査画像から、埃、塵、短い傷を検出する。長い傷は誤検出リスクが異なるため既定では除外し、`--include-long-scratches` で明示的に有効化する。

白埃検出と同じく、出力は黒白マスクであり、補修エンジンは検出方式を意識しない。

## 11. 補修エンジン

補修エンジンは `repair.py` の `repair_image(image, mask, config)` である。

処理順序:

1. RGB/RGBA入力検証
2. マスク正規化
3. threshold二値化
4. connected components
5. component filtering
6. `core_mask` 作成
7. `repair_mask` 作成
8. `blend_alpha` 作成
9. componentごとにpadding付きROIを切り出し
10. methodごとの補修候補生成
11. guardで不自然な候補を抑制
12. `blend_alpha * strength` で合成
13. dtype復元
14. マスク外画素を元画像で強制復元
15. metrics/debug artifact生成

### 11.1 core mask / repair mask / blend alpha

`core_mask` はthresholdとcomponent filtering後の本来の欠陥領域である。

`repair_mask` は候補生成用のunknown領域で、`dilate_radius` により広げられる場合がある。

`blend_alpha` は最終合成用のalphaである。膨張shellは候補生成に使っても、最終合成では弱くすることで正常画素の過剰変化を避ける。

### 11.2 マスク外不変性

最終合成後、`alpha <= 0` の画素は元画像で強制復元される。

```text
output[alpha <= 0] = original[alpha <= 0]
```

この性質はテストで固定されている。

## 12. 補修method

現行method:

```text
linear
kl
defect_aware
median
inpaint
denoise
hybrid
adaptive
aggressive
wide_scratch
```

### 12.1 linear

マスク内をunknownとして、4近傍平均を反復して埋める線形補間である。

初期値は周辺既知画素の中央値。反復回数はマスク画素数に応じて決まる。現在はNumPy一括計算へベクトル化されており、以前のPython二重ループより高速である。

向いているケース:

- グラデーション上の点状から中程度の欠陥
- 色分布補正をかけたくない場合
- 白埃GUIで高速に結果を見たい場合

### 12.2 kl

まず `linear` と同じ補間結果を作り、周辺contextのRGBヒストグラムに近づくよう、マスク内画素を代表色へ割り当てる。

KL divergenceを直接連続最適化するのではなく、RGBを8分割/チャンネルのヒストグラムに量子化し、周辺分布に近いquotaを満たすよう代表色を割り当てる実用的な近似である。

向いているケース:

- 周辺色分布に合わせたい白埃補修
- 線形補間だけだと色が偏るケース
- GUI既定の補修方式

### 12.3 median

ROI内のマスク外contextの中央値でマスク内を置換する。

均一背景では安定するが、グラデーションや模様は平坦化しやすい。

### 12.4 inpaint

近傍既知画素から平均値を広げる決定的な局所fillである。大規模MLやOpenCV必須のinpaintではない。

### 12.5 denoise

小さなbox blurをマスク内だけに適用する。明確な欠陥除去より、弱い局所均しに向く。

### 12.6 adaptive / hybrid

`adaptive` は、正規化畳み込み、局所平面近似、PCA方向補間、任意のOpenCV Teleaなどを組み合わせる品質重視methodである。

`hybrid` は後方互換名で、現在はadaptive寄りの処理へルーティングする。

### 12.7 aggressive

強めに欠陥を消すレビュー用methodである。ring中央値、inpaint、局所平滑化を組み合わせる。guardで暗い染みを抑制するが、細部は失われやすい。

### 12.8 wide_scratch

縦傷・横傷のような広いscratchを、主軸/副軸方向のspan fillで埋める。

### 12.9 defect_aware

改善計画用の新しい入口である。

スライス10では、`method="defect_aware"` ルーターとして、連結成分分類、tiny/small local、fast inpaint、directional、patch、grain reinjection、既存blend合成までを統合し、CLI/API/GUI/Webから選択できる。GUI既定値は従来通り `kl` である。

分類器は `src/dust_mask_repair/defects.py` にあり、形状、周辺context、輝度ばらつき、簡易勾配から `recommended_strategy` を決める。

現時点のstrategy:

```text
tiny_local
small_local
fast_inpaint
directional
patch
skip
```

この段階で追加されるmetrics:

| 項目 | 内容 |
| --- | --- |
| `defect_aware` | `defect_aware` 経路で実行されたか |
| `defect_aware_version` | defect-aware経路のバージョン。スライス1では `1` |
| `defect_classification_enabled` | 欠陥分類が有効か |
| `defect_component_count` | 分類したcomponent数 |
| `defect_strategy_counts` | 推奨strategy別component数 |
| `defect_area_histogram` | component面積の簡易ヒストグラム |
| `defect_texture_summary` | texture scoreとgradientの要約 |
| `defect_classifier_version` | 分類器バージョン。スライス2では `1` |
| `defect_aware_fallback_method` | 現時点の実補修fallback method。`adaptive` |

`debug_dir` 指定時は `defect_components.json` にsummaryとcomponent特徴量、`defect_strategy_summary.json` にsummaryのみを保存する。

小欠陥補修は `src/dust_mask_repair/local_repair.py` の `repair_small_local_roi()` で行う。現在は局所平面補間とcontext中央値を維持したまま、明確な周辺画像構造がある場合だけ edge-guided local repair を優先する。

目的は、点状または小面積の埃が斜めエッジ、輪郭、髪、枝、文字などに重なったとき、局所平面補間や単純medianで起きやすいぼけ、色混合、輪郭切断を減らすことである。新しい公開method名は追加せず、`defect_aware` の `tiny_local` / `small_local` 内部サブストラテジーとして扱う。

```text
component mask
↓
repair maskを除外した正常context ring
↓
RGB構造テンソルで方向信頼度を推定
↓ 方向が明確な場合
isophote方向の両側正常画素から距離重み補間
↓ 一部または全体が成立しない場合
local plane fit または context median
↓
robust clamp
↓
grain reinjection / guard / blend
```

構造テンソルは、RGB各チャンネルの局所勾配を合算して作る。輝度差だけでなく、同一輝度に近い色境界も検出できるよう、`gx` / `gy` はRGBベクトルとして扱う。

```text
Jxx = sum(weight * sum_channel(gx ** 2))
Jyy = sum(weight * sum_channel(gy ** 2))
Jxy = sum(weight * sum_channel(gx * gy))

coherence = (lambda_max - lambda_min) / (lambda_max + lambda_min + epsilon)
```

勾配サンプルは、中心画素と差分に使う上下左右画素がすべて `repair_mask` 外の既知画素である場合だけ使う。マスク内の元RGB値は構造推定にも候補生成にも使わない。RGBA入力ではalphaは構造推定から除外し、alphaチャンネルは変更しない。

主勾配方向に直交する方向をエッジ接線、つまりisophote方向として使う。各core画素から正負両方向を有限半径で決定的に探索し、両側に正常画素が見つかった場合だけ距離重みでRGB候補を作る。片側しか見つからない画素や、探索capに達した画素は既存のlocal plane/median結果で埋めるため、未補間画素は残らない。

適用条件:

- `defect_aware` の `tiny_local` / `small_local` componentであること。
- component面積、ROI面積、1画素あたり探索半径、componentあたり総探索量が上限内であること。
- RGB構造テンソルのgradient energyとcoherenceがしきい値以上であること。
- 正負両側に `repair_mask` 外の正常画素が見つかること。

fallback条件:

- edge-guidedが無効。
- component面積またはROI面積が上限超過。
- contextまたは有効勾配サンプルが不足。
- gradient energy不足。
- coherence不足。
- isophote正負両側の正常画素が揃わない。
- 探索総量capに達する。

追加config:

| 項目 | 既定値 | 内容 |
| --- | ---: | --- |
| `edge_guided_enabled` | `True` | tiny/small local内のedge-guided候補を有効化 |
| `edge_guided_max_component_area` | `64` | edge-guidedを試すcomponent面積上限 |
| `edge_guided_context_radius` | `4` | 構造推定用context ring半径 |
| `edge_guided_search_radius` | `8` | isophote方向の片側探索上限 |
| `edge_guided_min_coherence` | `0.35` | 方向信頼度の下限 |
| `edge_guided_min_gradient_energy` | `2.5e-4` | RGB勾配エネルギー下限 |
| `edge_guided_max_roi_area` | `4096` | edge-guidedを試すROI面積上限 |
| `edge_guided_max_total_search` | `4096` | componentあたりの総探索量cap |

追加metrics:

| 項目 | 内容 |
| --- | --- |
| `small_local_component_count` | tiny/small localで処理したcomponent数 |
| `small_local_pixel_count` | tiny/small localで処理したcore画素数 |
| `small_local_plane_count` | local planeを基礎候補またはfallbackとして使ったcomponent数 |
| `small_local_median_count` | context中央値を基礎候補またはfallbackとして使ったcomponent数 |
| `small_local_fallback_count` | context不足などでadaptive fallbackへ逃がしたcomponent数 |
| `small_local_edge_guided_component_count` | edge-guided候補を1画素以上採用したcomponent数 |
| `small_local_edge_guided_pixel_count` | edge-guidedで直接補間したcore画素数 |
| `small_local_edge_guided_fallback_count` | edge-guidedが不成立または部分成立でlocal fallbackを使ったcomponent数 |
| `small_local_edge_guided_low_confidence_count` | energy/coherence/context不足で方向信頼度不足だったcomponent数 |
| `small_local_edge_guided_coherence_mean` | 有効勾配サンプルがあったcomponentのcoherence平均 |

`debug_dir` 指定時の `defect_components.json` には、該当componentごとに `edge_guided_used`, `edge_guided_fallback_reason`, `edge_guided_coherence`, `edge_guided_gradient_energy`, `edge_guided_sample_count`, `local_method`, `local_fallback_method` を追加する。画素単位の巨大デバッグ情報は出力しない。

#### tone-guided local repair

黒い埃や細い黒線が「薄い灰色の跡」として残る問題に対し、`tiny_local` / `small_local` のfallback候補として tone-guided local repair を追加している。これはマスク内の破損RGBを参照せず、`repair_mask` 外のdonor patchだけから、期待色に近い候補を選ぶ局所補修である。

処理の要点:

1. local planeまたはcontext中央値で、欠陥位置の期待RGBを作る。
2. `repair_mask` 外で、patch全体が既知画素だけからなるdonor候補を集める。
3. 各core画素について、期待RGBとの差、空間距離、patch内のtexture量、RGB勾配方向差をscore化する。
4. score上位 `tone_guided_top_k` 件を逆score重みで平均し、低周波色と少量の高周波残差を合成する。
5. 成立しない場合は既存のlocal plane/context medianへfallbackする。

追加config:

| 項目 | 既定値 | 内容 |
| --- | ---: | --- |
| `tone_guided_enabled` | `True` | tone-guided候補を有効化 |
| `tone_guided_max_component_area` | `64` | 対象component面積上限 |
| `tone_guided_max_roi_area` | `4096` | 対象ROI面積上限 |
| `tone_guided_context_radius` | `6` | donor context ring半径 |
| `tone_guided_search_radius` | `10` | 画素ごとの近傍donor探索半径 |
| `tone_guided_patch_radius` | `1` | donor patch半径 |
| `tone_guided_candidate_cap` | `256` | component内で評価するdonor上限 |
| `tone_guided_top_k` | `5` | 採用donor数 |
| `tone_guided_tone_weight` | `2.0` | 期待RGB差の重み |
| `tone_guided_spatial_weight` | `0.35` | 空間距離の重み |
| `tone_guided_texture_weight` | `0.25` | texture差の重み |
| `tone_guided_gradient_weight` | `0.15` | RGB勾配方向差の重み |
| `tone_guided_min_context_pixels` | `8` | 最低context画素数 |

追加metrics:

| 項目 | 内容 |
| --- | --- |
| `tone_guided_component_count` | tone-guided候補を採用したcomponent数 |
| `tone_guided_pixel_count` | tone-guidedで補修したcore画素数 |
| `tone_guided_fallback_count` | tone-guidedが成立せずfallbackしたcomponent数 |
| `tone_guided_no_context_count` | donor/context不足のcomponent数 |
| `tone_guided_candidate_count_total` | 評価donor候補数合計 |
| `tone_guided_top_k_mean` | 画素ごとの採用donor数平均 |
| `tone_guided_score_mean` | 採用donor score平均 |
| `tone_guided_context_rgb_distance_mean` | 採用donorと期待RGBの平均距離 |

`fast_inpaint` は `repair_fast_inpaint_roi()` で行う。

```text
context plane/median init
↓
known pixels fixed
↓
unknown 8-neighbor iterative update
↓
iteration cap
```

追加metrics:

| 項目 | 内容 |
| --- | --- |
| `fast_inpaint_component_count` | fast inpaintで処理したcomponent数 |
| `fast_inpaint_pixel_count` | fast inpaint対象画素数 |
| `fast_inpaint_iterations_total` | 反復回数合計 |
| `fast_inpaint_fallback_count` | context不足などでadaptive fallbackへ逃がしたcomponent数 |

`directional` は細長いcomponent向けである。component座標のPCAから長手方向を推定し、その垂直方向へ近い正常画素を探して補間する。対象画素数がcapを超える場合はadaptive fallbackへ逃がす。

追加metrics:

| 項目 | 内容 |
| --- | --- |
| `directional_component_count` | directionalで処理したcomponent数 |
| `directional_pixel_count` | directional対象画素数 |
| `directional_fallback_count` | 未補間やcapでfallbackを使ったcomponent数 |
| `directional_cap_exceeded_count` | cap超過で専用処理を避けたcomponent数 |

`patch` は `repair_patch_match_roi()` で行う。target windowと同サイズのcandidate windowを周辺だけから決定的に走査し、target known pixelsとのSSDが最小の候補からcomponent内だけをコピーする。repair maskと重なる候補は除外する。

追加metrics:

| 項目 | 内容 |
| --- | --- |
| `patch_component_count` | patchで処理したcomponent数 |
| `patch_pixel_count` | patch対象core画素数 |
| `patch_candidate_count_total` | 評価した候補数 |
| `patch_fallback_count` | 候補不足などでadaptive fallbackへ逃がしたcomponent数 |
| `patch_best_score_mean` | 採用候補score平均 |
| `patch_stride_used_counts` | 探索stride別component数 |

grain reinjectionは `reinject_grain_roi()` で行う。元ROIの高周波残差をcontextから採取し、座標hashでcomponent内へ割り当てる。乱数は使わない。

追加config:

| 項目 | 既定値 | 内容 |
| --- | ---: | --- |
| `grain_reinject_strength` | `0.25` | 再注入強度。`0` で完全無効 |
| `grain_context_radius` | `8` | context ring半径 |
| `grain_blur_radius` | `1` | 高周波残差用box blur半径 |
| `grain_min_context_pixels` | `16` | 最低context画素数 |

追加metrics:

| 項目 | 内容 |
| --- | --- |
| `grain_reinject_enabled` | grain再注入が有効か |
| `grain_reinject_strength` | 使用強度 |
| `grain_reinject_component_count` | 実際に再注入したcomponent数 |
| `grain_reinject_pixel_count` | 再注入した画素数 |
| `grain_reinject_skipped_no_context_count` | context不足でskipしたcomponent数 |

blend metrics:

| 項目 | 内容 |
| --- | --- |
| `defect_aware_blend_shell_pixel_count` | shell領域でalphaが非0の画素数 |
| `defect_aware_alpha_nonzero_pixel_count` | 最終alphaが非0の画素数 |
| `defect_core_alpha_min` / `mean` / `max` | core領域の最終alpha統計 |
| `defect_core_alpha_below_full_count` | `strength=1.0` でcoreが完全置換されなかった画素数。通常は `0` |
| `defect_shell_alpha_min` / `mean` / `max` | shell領域の最終alpha統計 |

GUI/Web UIの選択肢にも追加済みである。GUI高速モードで `defect_aware` が選ばれた場合はlinear系fast fallbackを使い、coreを `alpha=1.0` で置換し、拡張shellだけを弱いalphaでなじませる。metricsには `gui_fast_fallback_method`, `gui_fast_core_alpha_below_full_count`, `defect_core_alpha_below_full_count` を記録する。

## 12.10 defect-aware品質評価

`benchmark.py` にdefect-aware用の軽量合成ケースを追加している。

```python
from dust_mask_repair.benchmark import evaluate_defect_aware_quality_case

result = evaluate_defect_aware_quality_case("gradient_dust")
```

ケース:

```text
flat_dots
gradient_dust
grain_dust
stripe_texture
diagonal_edge
thin_scratch
diagonal_edge_micro_dust
chroma_edge_micro_dust
thin_line_micro_dust
gradient_micro_dust
mottled_background_dark_dust
```

`diagonal_edge_micro_dust` は斜め境界上の2から5画素程度の微小欠陥、`chroma_edge_micro_dust` は輝度差が小さくRGB色差が大きい境界上の微小欠陥、`thin_line_micro_dust` は1から2画素級の細線上の点状欠陥、`gradient_micro_dust` は滑らかなグラデーション上の欠陥を表す。`mottled_background_dark_dust` は斑状の色背景に黒い埃点と短い黒線があるケースで、黒い跡が残る問題の回帰検証に使う。評価はmask内MAE、edge-guided無効時との比較、tone-guided isolate比較、core luminance MAE、local variance retention、residual dark contrast、マスク外不変性を含む。

pytestでは、少なくとも次を確認する。

- corruptedよりinside errorが下がるケースがある。
- `diagonal_edge_micro_dust` でedge-guided有効時が無効時より改善する。
- `chroma_edge_micro_dust` で色境界を悪化させない。
- `gradient_micro_dust` ではedge-guidedを使わずlocal planeへfallbackできる。
- deterministicである。
- `max_abs_diff_outside_mask == 0`。
- `mottled_background_dark_dust` でtone-guided isolate比較がtone-guided無効時より改善する。
- `defect_core_alpha_below_full_count == 0`。
- strategy countsがcomponent countと一致する。

## 13. GUI高速モード

ARWフル解像度で検出したマスクが大きい場合、通常のcomponent分割とROI補修は非常に重い。

実例として、`processing_status.json` に次の状態が記録された。

```json
{
  "phase": "repairing",
  "method": "linear",
  "mask_pixels": 559713
}
```

この規模では、線形補修とconnected componentsが純Python処理を多く含むため、処理が終わらないように見える。

そのためGUIでは、マスク画素数が `100000` を超えた場合に高速モードへ切り替える。

```text
GUI_FAST_MASK_PIXEL_THRESHOLD = 100000
```

高速モードの特徴:

- 連結成分解析を避ける。
- マスク座標だけをNumPyでまとめて処理する。
- `linear` はマスク座標の4近傍平均を一括反復する。
- `kl` は高速線形結果に対して、周辺contextの512-bin RGB分布へ近づける。
- マスク外画素は元画像で強制復元する。
- `repair_metrics.json` に `gui_fast_mode: true` を記録する。
- `processing_status.json` の `phase` は `repairing_fast` になる。

高速モードはGUIの応答性と完走性を優先する近似である。精密なcomponent別ROI処理が必要な場合は、マスク検出感度を下げてマスク量を減らすか、CLI/APIで対象範囲を絞る。

## 14. guard処理

補修候補は `_guard_repair_candidate()` を通る。

guardの目的は、補修候補が周辺contextから見て極端な色になることを防ぐことである。補正対象内の元画素は埃や傷で汚染されている前提なので、candidateとの優劣比較やfallbackの参照値には使わない。

評価の概念:

- context ringから分位点、中央値、MADを推定
- candidateのmask内画素をcontext由来の範囲へclip
- clipされたcoreは局所平面またはcontext中央値から作ったfallback候補へ差し替え可能
- fallback不能な場合でもmask内の元画素へ戻さず、contextで制限されたcandidateを維持
- shellも元画素へ戻さず、context範囲内のcandidateとして扱う

これにより、補正対象内に残っている埃色を再利用しない。`aggressive` などの強いmethodでも、contextから外れた色を出しにくくする。

## 15. metrics

通常補修metricsの代表項目:

| 項目 | 内容 |
| --- | --- |
| `changed_pixel_count` | 変化した画素数 |
| `changed_bbox_count` | 変化領域bbox数 |
| `max_abs_diff_outside_mask` | マスク外最大差分 |
| `mean_abs_diff_inside_mask` | マスク内平均差分 |
| `processing_time_ms` | 処理時間 |
| `mask_channel_used` | 使用したマスクチャンネル |
| `kept_component_count` | 補修対象component数 |
| `removed_large_component_count` | 面積上限で除外されたcomponent数 |
| `guard_rejected_pixel_count` | guardでrejectした画素数 |
| `guard_rejected_core_pixel_count` | core内でrejectされた画素数 |
| `guard_rejected_shell_pixel_count` | shell内でrejectされ元画素保持になった画素数 |
| `guard_core_fallback_success_count` | core reject後にfallback候補で再補修できた画素数 |
| `guard_core_unrepaired_pixel_count` | fallback不能で元画素へ戻したcore画素数 |

GUI高速モードでは次の項目が加わる。

| 項目 | 内容 |
| --- | --- |
| `gui_fast_mode` | 高速モードか |
| `linear_iterations` | 高速線形補間の反復数 |
| `mask_pixel_count` | マスク画素数 |
| `max_abs_diff_outside_mask` | マスク外最大差分 |

マスク外不変性を見る最重要指標は `max_abs_diff_outside_mask` である。

## 16. XMP sidecar

`xmp.py` はカスタムXMP sidecarを扱う。

namespace:

```text
https://dust-mask-repair.local/ns/1.0/
```

sidecarには次を含める。

- base64 PNG化したマスク
- detector manifest
- source path
- target path
- mask fitting metadata
- role
- outputMode

現在のXMPはホスト中立の交換ファイルであり、Adobe Camera Raw / Lightroom native local mask形式ではない。Adobe native compatibilityは `adobe_xmp.py` と `docs/adobe_native_xmp_notes.md` の境界で扱う。

## 17. Web API

`server.py` は `ThreadingHTTPServer` ベースのローカルAPIを提供する。

主要POST endpoint:

| endpoint | 内容 |
| --- | --- |
| `/api/repair` | 画像 + 既存マスクで補修 |
| `/api/repair-red` | 赤ハイライト画像からマスク生成して補修 |
| `/api/detect-white-dust` | 白埃マスク検出 |
| `/api/repair-white-dust` | 白埃マスク検出から補修 |

Webの白埃補修では、検査画像と対象画像の寸法やaspectが違う場合、`_fit_mask_to_image()` により中心crop + nearest-neighbor resizeで対象画像へ合わせる。これは簡易fitであり、画像内容に基づくregistrationではない。

## 18. テスト

標準テスト:

```powershell
py -3.12 -m pytest -q -p no:cacheprovider
```

構文/基本ビルド確認:

```powershell
py -3.12 -m compileall src
py -3.12 -m py_compile KLComplementary2_0_GUI.pyw
```

現時点で確認済みの代表性質:

- 空マスクでは入力を完全維持
- `strength=0` では入力を完全維持
- マスク外画素不変
- 8bit/16bit PNG処理
- ARW/RAW decode経路
- white dust検出
- red highlight検出
- CLI実行
- Web API実行
- GUI実行関数
- GUI ARW target/inspection
- GUI高速モード
- XMP sidecar読み書き
- Adobe native XMP adapter境界

最新確認では次が通っている。

```text
76 passed
```

## 19. 運用とトラブルシュート

### 19.1 GUIが処理中のままに見える

まず最新の実行フォルダの `processing_status.json` を確認する。

```powershell
Get-ChildItem gui_outputs -Recurse -Filter processing_status.json |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 |
  Get-Content
```

`phase` の意味:

| phase | 状態 |
| --- | --- |
| `started` | 入力を読み、実行フォルダ作成直後 |
| `detecting_mask` | 白埃マスク検出中 |
| `repairing` | 通常補修中 |
| `repairing_fast` | GUI高速補修中 |
| `complete` | 完了 |
| `failed` | 失敗 |

`mask_pixels` が非常に大きい場合、検出感度が高すぎる可能性がある。`threshold_sensitivity` を下げる、`detection_long_edge` を調整する、focus marginで端を除外するなどを検討する。

### 19.2 出力フォルダだけ作られて画像がない

失敗時は実行フォルダの `error.txt`、または出力親フォルダの `last_error.txt` を確認する。

```powershell
Get-Content gui_outputs\last_error.txt -Tail 80
```

`processing_status.json` が `detecting_mask` または `repairing` で止まっている場合、処理中にプロセスが閉じられた、またはメモリ/時間で停止した可能性がある。

### 19.3 ARWが読めない

`rawpy` を入れる。

```powershell
py -3.12 -m pip install -e .[raw]
```

それでも失敗する場合、LibRawがそのARW variantをサポートしていない可能性がある。

### 19.4 サイズ不一致

通常GUI/APIでは、補正対象画像とマスク作成用画像の解像度が一致していることが前提である。

サイズが違う場合は、同じ撮影条件・同じ解像度の画像を指定する。Web UIの一部には簡易mask fitがあるが、正確な位置合わせではない。

## 20. 性能上の注意

高解像度ARWでは、次が性能に効く。

- マスク画素数
- 検出用長辺 `detection_long_edge`
- 補修method
- component数
- debug artifact出力の有無

GUIではdebug中間画像を作らず、一定以上のマスク画素数で高速モードに切り替える。

CLI/APIで精密な補修を行う場合は、次を推奨する。

- 検出感度を上げすぎない。
- `max_component_area` を適切に設定する。
- 必要なら検査画像側でholderや明るい端を除外する。
- 大面積の汚れや背景巻き込みは、まずマスク検出設定で抑える。


### 20.1 局所空間周波数ガイド補修

`frequency_guided` は公開methodではなく、`method="defect_aware"` の内部sub-strategyである。目的は、大量の欠陥をすべて重い解析に掛けることではない。ユーザーが `frequency scope mask` で明示した少数の欠陥componentだけ、周辺の局所空間周波数と方向性を測って補修方式を補助する。

`frequency scope mask` は既存の欠陥マスクとは別である。

- defect mask: 実際に補修できる画素範囲を決める。
- frequency scope mask: その中で周波数ガイド補修を使ってよいcomponentを選ぶ。

scope maskは補修範囲を広げない。最終的に変更できる画素は従来通り `repair mask` / `blend alpha` 内だけで、mask外不変性は維持する。画像サイズが補正対象と一致しない場合はエラーにする。

Python API:

```python
result = repair_image(
    image,
    defect_mask,
    RepairConfig(method="defect_aware", frequency_guided_enabled=True),
    frequency_scope_mask=scope_mask,
)
```

CLI:

```powershell
py -3.12 -m dust_mask_repair.cli `
  --image target.png `
  --mask defect_mask.png `
  --output repaired.png `
  --method defect_aware `
  --frequency-scope-mask scope.png
```

`--frequency-scope-mask` を指定した場合は自動で有効化する。`--frequency-guided` は明示フラグとして残している。

GUIでは「空間周波数補正範囲マスク（任意）」を選択できる。指定された場合のみ、出力フォルダへ次を保存する。

- `frequency_scope_mask.png`
- `frequency_selected_core_mask.png`
- `frequency_selected_overlay.png`

内部descriptorは厳密なFFTスペクトルではない。全画像FFT、全欠陥FFT、全画像周波数マップは作らない。選択componentの小ROIだけで、既知画素から次を計算する。

- 正規化box filterによる低周波推定
- `small blur - large blur` による中周波近似
- 既知画素のみから作った `rgb - small blur` によるfine近似
- RGB差分二乗和による水平、垂直、斜め2方向の局所変化量
- dominant orientation
- anisotropy
- frequency centroid相当の近似値
- context known fraction

欠陥画素の元RGBはdescriptorに使わない。有限差分は両端が既知画素のときだけ有効にし、box filterは `sum(image * known) / sum(known)` の正規化畳み込みで求める。RGBAではalphaは解析にも補修にも使わず、出力alphaを保持する。

分類はcomponentごとに次の4つへ分ける。

| 分類 | 条件の概要 | 補修方針 |
| --- | --- | --- |
| `smooth_gradient` | 中高周波が低く、方向性も弱い | 既存local planeを優先 |
| `directional` | anisotropyが高く、方向別エネルギー差が大きい | 既存edge-guided系を優先 |
| `textured` | 中周波またはfine energyが高く、単一方向に偏らない | 周波数descriptorを含むpatch donor scoreを使う |
| `ambiguous` | context不足、cap超過、分類信頼度不足 | 既存defect_aware fallbackへ戻す |

low / mid / fineの役割は分けている。`frequency_guided` は低周波の色・グラデーションと構造的な中周波を扱う。最も細かい非構造の粒状感は既存のgrain reinjectionが担当する。frequency側で中周波を転写したcomponentでは、grain reinjectionの寄与を弱め、高周波を二重に重ねない。

主なconfig:

- `frequency_guided_enabled`: 既定False
- `frequency_context_radius`
- `frequency_scales`: 既定 `(1, 2, 4)`
- `frequency_max_selected_regions`
- `frequency_max_component_area`
- `frequency_max_roi_side`
- `frequency_max_roi_pixels`
- `frequency_search_radius`
- `frequency_patch_radius`
- `frequency_candidate_cap`
- `frequency_top_k`
- `frequency_min_context_pixels`
- `frequency_min_known_fraction`
- `frequency_smooth_threshold`
- `frequency_anisotropy_threshold`
- `frequency_*_weight`
- `frequency_midband_strength`

主なmetrics:

- `frequency_guided_enabled`
- `frequency_scope_mask_pixel_count`
- `frequency_selected_region_count`
- `frequency_selected_component_count`
- `frequency_selected_core_pixel_count`
- `frequency_analyzed_component_count`
- `frequency_roi_pixel_count_total`
- `frequency_candidate_count_total`
- `frequency_descriptor_time_ms`
- `frequency_repair_time_ms`
- `frequency_cap_exceeded_count`
- `frequency_pattern_counts`
- `frequency_fallback_reason_counts`
- `frequency_context_low_energy_mean`
- `frequency_context_mid_energy_mean`
- `frequency_context_fine_energy_mean`
- `frequency_anisotropy_mean`
- `frequency_signature_distance_before_mean`
- `frequency_signature_distance_after_mean`
- `frequency_fast_mode_override_count`

GUI fast modeでは、まず従来fast補修で全体を完走させる。その後、frequency scope maskから選択componentだけを抽出し、そのcomponentに限って通常の `defect_aware` + `frequency_guided` を実行し、該当alpha範囲だけfast結果へ上書きする。選択されていない大量欠陥は従来fast結果のままで、同じ画素をfast補修と精密補修で二重blendしない。

benchmarkには次を追加した。

- `selected_sky_gradient`
- `selected_vertical_hair`
- `selected_grid`
- `many_defects_one_selected`
- `no_selection_regression`

現時点の制約として、これは万能な復元器ではない。大面積欠損、位置ずれ、意味内容の生成、全画面denoise/sharpenは対象外である。descriptorは局所空間周波数の近似であり、厳密なFFT解析ではない。

## 21. フィルムネガ反転アプリへの統合方針

親アプリ本体へ統合する場合、次の責務分離を守る。

### 親アプリ側

- RAW/DNG decode
- scene-linearまたはworking-spaceへの変換
- フィルムネガ反転
- preview/exportの色管理
- tile/cache/render DAG
- UI上の位置合わせ、crop、rotation

### dust-mask-repair側

- RGB/RGBA配列とマスク配列を受け取る。
- マスク部分だけ局所補修する。
- マスク外不変性を守る。
- 補修結果とmetricsを返す。

統合時は、RAWそのものをこのモジュールへ渡すよりも、親アプリの正規パイプラインで生成したRGB/RGBA working bufferと、同じ幾何変換を適用したマスクを渡す方が安全である。

## 22. 現在の制約と今後の課題

制約:

- RAW decodeは簡易RGBレンダリングであり、親アプリのscene-linear RAW pipelineではない。
- ARW出力はしない。補修結果はPNG/JPEG/TIFFなどの画像ファイル。
- GUI高速モードは完走性重視の近似であり、component別ROI補修と完全同一ではない。
- マスク検出が背景を巻き込むと、補修対象が大きくなり品質と速度が落ちる。
- 自動位置合わせはない。
- GPU処理、tile cache、キャンセル可能なjob制御は未実装。

今後の課題:

- GUIにキャンセルボタンを追加する。
- `processing_status.json` をGUI上に逐次表示する。
- 検出マスクのpreviewを見てから補修を開始する2段階GUIにする。
- マスク画素数が大きすぎる場合に警告し、補修前に確認する。
- 特徴点または手動offsetによるmask registrationを追加する。
- 親アプリのrender DAGへ局所補修stageとして統合する。
- preview/export consistencyを親アプリ側で検証する。

## 23. 開発時の作業手順

作業前:

```powershell
cd C:\Users\windo\OneDrive\ドキュメント\Codex_projects\ConvertCodex\dust-mask-repair
git status --short
```

変更後:

```powershell
py -3.12 -m pytest -q -p no:cacheprovider
py -3.12 -m compileall src
py -3.12 -m py_compile KLComplementary2_0_GUI.pyw
```

画像処理の変更で最低限確認すべきこと:

- マスク外画素が変わっていないか。
- 8bit/16bit dtypeが維持されるか。
- RAW/ARW経路が壊れていないか。
- GUIの出力フォルダに結果とstatus/errorが出るか。
- 大きいマスクで高速モードに入るか。
- docs/README/HANDOFFが実装とズレていないか。

## 24. 用語集

| 用語 | 意味 |
| --- | --- |
| 補正対象写真 | 補修したい通常画像 |
| マスク作成用写真 | 埃を検出するための検査画像 |
| core mask | 実際の欠陥中心領域 |
| repair mask | 補修候補生成用のunknown領域 |
| blend alpha | 最終合成用alpha |
| shell | repair maskからcore maskを除いた周辺領域 |
| context | マスク外の周辺画素 |
| KL補完 | 周辺RGB分布に近づける色分布補正付き補完 |
| linear補完 | 近傍平均反復による線形補間 |
| GUI fast mode | 大マスク時のGUI専用高速補修 |
| XMP sidecar | マスクとmanifestを含む補助ファイル |
| frequency scope mask | 周波数ガイド補修を許可する選択マスク |
| frequency_guided | defect_aware内部で選択componentだけ使う局所空間周波数ガイド補修 |
