# Repair Algorithm

最終更新: 2026-06-13

この文書は、現在の補正機能がどのようなアルゴリズムで動いているかを、実装に沿って整理したものです。対象コードは主に `src/dust_mask_repair/repair.py`、`src/dust_mask_repair/mask.py`、`src/dust_mask_repair/server.py` です。

## 要約

補正機能は、機械学習や大規模な画像復元モデルではなく、マスクガイド型の局所補修アルゴリズムです。

入力画像と埃・傷マスクを受け取り、マスク部分だけを局所的に置き換えます。補修候補は、線形補間、KL分布補正、周辺画素の中央値、近傍からの拡散inpaint、正規化畳み込み、局所平面近似、PCA方向補間、軽い平滑化を使って作成します。最後にblend alphaとstrengthで元画像へ合成し、合成マスク外の画素は完全に元画像のまま保持します。

現在の基本方針は次の通りです。

- マスク外は変更しない。
- `core_mask` / `repair_mask` / `blend_alpha` を分離する。
- マスク内だけを補修し、膨張shellは弱いalphaでなじませる。
- 小さい埃は周辺画素から埋める。
- 細長い毛や傷はPCAで主方向を推定し、垂直方向のcontextから補間する。
- 補修候補が周辺より不自然に悪化する場合は元画素を維持する。
- デバッグ用に正規化マスク、core/repair/blend mask、guard前後候補、補修結果、差分画像、metricsを出せる。

## 入力と出力

### 入力画像

`repair_image(image, mask, config)` は、RGBまたはRGBA配列を受け取ります。

- 対応形状: `H x W x 3` または `H x W x 4`
- 対応dtype: 主に `uint8` / `uint16` / float
- 補修処理はRGBチャンネルに対して行います。
- RGBA入力の場合、alphaチャンネルは補修対象ではなく元の値を維持します。

Web UIの白埃補正経路では、2026-06-04時点で補正対象はJPEG/JPGです。

- 補正対象: `.jpg` / `.jpeg`
- 検査画像: PNG / JPEG / TIFF / RAW系、rawpy導入時はARW/DNG/RW2/FFFなど
- 出力: `repaired.jpg`

### 入力マスク

マスクは画像と同じ縦横サイズである必要があります。白埃補正Web経路では、検査画像から生成したマスクと補正対象JPEGのサイズや縦横比が異なる場合、補正対象側に合わせてマスクを投影します。

通常の `repair_image()` 本体では、画像とマスクのサイズが違う場合はエラーにします。

## 全体パイプライン

補正処理は次の順番で実行されます。

1. 入力画像の検証
2. マスクの正規化
3. thresholdによる二値化
4. connected componentsで領域分割
5. 面積によるcomponent filtering
6. `core_mask` を作る
7. `repair_mask` を作る
8. `blend_alpha` を作る
9. componentごとにpadding付きROIを切り出す
10. methodごとにROI内の補修候補を作る
11. guardで不自然な補修候補を抑制する
12. `blend_alpha * strength` で元画像と合成する
13. 元dtypeへ戻す
14. metricsとdebug artifactを生成する

## マスク処理

### 1. normalize

`normalize_mask()` でマスクを `0.0..1.0` のfloat32に正規化します。

`mask_channel` は次の値を取れます。

| 値 | 内容 |
| --- | --- |
| `auto` | マスク形状とチャンネル内容から自動判定 |
| `grayscale` | 輝度として扱う。RGBの場合は `0.2126 R + 0.7152 G + 0.0722 B` |
| `alpha` | alphaチャンネルを使う |
| `red` | redチャンネルを使う |
| `max_rgb` | RGBの最大値を使う |

`auto` の判定は次のような順です。

- 2Dならgrayscale
- alphaが有効で、かつ全面不透明RGB画像ではない場合はalpha
- RGBが完全に同値ならgrayscale
- redが他チャンネルより強い場合はred
- それ以外はmax_rgb

### 2. threshold

正規化マスクを `threshold` で二値化します。

```text
binary = normalized_mask >= threshold
```

既定値は `0.5` です。

### 3. connected components

二値マスクを8近傍でcomponent分割します。

斜め接続も同じcomponentとして扱います。各componentには次を持たせます。

- label
- area
- bbox: `(x0, y0, x1, y1)`

### 4. component filtering

`min_component_area` 未満のcomponentを除去します。

`max_component_area` が指定されている場合、それを超えるcomponentも除去します。これは、誤検出された大きな領域や背景全体の巻き込みを補修対象から外すためです。

### 5. core_mask

`core_mask` は、thresholdとcomponent filtering後に残った元の欠陥マスクです。

本当に強く補修したい中心領域として扱います。`core_mask` 内の `blend_alpha` は基本的に `1.0` です。

### 6. repair_mask

`repair_mask` は、補修候補を生成するときのunknown領域です。

```text
repair_mask = dilate(core_mask, dilate_radius)
```

`repair_mask` はcandidate生成用なので、coreより少し広げられます。ただし広げたshell部分を最終合成で完全置換しないよう、後段の `blend_alpha` で弱く扱います。

### 7. blend_alpha

`blend_alpha` は最終合成用のalphaです。`repair_mask` とは別です。

現在の実装では次のように作ります。

- `core_mask`: alpha `1.0`
- `shell_mask = repair_mask & ~core_mask`: alpha `0.15..0.45`
- `feather_radius > 0` の場合、coreをbox blurしてshell alphaを滑らかにする
- `strength` は最後に `blend_alpha` へ掛ける

これにより、欠陥中心はしっかり補修しつつ、膨張で追加された周辺正常画素の極端な色変化を避けます。

## ROI単位の補修

補修は画像全体に直接かけず、componentごとにpadding付きROIを作って行います。

```text
roi_bbox = component.bbox + padding
```

このROI内で補修候補を作り、最後に元画像へ戻します。paddingは周辺文脈を確保するための値です。

## 補修method

現在の補修methodは次の10種類です。

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

### linear

`linear` は、マスク芯だけを unknown として扱い、周辺既知画素から4近傍平均を反復して埋める線形補間です。膨張shellを補修候補生成には使わないため、基本的にマスク芯の補修に集中します。

特徴:

- グラデーションをまたぐ小から中程度の欠陥に向く。
- 決定的で軽量。
- テクスチャや繰り返し模様の再現力は限定的。

### kl

`kl` は、まず `linear` と同じ補間結果を作り、その後、マスク近傍のRGBヒストグラム分布にマスク内の色分布が近づくよう、マスク内画素を代表色へ割り当てます。KL divergenceを小さくすることを目的にした分布補正で、白埃検査画像からの補修連結では既定methodです。

特徴:

- 周辺色分布に対して線形補間だけでは偏るケースを補正しやすい。
- マスク外画素は変更しない。
- 画素配置の完全な構造復元ではなく、色分布の整合を優先する。

### median

マスク外のcontext画素の中央値を計算し、マスク内をその中央値で置き換えます。

```text
median = median(roi[~mask])
roi[mask] = median
```

特徴:

- 背景が均一な箇所では安定しやすい。
- グラデーションや模様がある箇所では平坦になりやすい。
- 大きめの単純な汚れを素早く消す用途に向く。

### inpaint

単純な拡散型inpaintです。OpenCV等の高度なinpaintではなく、周囲の既知画素から未確定領域へ平均値を広げていきます。

処理内容:

1. mask内をunknown、それ以外をfilledとする。
2. unknown画素のうち、8近傍にfilled画素があるものを候補にする。
3. 近傍filled画素の平均RGBでその画素を埋める。
4. 埋めた画素をfilledにする。
5. すべて埋まるか、最大反復回数に達するまで繰り返す。
6. まだunknownが残った場合はfilled領域の中央値で埋める。

最大反復回数は次の値です。

```text
max_iterations = max(4, roi_height + roi_width)
```

特徴:

- 小さい点状の埃に向く。
- 周辺から自然に色を押し込む。
- 大きい領域ではぼけやすい。

### denoise

ROI全体を半径1のbox blurでぼかし、mask内だけをblur結果で置き換えます。

特徴:

- 強い補修ではなく、局所的なざらつきを軽く均す用途。
- 明確な白点・黒点の除去力は弱め。

### adaptive

品質重視の決定的な補修方式で、機械学習や生成モデルは使いません。白埃補正Web UIの既定methodは現在 `kl` ですが、複雑なグラデーション背景や細い傷では `adaptive` も有効です。

componentごとに形状を見て処理を切り替えます。

```text
小さい点状埃       : cv2 Teleaがあれば任意使用、なければ正規化畳み込み
中サイズの埃       : 正規化畳み込み + 局所平面近似
細長い毛・傷       : PCA方向推定 + 垂直方向context補間
大面積/context不足 : 弱いalpha + fallback
```

正規化畳み込みでは、壊れたmask内画素を平均に混ぜません。

```text
known = ~unknown_mask
fill = blur(roi * known) / max(blur(known), eps)
```

局所平面近似では、contextからチャンネルごとに `value = ax + by + c` を最小二乗で当てはめます。上下5%相当の外れ値やMADで外れた値は弱め、グラデーション背景でmedian一発置換より自然になるようにしています。

細長い欠陥ではbboxだけでなくmask座標のPCAで主方向を推定し、主方向に垂直な両側contextから補間します。これにより斜めの毛や小傷にも対応しやすくしています。

### defect_aware

段階的な高品質補修パイプライン用の新しい入口です。

スライス9時点では、`defect_aware` ルーターとして、分類、tiny/small local、fast inpaint、directional、patch、grain reinjection、既存blend合成までを統合しています。候補不足やcap超過では既存の `adaptive` へフォールバックします。

分類器は `src/dust_mask_repair/defects.py` にあり、各componentについて次を計算します。

- area / bbox / width / height
- PCA由来の long_axis / short_axis / elongation
- thickness / density / touches_border
- context_pixel_count / context_available_ratio
- luminance_std / gradient_mean / gradient_anisotropy / texture_score
- `recommended_strategy`

現時点の `recommended_strategy`:

- `tiny_local`
- `small_local`
- `fast_inpaint`
- `directional`
- `patch`
- `skip`

追加metrics:

- `defect_aware`
- `defect_aware_version`
- `defect_classification_enabled`
- `defect_component_count`
- `defect_strategy_counts`
- `defect_area_histogram`
- `defect_texture_summary`
- `defect_classifier_version`
- `defect_aware_fallback_method`

`debug_dir` が指定されている場合は `defect_components.json` と `defect_strategy_summary.json` も保存します。component詳細は巨大化を避けるため上限付きで、summaryは全体件数を保持します。

小欠陥補修は `src/dust_mask_repair/local_repair.py` の `repair_small_local_roi()` で行います。

処理順:

1. component周辺だけをcontext ringとして集める。
2. contextが足りなければ `adaptive` fallbackへ戻す。
3. contextからチャンネルごとの一次平面を最小二乗で推定する。
4. 平面推定できない場合はcontext中央値で埋める。
5. contextのrobust範囲へclipする。
6. alpha channelは変更しない。

追加metrics:

- `small_local_component_count`
- `small_local_pixel_count`
- `small_local_plane_count`
- `small_local_median_count`
- `small_local_fallback_count`

`fast_inpaint` は同じ `local_repair.py` の `repair_fast_inpaint_roi()` で行います。

処理内容:

1. context ringの局所平面または中央値でunknownを初期化する。
2. 既知画素を毎回元画像へ戻して固定する。
3. unknownだけを8近傍平均で反復更新する。
4. 反復回数は `max_iterations` とROIサイズからcapする。
5. alpha channelは変更しない。

追加metrics:

- `fast_inpaint_component_count`
- `fast_inpaint_pixel_count`
- `fast_inpaint_iterations_total`
- `fast_inpaint_fallback_count`

`directional` はcomponentの主軸をPCAで推定し、傷の長手方向ではなく短軸方向に正常画素を探して補間します。対象画素数がcapを超える場合は重さを避けるため `adaptive` fallbackへ逃がします。

追加metrics:

- `directional_component_count`
- `directional_pixel_count`
- `directional_fallback_count`
- `directional_cap_exceeded_count`

`patch` は `src/dust_mask_repair/patch_repair.py` の `repair_patch_match_roi()` で行います。component bboxにmarginを足したtarget windowを作り、周辺search windowから同じサイズの候補を決定的に走査します。repair maskと重なる候補は使いません。

追加metrics:

- `patch_component_count`
- `patch_pixel_count`
- `patch_candidate_count_total`
- `patch_fallback_count`
- `patch_best_score_mean`
- `patch_stride_used_counts`

grain reinjectionは `src/dust_mask_repair/grain.py` の `reinject_grain_roi()` で行います。周辺contextのbox blur残差を集め、座標hashでcomponent内画素へ割り当てるため、同じ入力なら完全に同じ出力になります。`grain_reinject_strength=0` で完全無効です。

追加config:

- `grain_reinject_strength`
- `grain_context_radius`
- `grain_blur_radius`
- `grain_min_context_pixels`

追加metrics:

- `grain_reinject_enabled`
- `grain_reinject_strength`
- `grain_reinject_component_count`
- `grain_reinject_pixel_count`
- `grain_reinject_skipped_no_context_count`

blend安定化では既存の最終合成式を維持します。

```text
final_alpha = blend_alpha * component_alpha * strength
if method == "defect_aware" and defect_core_full_replace:
    final_alpha[core_mask] = strength
output = original * (1 - final_alpha) + candidate * final_alpha
output[final_alpha <= 0] = original
```

追加metrics:

- `defect_aware_blend_shell_pixel_count`
- `defect_aware_alpha_nonzero_pixel_count`
- `defect_core_alpha_min` / `defect_core_alpha_mean` / `defect_core_alpha_max`
- `defect_core_alpha_below_full_count`
- `defect_shell_alpha_min` / `defect_shell_alpha_mean` / `defect_shell_alpha_max`

CLI/API/GUI/Webから指定できます。GUIの既定値は従来通り `kl` です。GUI高速モードで `defect_aware` が選ばれた場合は、完走性を優先してlinear系fast fallbackを使う。ただしcoreは `alpha=1.0` で置換し、拡張shellだけ弱いalphaで合成する。metricsに `gui_fast_fallback_method: "linear"`, `gui_fast_core_alpha_below_full_count`, `defect_core_alpha_below_full_count` を記録します。



#### frequency_guided（選択component限定）

`frequency_guided` は公開methodではなく、`defect_aware` 内部の選択sub-strategyである。`RepairConfig.frequency_guided_enabled=True` と `frequency_scope_mask` が渡された場合だけ動作する。scope maskと重なる欠陥component全体を選択し、選択componentだけ局所空間周波数descriptorを計算する。scope maskは補修範囲を広げず、最終変更範囲は従来通り `repair_mask` / `blend_alpha` 内に限定される。

descriptorは全画像FFTではなく、選択ROI内の既知画素だけを使う軽量近似である。正規化box filter、RGB差分二乗和、scale `(1, 2, 4)` の局所変化量から、low / mid / fine energy、方向別energy、dominant orientation、anisotropy、context known fractionを求める。欠陥画素の元RGBはdescriptor、donor score、期待RGBの計算に使わない。

分類は `smooth_gradient`、`directional`、`textured`、`ambiguous`。smoothはlocal plane、directionalはedge-guided系、texturedはfrequency-aware patch donor scoreを優先し、ambiguousやcap超過は既存fallbackへ戻す。中周波残差を転写する場合はgrain reinjectionを弱め、高周波を二重に足さない。

追加metricsは `frequency_selected_component_count`、`frequency_analyzed_component_count`、`frequency_pattern_counts`、`frequency_context_mid_energy_mean`、`frequency_signature_distance_after_mean`、`frequency_fast_mode_override_count` など。`debug_dir` 指定時は `frequency_scope_mask.png`、`frequency_selected_core_mask.png`、`frequency_selected_overlay.png`、`frequency_pattern_map.png`、`frequency_components.json` を出力する。

#### defect_awareの黒い跡対策

黒い埃や黒線が薄い跡として残る主因は、core候補ができていても最終合成で `component_alpha` が掛かり、破損した元RGBが混ざることだった。現在の `defect_aware` は補正対象内の元色を信用せず、次で抑制する。

- `defect_core_full_replace=True` かつ `strength > 0` の場合、coreの最終alphaを `strength` に戻す。
- shellだけを弱いalphaでなじませ、coreには元の欠陥色を混ぜない。
- guardのcore rejectは `_guard_core_fallback_candidate()` で局所平面/context中央値へ再補修し、fallback不能な場合も元画素へ戻さずcontextで制限したcandidateを維持する。
- tiny/small localには tone-guided local repair を追加し、周辺donor patchのRGB、texture、勾配方向から黒点・黒線に近い破損色を使わない候補を作る。

関連metrics:

- `guard_rejected_core_pixel_count`
- `guard_rejected_shell_pixel_count`
- `guard_core_fallback_success_count`
- `guard_core_unrepaired_pixel_count`
- `tone_guided_component_count`
- `tone_guided_pixel_count`
- `tone_guided_score_mean`
- `tone_guided_context_rgb_distance_mean`

### defect-aware品質評価

`src/dust_mask_repair/benchmark.py` には軽量な合成品質評価helperがあります。

- `make_defect_aware_quality_case()`
- `evaluate_defect_aware_quality_case()`

ケース:

- `flat_dots`
- `gradient_dust`
- `grain_dust`
- `stripe_texture`
- `diagonal_edge`
- `thin_scratch`
- `diagonal_edge_micro_dust`
- `chroma_edge_micro_dust`
- `thin_line_micro_dust`
- `gradient_micro_dust`
- `mottled_background_dark_dust`

評価指標:

- `processing_time_ms`
- `mask_pixel_count`
- `component_count`
- `mean_abs_error_inside_mask`
- `corrupted_mean_abs_error_inside_mask`
- `max_abs_diff_outside_mask`
- `strategy_counts`
- `core_rgb_mae` / `core_luminance_mae`
- `corrupted_improvement_ratio`
- `tone_guided_enabled_isolated_mean_abs_error_inside_mask`
- `tone_guided_disabled_mean_abs_error_inside_mask`
- `local_variance_retention`
- `residual_dark_contrast`

### hybrid

後方互換のため残しているmethodです。

旧実装では `area > 256` をmedian一発置換していましたが、現在はadaptiveと同じ補修カーネルを使います。大きめの領域でも、正規化畳み込み・局所平面近似・PCA方向補間を経由し、直接medianへは進みません。

特徴:

- 既存の `method="hybrid"` 指定を壊さない。
- 内部品質はadaptive寄り。
- 旧hybridよりグラデーション背景で平坦化しにくい。

### aggressive

強めに除去します。レビュー/testing向けで、通常の白埃補修では `kl` または `linear` を先に確認してください。

処理内容:

1. まず `inpaint` で補修候補を作る。
2. mask周辺のringを半径8で取り、ring内の中央値を計算する。
3. mask内を次の比率で混合する。

```text
out[mask] = inpaint_result * 0.55 + ring_median * 0.45
```

4. 半径1のbox blurを3回かける。
5. mask外contextは元ROIへ戻す。

特徴:

- 白点・黒点を強めに消しやすい。
- 周囲の平均的な色へ寄せるため、細部は失われやすい。
- guard処理で明るい正常領域に暗い染みを作るケースを抑制している。

### wide_scratch

長い傷や幅のある欠陥向けのmethodです。

まずmask領域のbboxを見て、主方向を決めます。

```text
height >= width: vertical defectとみなし、縦方向を主軸
height < width : horizontal defectとみなし、横方向を主軸
```

その後、主軸方向と副軸方向でspan fillを行います。

span fillでは、同じ行または列のmask連続区間について、左右または上下の最も近い非mask画素を探します。

- 両側にcontextがある場合: 線形補間
- 片側だけある場合: 片側の値で埋める
- 両側にない場合: 未補修として残す

主軸方向の結果を優先し、副軸方向は主軸で埋まらなかったところに使います。どちらでも埋まらない部分があれば `inpaint` にfallbackします。

最後に半径1のbox blurを20%だけ混ぜます。

```text
out[mask] = out * 0.8 + smoothed * 0.2
```

特徴:

- フィルム傷のような長い欠陥に向く。
- グラデーションをまたぐ傷ではmedianより自然になりやすい。
- 複雑な模様や斜めの構造には限界がある。

## guard処理

すべてのmethodの最後に `_guard_repair_candidate()` を通します。

guardの目的は、補修候補を周辺contextの色分布から見て極端な値にしないことです。補正対象内の元画素は、埃や傷の色で汚染されている前提なので、候補評価やfallbackの参照値として信用しません。

処理内容:

1. mask周辺のcontext ringを半径10で取得する。
2. ringが空ならmask外contextを使う。
3. contextの分位点、中央値、MADを計算する。
4. candidateのmask内画素をcontext由来の範囲へclipする。

```text
lower = percentile(context, 3%)  - max(context_mad * 3.0, 0.03)
upper = percentile(context, 97%) + max(context_mad * 3.0, 0.03)
candidate[repair_mask] = clip(candidate[repair_mask], lower, upper)
```

clipされたcore画素は、必要に応じて `_guard_core_fallback_candidate()` により局所平面またはcontext中央値から作った候補へ差し替えます。fallbackできない場合でも、元のmask内画素へ戻さず、contextでclip済みのcandidateを維持します。

これにより、補正対象内の埃色を再利用せず、強めの補修methodでもcontextから外れた色を出しにくくします。

## 合成

各ROIで作成した補修候補は、すぐに最終出力へ確定されるわけではありません。

最終的には画像全体でsoft maskとstrengthを使って合成します。

```text
alpha = clip(soft_mask * strength, 0.0, 1.0)
output = original * (1 - alpha) + repair_candidate * alpha
```

`alpha <= 0` の画素は、最後に必ず元画像の値へ戻します。

```text
output[alpha <= 0] = original[alpha <= 0]
```

このため、metrics上もマスク外差分が `0.0` になることをテストで確認しています。

## JPEG補正経路でのマスク合わせ

白埃検出UIでは、検査画像から生成したマスクと、補正対象JPEGの解像度や縦横比が一致しない場合があります。

この場合、`server.py` の `_fit_mask_to_image()` で補正対象JPEGに合わせます。

処理内容:

1. maskがRGBならRGB最大値で2D化する。
2. maskとtargetのaspect ratioを比較する。
3. 差が0.5%以内ならそのままresizeする。
4. 差が0.5%を超える場合は、maskを中心基準でtarget aspectへcropする。
5. nearest-neighborでtarget解像度へresizeする。
6. `> 0` を白、その他を黒として二値化する。

nearest-neighborを使う理由は、マスクの境界値を中途半端に増やさず、白黒マスクとして扱うためです。

この処理は位置合わせを行うものではありません。中心基準のaspect fitです。撮影位置、回転、パースが大きく違う場合は、手動オフセットや特徴点ベースのregistrationが別途必要です。

## デフォルト設定

### コア/CLI既定値

`RepairConfig` とCLIの基本既定値は次の通りです。

| 項目 | 既定値 |
| --- | --- |
| `method` | `hybrid` |
| `mask_channel` | `auto` |
| `threshold` | `0.5` |
| `dilate_radius` | `2` |
| `feather_radius` | `2` |
| `strength` | `1.0` |
| `min_component_area` | `1` |
| `max_component_area` | `5000` |
| `padding` | `16` |

赤照明補正CLIでは、maskは生成済みgrayscaleとして扱うため `mask_channel=grayscale` が既定です。

### Web API既定値

Web APIはUI用途に合わせて、コア既定値と少し違います。

通常補正 `/api/repair` と赤照明補正 `/api/repair-red`:

| 項目 | 既定値 |
| --- | --- |
| `method` | `aggressive` |
| `mask_channel` | `grayscale` |
| `threshold` | `0.5` |
| `dilate_radius` | `1` |
| `feather_radius` | `1` |
| `strength` | `1.0` |
| `min_component_area` | `1` |
| `max_component_area` | `200000` |
| `padding` | `32` |

白埃補正 `/api/repair-white-dust`:

| 項目 | 既定値 |
| --- | --- |
| `method` | `kl` |
| `mask_channel` | `grayscale` |
| `threshold` | `0.5` |
| `dilate_radius` | `2` |
| `feather_radius` | `1` |
| `strength` | `1.0` |
| `min_component_area` | `1` |
| `max_component_area` | `200000` |
| `padding` | `32` |

## metrics

補正後には次のmetricsを出します。

| 項目 | 内容 |
| --- | --- |
| `changed_pixel_count` | 元画像から値が変わった画素数 |
| `changed_bbox_count` | soft mask領域のcomponent数 |
| `max_abs_diff_outside_mask` | マスク外の最大差分 |
| `mean_abs_diff_inside_mask` | マスク内の平均差分 |
| `mean_abs_diff_outside_mask` | マスク外の平均差分 |
| `processing_time_ms` | 補正処理時間 |
| `mask_channel_used` | 実際に使ったマスクチャンネル |
| `kept_component_count` | 補修対象として残ったcomponent数 |
| `removed_small_component_count` | 小さすぎて除外したcomponent数 |
| `removed_large_component_count` | 大きすぎて除外したcomponent数 |
| `changed_pixel_count_core` | core内で変化した画素数 |
| `changed_pixel_count_shell` | shell内で変化した画素数 |
| `mean_abs_diff_core` | core内の平均差分 |
| `mean_abs_diff_shell` | shell内の平均差分 |
| `max_abs_diff_outside_original_mask` | core外の最大差分。shellの弱い変化も含む |
| `max_abs_diff_outside_repair_mask` | repair mask外の最大差分 |
| `guard_rejected_pixel_count` | guardでclipまたは元画素保持された画素数 |
| `average_component_alpha` | component単位のalpha scale平均 |
| `low_confidence_component_count` | seam scoreや大面積fallbackで低信頼扱いになったcomponent数 |

特に `max_abs_diff_outside_mask` は、マスク外を壊していないか確認する重要な値です。

## debug artifact

`debug_dir` を指定した場合、次のファイルが出力されます。

| ファイル | 内容 |
| --- | --- |
| `normalized_mask.png` | 正規化後マスク |
| `binary_mask.png` | threshold/filter/dilate後の二値マスク |
| `soft_mask.png` | feather後のsoft mask |
| `core_mask.png` | threshold/filter後の中心欠陥マスク |
| `repair_mask.png` | candidate生成用unknownマスク |
| `blend_alpha.png` | 最終合成用alpha |
| `candidate_before_guard.png` | guard適用前の補修候補 |
| `candidate_after_guard.png` | guard適用後の補修候補 |
| `rejected_by_guard.png` | guardで抑制された画素 |
| `shell_mask.png` | repair maskからcoreを除いた弱合成領域 |
| `repaired_preview.png` | 補修結果 |
| `diff_visualization.png` | 赤=差分、緑=soft maskの可視化 |
| `metrics.json` | metrics |

## method選択の目安

| method | 向いているケース | 注意点 |
| --- | --- | --- |
| `kl` | 白埃検査画像から生成したマスクの既定補修、周辺色分布に合わせたい点状欠陥 | 構造復元より色分布整合を優先する |
| `linear` | グラデーション上の小から中程度の欠陥、分布補正をかけたくない場合 | テクスチャ再現力は限定的 |
| `defect_aware` | 小さい点ゴミ、普通の埃、細長い線、繰り返し模様上の小中欠陥、粒状感のある背景 | GUI高速モードではlinear系fallback |
| `adaptive` | 一般的な点状埃、グラデーション背景、斜めの細い傷 | 複雑な周期テクスチャの再構成はできない |
| `hybrid` | 既存指定との互換 | 内部はadaptive寄りで旧hybridと完全同一ではない |
| `inpaint` | 小さい点状欠陥 | 大面積ではぼけやすい |
| `median` | 均一背景の大きめ欠陥 | グラデーションや模様では平坦になる |
| `aggressive` | 強い白点・黒点除去 | 細部が失われやすい |
| `wide_scratch` | 長い縦傷・横傷 | 複雑な模様や斜め傷には弱い |
| `denoise` | 軽いざらつき・弱い補正 | 明確な欠陥除去力は低い |

## 現在の制約

- 補正はマスクガイド型であり、マスクが外れている箇所は原則として補正しない。
- 白埃Web経路のマスク合わせは中心基準のaspect fitであり、画像内容に基づく位置合わせではない。
- 斜め傷や複雑なテクスチャの再構成は得意ではない。
- JPEG出力は品質95で保存するため、出力時にJPEG再圧縮が入る。
- PNG/TIFFの16bit補正は内部的には維持できるが、JPEG補正対象は8bit RGBとして読み込まれる。
- 補修候補の生成はCPU上のNumPy実装であり、GPU処理やタイルキャッシュはまだない。

## 関連テスト

主な補正テストは次のファイルにあります。

- `tests/test_repair.py`
- `tests/test_workflow.py`
- `tests/test_server.py`

確認している代表的な性質:

- 空マスクでは入力を完全に維持する。
- `strength=0` では入力を完全に維持する。
- マスク外の画素を変更しない。
- 小さい白埃・黒埃を補修できる。
- `aggressive` が明るい正常領域に暗い染みを作りにくい。
- `wide_scratch` が縦横の広い欠陥を補間できる。
- `uint16` 入力ではdtypeを維持する。
- RAW検査画像から生成したマスクをJPEG補正対象へ投影して補修できる。
- PNG補正対象は現在の白埃補正Web経路では拒否する。
