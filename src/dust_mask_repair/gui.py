from __future__ import annotations

import json
import queue
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .benchmark import make_defect_aware_quality_case
from .config import RepairConfig
from .frequency_repair import normalize_frequency_scope_mask, prepare_frequency_selection
from .io import RAW_EXTENSIONS, as_float32, read_image, restore_dtype, write_image
from .mask import dilate_mask
from .repair import repair_image
from .white_dust import WhiteDustConfig, detect_white_dust_source_image


GUI_VERSION = "2.0"
GUI_METHODS = ("kl", "linear", "defect_aware")
GUI_IMAGE_PATTERN = "*.png *.jpg *.jpeg *.tif *.tiff " + " ".join(f"*{suffix}" for suffix in sorted(RAW_EXTENSIONS))
GUI_PREVIEW_LONG_EDGE = 1600
GUI_MAX_COMPONENT_AREA = 200000
GUI_FAST_MASK_PIXEL_THRESHOLD = 100000
GUI_DEFAULT_REPAIR_EXPAND_RADIUS = 1
GUI_DEFAULT_FEATHER_RADIUS = 2
GUI_DEFAULT_COLOR_MATCH_STRENGTH = 0.65
GUI_DEFAULT_GRAIN_STRENGTH = 0.45
GUI_EDGE_GUIDED_TEST_CASE = "diagonal_edge_micro_dust"
GUI_EDGE_GUIDED_TEST_WIDTH = 96
GUI_EDGE_GUIDED_TEST_HEIGHT = 72
_FAST_CONTEXT_SAMPLE_LIMIT = 500000


@dataclass(frozen=True)
class GuiRepairJobResult:
    run_dir: Path
    repaired_path: Path
    target_preview_path: Path
    mask_path: Path
    overlay_path: Path
    score_path: Path
    manifest_path: Path
    metrics_path: Path
    manifest: dict[str, Any]


def run_white_dust_gui_job(
    *,
    target_path: str | Path,
    inspection_path: str | Path,
    output_dir: str | Path,
    method: str = "kl",
    detection_long_edge: int = 1024,
    threshold_sensitivity: float = 1.0,
    repair_expand_radius: int = 0,
    feather_radius: int = 0,
    color_match_strength: float = 0.0,
    grain_strength: float = 0.25,
    frequency_scope_mask_path: str | Path | None = None,
) -> GuiRepairJobResult:
    if method not in GUI_METHODS:
        raise ValueError(f"GUI repair method must be one of {GUI_METHODS}, got: {method}")
    if repair_expand_radius < 0:
        raise ValueError("repair_expand_radius must be >= 0")
    if feather_radius < 0:
        raise ValueError("feather_radius must be >= 0")
    if not 0.0 <= color_match_strength <= 1.0:
        raise ValueError("color_match_strength must be in the range 0.0..1.0")
    if grain_strength < 0.0:
        raise ValueError("grain_strength must be >= 0")

    target_file = Path(target_path)
    inspection_file = Path(inspection_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    frequency_scope_file = Path(frequency_scope_mask_path) if frequency_scope_mask_path else None

    target = _read_gui_image(target_file, role="補正対象写真", raw_output_bps=16)
    inspection = _read_gui_image(inspection_file, role="マスク作成用写真", raw_output_bps=8)
    if target.pixels.shape[:2] != inspection.pixels.shape[:2]:
        raise ValueError(
            "補正対象写真とマスク作成用写真の画像サイズが一致していません。\n"
            f"補正対象写真: {target.pixels.shape[1]}x{target.pixels.shape[0]}\n"
            f"マスク作成用写真: {inspection.pixels.shape[1]}x{inspection.pixels.shape[0]}\n"
            "同じ撮影条件・同じ解像度の画像を指定してください。"
        )
    frequency_scope_pixels = _read_frequency_scope_pixels(frequency_scope_file, target.pixels.shape[:2])

    run_dir = _unique_run_dir(output_root, target_file.stem, method)
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "processing_status.json"

    output_suffix = _output_suffix_for_target(target_file)
    repaired_path = run_dir / f"{target_file.stem}_repaired_{method}{output_suffix}"
    target_preview_path = run_dir / "target_preview.png"
    mask_path = run_dir / "generated_mask.png"
    repair_mask_path = run_dir / "repair_mask_expanded.png"
    overlay_path = run_dir / "inspection_overlay.png"
    score_path = run_dir / "white_dust_score.png"
    frequency_scope_path = run_dir / "frequency_scope_mask.png"
    frequency_selected_path = run_dir / "frequency_selected_core_mask.png"
    frequency_overlay_path = run_dir / "frequency_selected_overlay.png"
    manifest_path = run_dir / "repair_result.json"
    metrics_path = run_dir / "repair_metrics.json"

    _write_processing_status(
        status_path,
        phase="started",
        target_file=target_file,
        inspection_file=inspection_file,
        method=method,
    )
    try:
        write_image(target_preview_path, _preview_rgb_u8(target.pixels))
        _write_processing_status(
            status_path,
            phase="detecting_mask",
            target_file=target_file,
            inspection_file=inspection_file,
            method=method,
        )

        white_cfg = WhiteDustConfig(
            detection_long_edge=int(detection_long_edge),
            threshold_sensitivity=float(threshold_sensitivity),
            background_mode="dark",
            mask_edge_mode="normal",
            visual_artifacts=False,
        )
        white_result = detect_white_dust_source_image(inspection.pixels, white_cfg, source_overlay=False)
        write_image(mask_path, white_result.mask)
        write_image(repair_mask_path, _expanded_repair_mask_preview(white_result.mask, repair_expand_radius))
        write_image(overlay_path, _mask_overlay_preview(inspection.pixels, white_result.mask))
        write_image(score_path, white_result.score_map)
        frequency_selection = None
        if frequency_scope_pixels is not None:
            scope_bool = normalize_frequency_scope_mask(frequency_scope_pixels, target.pixels.shape[:2])
            scope_cfg = RepairConfig(method="defect_aware", frequency_guided_enabled=True)
            frequency_selection = prepare_frequency_selection(white_result.mask > 0, scope_bool, scope_cfg)
            write_image(frequency_scope_path, scope_bool.astype(np.uint8) * 255)
            write_image(frequency_selected_path, frequency_selection.selected_core_mask.astype(np.uint8) * 255)
            write_image(frequency_overlay_path, _mask_overlay_preview(target.pixels, frequency_selection.selected_core_mask.astype(np.uint8) * 255))

        mask_pixels = int(np.count_nonzero(white_result.mask))
        if mask_pixels > GUI_FAST_MASK_PIXEL_THRESHOLD:
            _write_processing_status(
                status_path,
                phase="repairing_fast",
                target_file=target_file,
                inspection_file=inspection_file,
                method=method,
                mask_pixels=mask_pixels,
                fast_threshold=GUI_FAST_MASK_PIXEL_THRESHOLD,
            )
            repaired_image, repair_metrics = _fast_gui_mask_repair(
                target.pixels,
                white_result.mask,
                method,
                repair_expand_radius=repair_expand_radius,
                feather_radius=feather_radius,
                color_match_strength=color_match_strength,
                grain_strength=grain_strength,
                frequency_scope_mask=frequency_scope_pixels,
            )
        else:
            _write_processing_status(
                status_path,
                phase="repairing",
                target_file=target_file,
                inspection_file=inspection_file,
                method=method,
                mask_pixels=mask_pixels,
            )
            repair_cfg = RepairConfig(
                method=method,
                mask_channel="grayscale",
                threshold=0.5,
                dilate_radius=int(repair_expand_radius),
                feather_radius=int(feather_radius),
                strength=1.0,
                padding=16,
                max_component_area=GUI_MAX_COMPONENT_AREA,
                grain_reinject_strength=float(grain_strength),
                color_match_strength=float(color_match_strength),
                frequency_guided_enabled=frequency_scope_pixels is not None,
            )
            repair_result = repair_image(target.pixels, white_result.mask, repair_cfg, frequency_scope_mask=frequency_scope_pixels)
            repaired_image = repair_result.repaired_image
            repair_metrics = repair_result.metrics
        write_image(repaired_path, repaired_image)
    except Exception as exc:
        _write_processing_status(
            status_path,
            phase="failed",
            target_file=target_file,
            inspection_file=inspection_file,
            method=method,
            error=str(exc),
        )
        _write_error_log(run_dir / "error.txt", exc)
        raise

    repair_options = {
        "mask_expand_radius": int(repair_expand_radius),
        "feather_radius": int(feather_radius),
        "color_match_strength": float(color_match_strength),
        "grain_strength": float(grain_strength),
    }
    if frequency_scope_file is not None:
        repair_options["frequency_scope_mask"] = str(frequency_scope_file)

    manifest = _build_manifest(
        target_file=target_file,
        inspection_file=inspection_file,
        method=method,
        white_manifest=white_result.manifest,
        repair_metrics=repair_metrics,
        repaired_path=repaired_path,
        mask_path=mask_path,
        repair_mask_path=repair_mask_path,
        repair_options=repair_options,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(repair_metrics, indent=2, sort_keys=True), encoding="utf-8")
    _write_processing_status(
        status_path,
        phase="complete",
        target_file=target_file,
        inspection_file=inspection_file,
        method=method,
        repaired_path=repaired_path,
        mask_path=mask_path,
        repair_mask_path=repair_mask_path,
        repair_options=repair_options,
    )

    return GuiRepairJobResult(
        run_dir=run_dir,
        repaired_path=repaired_path,
        target_preview_path=target_preview_path,
        mask_path=mask_path,
        overlay_path=overlay_path,
        score_path=score_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        manifest=manifest,
    )



def run_edge_guided_gui_test(
    *,
    output_dir: str | Path,
    case: str = GUI_EDGE_GUIDED_TEST_CASE,
) -> GuiRepairJobResult:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = _unique_run_dir(output_root, f"edge_guided_test_{case}", "defect_aware")
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "processing_status.json"

    clean_path = run_dir / "edge_guided_clean_answer.png"
    damaged_path = run_dir / "edge_guided_damaged_input.png"
    repaired_path = run_dir / "edge_guided_repaired_defect_aware.png"
    disabled_path = run_dir / "edge_guided_disabled_comparison.png"
    mask_path = run_dir / "edge_guided_mask.png"
    comparison_path = run_dir / "edge_guided_comparison.png"
    diff_path = run_dir / "edge_guided_error_heatmap.png"
    manifest_path = run_dir / "edge_guided_test_result.json"
    metrics_path = run_dir / "edge_guided_test_metrics.json"
    debug_dir = run_dir / "debug"

    _write_processing_status(status_path, phase="started_edge_guided_test", case=case, method="defect_aware")
    try:
        clean, damaged, mask = make_defect_aware_quality_case(
            case,
            width=GUI_EDGE_GUIDED_TEST_WIDTH,
            height=GUI_EDGE_GUIDED_TEST_HEIGHT,
        )
        write_image(clean_path, clean)
        write_image(damaged_path, damaged)
        write_image(mask_path, mask)

        common_config = {
            "method": "defect_aware",
            "mask_channel": "grayscale",
            "dilate_radius": 0,
            "feather_radius": 0,
            "padding": 18,
            "max_component_area": 5000,
            "grain_reinject_strength": 0.0,
        }
        _write_processing_status(status_path, phase="repairing_edge_guided_test", case=case, method="defect_aware")
        result = repair_image(damaged, mask, RepairConfig(**common_config, debug_dir=debug_dir))
        disabled = repair_image(
            damaged,
            mask,
            RepairConfig(**common_config, edge_guided_enabled=False),
        )

        write_image(repaired_path, result.repaired_image)
        write_image(disabled_path, disabled.repaired_image)
        write_image(
            comparison_path,
            _edge_guided_test_comparison_preview(damaged, result.repaired_image, disabled.repaired_image, clean),
        )
        write_image(diff_path, _edge_guided_test_diff_preview(result.repaired_image, clean, mask))
    except Exception as exc:
        _write_processing_status(status_path, phase="failed", case=case, method="defect_aware", error=str(exc))
        _write_error_log(run_dir / "error.txt", exc)
        raise

    inside = mask > 0
    edge_guided_mae = _masked_mae_0_255(result.repaired_image, clean, inside)
    edge_guided_disabled_mae = _masked_mae_0_255(disabled.repaired_image, clean, inside)
    corrupted_mae = _masked_mae_0_255(damaged, clean, inside)
    outside_unchanged = float(result.metrics.get("max_abs_diff_outside_mask", 0.0)) == 0.0
    edge_component_count = int(result.metrics.get("small_local_edge_guided_component_count", 0))
    passed = bool(
        edge_component_count > 0
        and edge_guided_mae < edge_guided_disabled_mae
        and edge_guided_mae < corrupted_mae
        and outside_unchanged
    )
    test_metrics = {
        **result.metrics,
        "gui_edge_guided_test": True,
        "gui_edge_guided_test_case": case,
        "edge_guided_mae_0_255": edge_guided_mae,
        "edge_guided_disabled_mae_0_255": edge_guided_disabled_mae,
        "corrupted_mae_0_255": corrupted_mae,
        "edge_guided_test_passed": passed,
    }
    manifest = {
        "実行日": datetime.now().isoformat(timespec="seconds"),
        "version": f"KLComplementary {GUI_VERSION} / dust-mask-repair {_package_version()}",
        "対象写真": str(damaged_path),
        "マスク作成用写真": str(mask_path),
        "補正方法": "defect_aware",
        "テスト": "edge-guided local repair GUI test",
        "テストケース": case,
        "正答率": {
            "edge_guided_mae_0_255": edge_guided_mae,
            "edge_guided_disabled_mae_0_255": edge_guided_disabled_mae,
            "corrupted_mae_0_255": corrupted_mae,
        },
        "合否": passed,
        "補正後画像": str(repaired_path),
        "模範解答": str(clean_path),
        "比較画像": str(comparison_path),
        "誤差ヒートマップ": str(diff_path),
        "repair_metrics": _json_safe(test_metrics),
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(_json_safe(test_metrics), indent=2, sort_keys=True), encoding="utf-8")
    _write_processing_status(
        status_path,
        phase="complete",
        case=case,
        method="defect_aware",
        passed=passed,
        repaired_path=repaired_path,
        comparison_path=comparison_path,
    )

    return GuiRepairJobResult(
        run_dir=run_dir,
        repaired_path=repaired_path,
        target_preview_path=damaged_path,
        mask_path=mask_path,
        overlay_path=comparison_path,
        score_path=diff_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        manifest=manifest,
    )

def main(argv: list[str] | None = None) -> int:
    if argv:
        raise SystemExit("dust-mask-repair-gui does not accept command line arguments.")
    _launch_gui()
    return 0


def _launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from PIL import Image, ImageTk

    class App(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("KLComplementary 2.0 GUI")
            self.minsize(980, 700)
            self._result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
            self._preview_refs: list[Any] = []

            self.target_var = tk.StringVar()
            self.inspection_var = tk.StringVar()
            self.frequency_scope_var = tk.StringVar()
            self.output_var = tk.StringVar(value=str(Path.cwd() / "gui_outputs"))
            self.method_var = tk.StringVar(value="kl")
            self.long_edge_var = tk.StringVar(value="1024")
            self.sensitivity_var = tk.StringVar(value="1.0")
            self.expand_radius_var = tk.StringVar(value=str(GUI_DEFAULT_REPAIR_EXPAND_RADIUS))
            self.feather_radius_var = tk.StringVar(value=str(GUI_DEFAULT_FEATHER_RADIUS))
            self.color_match_var = tk.StringVar(value=str(GUI_DEFAULT_COLOR_MATCH_STRENGTH))
            self.grain_strength_var = tk.StringVar(value=str(GUI_DEFAULT_GRAIN_STRENGTH))
            self.status_var = tk.StringVar(value="待機中")
            self.result_var = tk.StringVar(value="")

            self._build_ui(ttk)
            self.after(100, self._poll_result_queue)

        def _build_ui(self, ttk_module: Any) -> None:
            self.columnconfigure(1, weight=1)
            self.rowconfigure(0, weight=1)

            controls = ttk_module.Frame(self, padding=14)
            controls.grid(row=0, column=0, sticky="nsw")
            controls.columnconfigure(1, weight=1)

            ttk_module.Label(controls, text="補正対象写真").grid(row=0, column=0, sticky="w", pady=(0, 4))
            ttk_module.Entry(controls, textvariable=self.target_var, width=42).grid(row=1, column=0, columnspan=2, sticky="ew")
            ttk_module.Button(controls, text="選択", command=self._choose_target).grid(row=1, column=2, padx=(8, 0))

            ttk_module.Label(controls, text="マスク作成用写真").grid(row=2, column=0, sticky="w", pady=(14, 4))
            ttk_module.Entry(controls, textvariable=self.inspection_var, width=42).grid(row=3, column=0, columnspan=2, sticky="ew")
            ttk_module.Button(controls, text="選択", command=self._choose_inspection).grid(row=3, column=2, padx=(8, 0))

            ttk_module.Label(controls, text="空間周波数補正範囲マスク（任意）").grid(row=4, column=0, sticky="w", pady=(14, 4))
            ttk_module.Entry(controls, textvariable=self.frequency_scope_var, width=42).grid(row=5, column=0, columnspan=2, sticky="ew")
            ttk_module.Button(controls, text="選択", command=self._choose_frequency_scope).grid(row=5, column=2, padx=(8, 0))

            ttk_module.Label(controls, text="出力フォルダ").grid(row=6, column=0, sticky="w", pady=(14, 4))
            ttk_module.Entry(controls, textvariable=self.output_var, width=42).grid(row=7, column=0, columnspan=2, sticky="ew")
            ttk_module.Button(controls, text="選択", command=self._choose_output).grid(row=7, column=2, padx=(8, 0))

            ttk_module.Label(controls, text="補正方式").grid(row=8, column=0, sticky="w", pady=(14, 4))
            method = ttk_module.Combobox(controls, textvariable=self.method_var, values=GUI_METHODS, state="readonly", width=14)
            method.grid(row=9, column=0, sticky="w")

            ttk_module.Label(controls, text="検出 long edge").grid(row=10, column=0, sticky="w", pady=(14, 4))
            ttk_module.Entry(controls, textvariable=self.long_edge_var, width=10).grid(row=11, column=0, sticky="w")

            ttk_module.Label(controls, text="検出感度").grid(row=10, column=1, sticky="w", pady=(14, 4))
            ttk_module.Entry(controls, textvariable=self.sensitivity_var, width=10).grid(row=11, column=1, sticky="w")

            ttk_module.Label(controls, text="周辺補正 px").grid(row=12, column=0, sticky="w", pady=(14, 4))
            ttk_module.Entry(controls, textvariable=self.expand_radius_var, width=10).grid(row=13, column=0, sticky="w")

            ttk_module.Label(controls, text="境界なじませ px").grid(row=12, column=1, sticky="w", pady=(14, 4))
            ttk_module.Entry(controls, textvariable=self.feather_radius_var, width=10).grid(row=13, column=1, sticky="w")

            ttk_module.Label(controls, text="色なじませ強度").grid(row=14, column=0, sticky="w", pady=(14, 4))
            ttk_module.Entry(controls, textvariable=self.color_match_var, width=10).grid(row=15, column=0, sticky="w")

            ttk_module.Label(controls, text="粒状感強度").grid(row=14, column=1, sticky="w", pady=(14, 4))
            ttk_module.Entry(controls, textvariable=self.grain_strength_var, width=10).grid(row=15, column=1, sticky="w")

            self.run_button = ttk_module.Button(controls, text="補正実行", command=self._run)
            self.run_button.grid(row=16, column=0, columnspan=3, sticky="ew", pady=(22, 8))

            self.test_button = ttk_module.Button(controls, text="edge-guidedテスト", command=self._run_edge_guided_test)
            self.test_button.grid(row=17, column=0, columnspan=3, sticky="ew", pady=(0, 8))

            ttk_module.Label(controls, textvariable=self.status_var, foreground="#0f5964").grid(
                row=18, column=0, columnspan=3, sticky="w"
            )
            ttk_module.Label(controls, textvariable=self.result_var, wraplength=360).grid(
                row=19, column=0, columnspan=3, sticky="w", pady=(8, 0)
            )

            preview = ttk_module.Frame(self, padding=(0, 14, 14, 14))
            preview.grid(row=0, column=1, sticky="nsew")
            preview.columnconfigure(0, weight=1)
            preview.columnconfigure(1, weight=1)
            preview.rowconfigure(1, weight=1)
            preview.rowconfigure(3, weight=1)

            ttk_module.Label(preview, text="補正前").grid(row=0, column=0, sticky="w")
            ttk_module.Label(preview, text="補正後").grid(row=0, column=1, sticky="w")
            self.before_label = ttk_module.Label(preview, anchor="center", background="#101820")
            self.after_label = ttk_module.Label(preview, anchor="center", background="#101820")
            self.before_label.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(4, 12))
            self.after_label.grid(row=1, column=1, sticky="nsew", pady=(4, 12))

            ttk_module.Label(preview, text="生成マスク").grid(row=2, column=0, sticky="w")
            ttk_module.Label(preview, text="検査画像上の検出").grid(row=2, column=1, sticky="w")
            self.mask_label = ttk_module.Label(preview, anchor="center", background="#101820")
            self.overlay_label = ttk_module.Label(preview, anchor="center", background="#101820")
            self.mask_label.grid(row=3, column=0, sticky="nsew", padx=(0, 8), pady=(4, 0))
            self.overlay_label.grid(row=3, column=1, sticky="nsew", pady=(4, 0))

        def _choose_target(self) -> None:
            path = filedialog.askopenfilename(
                title="補正対象写真を選択",
                filetypes=[
                    ("Image files", GUI_IMAGE_PATTERN),
                    ("All files", "*.*"),
                ],
            )
            if path:
                self.target_var.set(path)

        def _choose_inspection(self) -> None:
            path = filedialog.askopenfilename(
                title="マスク作成用写真を選択",
                filetypes=[
                    ("Image files", GUI_IMAGE_PATTERN),
                    ("All files", "*.*"),
                ],
            )
            if path:
                self.inspection_var.set(path)

        def _choose_frequency_scope(self) -> None:
            path = filedialog.askopenfilename(
                title="空間周波数補正範囲マスクを選択",
                filetypes=[
                    ("Mask image files", "*.png *.jpg *.jpeg *.tif *.tiff"),
                    ("All files", "*.*"),
                ],
            )
            if path:
                self.frequency_scope_var.set(path)

        def _choose_output(self) -> None:
            path = filedialog.askdirectory(title="出力フォルダを選択")
            if path:
                self.output_var.set(path)

        def _run(self) -> None:
            try:
                target = Path(self.target_var.get())
                inspection = Path(self.inspection_var.get())
                output = Path(self.output_var.get())
                frequency_scope = Path(self.frequency_scope_var.get()) if self.frequency_scope_var.get().strip() else None
                if not target.is_file():
                    raise ValueError("補正対象写真を選択してください。")
                if not inspection.is_file():
                    raise ValueError("マスク作成用写真を選択してください。")
                if frequency_scope is not None and not frequency_scope.is_file():
                    raise ValueError("空間周波数補正範囲マスクが見つかりません。")
                long_edge = int(self.long_edge_var.get())
                sensitivity = float(self.sensitivity_var.get())
                expand_radius = int(self.expand_radius_var.get())
                feather_radius = int(self.feather_radius_var.get())
                color_match = float(self.color_match_var.get())
                grain_strength = float(self.grain_strength_var.get())
                if long_edge <= 0:
                    raise ValueError("検出 long edge は1以上にしてください。")
                if sensitivity <= 0.0:
                    raise ValueError("検出感度は0より大きい値にしてください。")
                if expand_radius < 0:
                    raise ValueError("周辺補正 px は0以上にしてください。")
                if feather_radius < 0:
                    raise ValueError("境界なじませ px は0以上にしてください。")
                if not 0.0 <= color_match <= 1.0:
                    raise ValueError("色なじませ強度は0.0から1.0にしてください。")
                if grain_strength < 0.0:
                    raise ValueError("粒状感強度は0以上にしてください。")
            except Exception as exc:  # noqa: BLE001 - show validation errors in the GUI.
                messagebox.showerror("入力エラー", str(exc))
                return

            self.run_button.configure(state="disabled")
            self.test_button.configure(state="disabled")
            self.status_var.set("処理中")
            self.result_var.set("")

            worker = threading.Thread(
                target=self._run_worker,
                kwargs={
                    "target": target,
                    "inspection": inspection,
                    "output": output,
                    "method": self.method_var.get(),
                    "long_edge": long_edge,
                    "sensitivity": sensitivity,
                    "expand_radius": expand_radius,
                    "feather_radius": feather_radius,
                    "color_match": color_match,
                    "grain_strength": grain_strength,
                    "frequency_scope": frequency_scope,
                },
                daemon=True,
            )
            worker.start()

        def _run_edge_guided_test(self) -> None:
            try:
                output = Path(self.output_var.get())
                if not str(output):
                    raise ValueError("出力フォルダを指定してください。")
            except Exception as exc:  # noqa: BLE001 - show validation errors in the GUI.
                messagebox.showerror("入力エラー", str(exc))
                return

            self.run_button.configure(state="disabled")
            self.test_button.configure(state="disabled")
            self.status_var.set("edge-guidedテスト中")
            self.result_var.set("")

            worker = threading.Thread(
                target=self._run_edge_guided_test_worker,
                kwargs={"output": output},
                daemon=True,
            )
            worker.start()

        def _run_worker(
            self,
            *,
            target: Path,
            inspection: Path,
            output: Path,
            method: str,
            long_edge: int,
            sensitivity: float,
            expand_radius: int,
            feather_radius: int,
            color_match: float,
            grain_strength: float,
            frequency_scope: Path | None,
        ) -> None:
            try:
                result = run_white_dust_gui_job(
                    target_path=target,
                    inspection_path=inspection,
                    output_dir=output,
                    method=method,
                    detection_long_edge=long_edge,
                    threshold_sensitivity=sensitivity,
                    repair_expand_radius=expand_radius,
                    feather_radius=feather_radius,
                    color_match_strength=color_match,
                    grain_strength=grain_strength,
                    frequency_scope_mask_path=frequency_scope,
                )
            except Exception as exc:  # noqa: BLE001 - return errors to the GUI.
                self._result_queue.put(("error", _format_gui_error(exc, output)))
                return
            self._result_queue.put(("done", result))

        def _run_edge_guided_test_worker(self, *, output: Path) -> None:
            try:
                result = run_edge_guided_gui_test(output_dir=output)
            except Exception as exc:  # noqa: BLE001 - return errors to the GUI.
                self._result_queue.put(("error", _format_gui_error(exc, output)))
                return
            self._result_queue.put(("test_done", result))

        def _poll_result_queue(self) -> None:
            try:
                kind, payload = self._result_queue.get_nowait()
            except queue.Empty:
                self.after(100, self._poll_result_queue)
                return

            self.run_button.configure(state="normal")
            self.test_button.configure(state="normal")
            if kind == "error":
                self.status_var.set("エラー")
                messagebox.showerror("補正エラー", str(payload))
            elif kind == "test_done":
                result = payload
                assert isinstance(result, GuiRepairJobResult)
                passed = "PASS" if result.manifest.get("合否") else "FAIL"
                accuracy = result.manifest.get("正答率", {})
                self.status_var.set("テスト完了")
                self.result_var.set(
                    "edge-guidedテスト: "
                    f"{passed}\n出力: {result.run_dir}\n"
                    f"MAE: {accuracy.get('edge_guided_mae_0_255', 'n/a')}"
                )
                self._show_result(result)
            else:
                result = payload
                assert isinstance(result, GuiRepairJobResult)
                self.status_var.set("完了")
                self.result_var.set(f"出力: {result.run_dir}")
                self._show_result(result)
            self.after(100, self._poll_result_queue)

        def _show_result(self, result: GuiRepairJobResult) -> None:
            self._preview_refs = [
                self._set_preview(self.before_label, result.target_preview_path, Image, ImageTk),
                self._set_preview(self.after_label, result.repaired_path, Image, ImageTk),
                self._set_preview(self.mask_label, result.mask_path, Image, ImageTk),
                self._set_preview(self.overlay_label, result.overlay_path, Image, ImageTk),
            ]

        def _set_preview(self, label: Any, path: Path, image_module: Any, imagetk_module: Any) -> Any:
            try:
                with image_module.open(path) as image:
                    preview = image.convert("RGB")
                    preview.thumbnail((420, 250))
                    photo = imagetk_module.PhotoImage(preview)
            except Exception:
                label.configure(image="", text=f"プレビュー不可\n{path.name}", foreground="#ffffff")
                return None
            label.configure(image=photo, text="")
            return photo

    app = App()
    app.mainloop()


def _unique_run_dir(output_root: Path, target_stem: str, method: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"{target_stem}_{method}_{stamp}"
    if not base.exists():
        return base
    for index in range(2, 1000):
        candidate = output_root / f"{target_stem}_{method}_{stamp}_{index}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate a unique GUI output directory.")


def _read_gui_image(path: Path, *, role: str, raw_output_bps: int) -> Any:
    image_path = Path(path)
    if not image_path.is_file():
        raise ValueError(f"{role}が見つかりません: {image_path}")

    try:
        if image_path.suffix.lower() in RAW_EXTENSIONS:
            return read_image(image_path, raw_half_size=False, raw_output_bps=raw_output_bps)
        return read_image(image_path)
    except ValueError as exc:
        message = str(exc)
        if "rawpy" in message.lower():
            raise ValueError(
                f"{role}のRAW/ARW読み込みに失敗しました。\n"
                "ARWを使うには rawpy が必要です。PowerShellで次を実行してください:\n"
                "py -3.12 -m pip install -e .[raw]\n\n"
                f"対象ファイル: {image_path}"
            ) from exc
        raise ValueError(f"{role}の読み込みに失敗しました。\n対象ファイル: {image_path}\n詳細: {message}") from exc



def _read_frequency_scope_pixels(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    if not path.is_file():
        raise ValueError(f"空間周波数補正範囲マスクが見つかりません: {path}")
    try:
        pixels = read_image(path).pixels
    except ValueError as exc:
        raise ValueError(
            f"空間周波数補正範囲マスクの読み込みに失敗しました。\n"
            f"対象ファイル: {path}\n"
            f"詳細: {exc}"
        ) from exc
    normalize_frequency_scope_mask(pixels, shape)
    return pixels

def _output_suffix_for_target(target_file: Path) -> str:
    suffix = target_file.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return ".jpg"
    return ".png"


def _preview_rgb_u8(pixels: np.ndarray, max_long_edge: int = GUI_PREVIEW_LONG_EDGE) -> np.ndarray:
    arr = np.asarray(pixels)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Unsupported preview image shape: {arr.shape}")
    rgb = arr[:, :, :3]
    if rgb.dtype == np.uint8:
        preview = np.ascontiguousarray(rgb)
    else:
        preview = np.rint(as_float32(rgb) * 255.0).astype(np.uint8)
    return _resize_rgb_preview(preview, max_long_edge)


def _mask_overlay_preview(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = _preview_rgb_u8(pixels)
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = np.max(mask_arr[:, :, :3], axis=2)
    if mask_arr.ndim != 2:
        raise ValueError(f"Unsupported mask shape for overlay preview: {mask_arr.shape}")
    if mask_arr.shape != np.asarray(pixels).shape[:2]:
        raise ValueError(
            "image and mask dimensions differ for overlay preview: "
            f"image={np.asarray(pixels).shape[1]}x{np.asarray(pixels).shape[0]}, "
            f"mask={mask_arr.shape[1]}x{mask_arr.shape[0]}"
        )

    mask_preview = np.asarray(
        Image.fromarray(np.where(mask_arr > 0, 255, 0).astype(np.uint8)).resize(
            (preview.shape[1], preview.shape[0]),
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )
    active = mask_preview > 0
    overlay = preview.copy()
    if active.any():
        color = np.array([255.0, 32.0, 16.0], dtype=np.float32)
        alpha = 0.68
        blended = overlay[active].astype(np.float32) * (1.0 - alpha) + color * alpha
        overlay[active] = np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)
    return overlay




def _expanded_repair_mask_preview(mask: np.ndarray, expand_radius: int) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = np.max(mask_arr[:, :, :3], axis=2)
    if mask_arr.ndim != 2:
        raise ValueError(f"Unsupported mask shape for repair mask preview: {mask_arr.shape}")
    mask_bool = mask_arr > 0
    if expand_radius > 0:
        mask_bool = dilate_mask(mask_bool, int(expand_radius))
    return (mask_bool.astype(np.uint8) * 255)
def _edge_guided_test_comparison_preview(
    damaged: np.ndarray,
    repaired: np.ndarray,
    disabled: np.ndarray,
    clean: np.ndarray,
) -> np.ndarray:
    panels = [_preview_rgb_u8(item, max_long_edge=480) for item in (damaged, repaired, disabled, clean)]
    height = max(panel.shape[0] for panel in panels)
    padded: list[np.ndarray] = []
    for panel in panels:
        if panel.shape[0] < height:
            pad = np.zeros((height - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.concatenate([panel, pad], axis=0)
        padded.append(panel)
    gutter = np.full((height, 4, 3), 255, dtype=np.uint8)
    return np.concatenate([padded[0], gutter, padded[1], gutter, padded[2], gutter, padded[3]], axis=1)


def _edge_guided_test_diff_preview(image: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_float = as_float32(np.asarray(image))[:, :, :3]
    reference_float = as_float32(np.asarray(reference))[:, :, :3]
    diff = np.max(np.abs(image_float - reference_float), axis=2)
    heat = np.clip(diff * 8.0, 0.0, 1.0)
    mask_bool = np.asarray(mask) > 0
    if mask_bool.ndim == 3:
        mask_bool = np.max(mask_bool[:, :, :3], axis=2)
    preview = np.zeros((*diff.shape, 3), dtype=np.uint8)
    preview[:, :, 0] = np.rint(heat * 255.0).astype(np.uint8)
    preview[:, :, 1] = np.where(mask_bool, 160, 0).astype(np.uint8)
    preview[:, :, 2] = np.rint((1.0 - heat) * 80.0).astype(np.uint8)
    return preview


def _masked_mae_0_255(image: np.ndarray, reference: np.ndarray, mask_bool: np.ndarray) -> float:
    if not np.any(mask_bool):
        return 0.0
    image_float = as_float32(np.asarray(image))[:, :, :3]
    reference_float = as_float32(np.asarray(reference))[:, :, :3]
    return float(np.mean(np.abs(image_float[mask_bool] - reference_float[mask_bool])) * 255.0)

def _resize_rgb_preview(rgb: np.ndarray, max_long_edge: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return rgb
    scale = max_long_edge / float(long_edge)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)


def _fast_gui_mask_repair(
    image: np.ndarray,
    mask: np.ndarray,
    method: str,
    *,
    repair_expand_radius: int = 0,
    feather_radius: int = 0,
    color_match_strength: float = 0.0,
    grain_strength: float = 0.25,
    frequency_scope_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = datetime.now()
    original = np.asarray(image)
    if original.ndim != 3 or original.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported target image shape for fast GUI repair: {original.shape}")

    core_mask_bool = np.asarray(mask) > 0
    if core_mask_bool.ndim == 3:
        core_mask_bool = np.max(core_mask_bool[:, :, :3], axis=2)
    if core_mask_bool.shape != original.shape[:2]:
        raise ValueError(
            "image and mask dimensions differ for fast GUI repair: "
            f"image={original.shape[1]}x{original.shape[0]}, "
            f"mask={core_mask_bool.shape[1]}x{core_mask_bool.shape[0]}"
        )

    repair_mask_bool = dilate_mask(core_mask_bool, int(repair_expand_radius)) if repair_expand_radius > 0 else core_mask_bool
    ys, xs = np.nonzero(repair_mask_bool)
    if ys.size == 0:
        return original.copy(), _fast_gui_metrics(
            original,
            original,
            core_mask_bool,
            repair_mask_bool,
            0.0,
            method,
            0,
            repair_expand_radius,
            feather_radius,
            color_match_strength,
            grain_strength,
            final_alpha=np.zeros(core_mask_bool.shape, dtype=np.float32),
        )

    image_float = as_float32(original)
    current_rgb = image_float[:, :, :3].copy()
    context_samples = _fast_context_samples(current_rgb, repair_mask_bool, ys, xs)
    if context_samples.size == 0:
        return original.copy(), _fast_gui_metrics(
            original,
            original,
            core_mask_bool,
            repair_mask_bool,
            0.0,
            method,
            0,
            repair_expand_radius,
            feather_radius,
            color_match_strength,
            grain_strength,
            final_alpha=np.zeros(core_mask_bool.shape, dtype=np.float32),
        )

    current_rgb[ys, xs] = np.median(context_samples, axis=0).astype(np.float32)
    iterations = int(np.clip(np.ceil(np.sqrt(float(ys.size))) / 8.0, 8, 40))
    height, width = repair_mask_bool.shape

    for _ in range(iterations):
        sums = np.zeros((ys.size, 3), dtype=np.float32)
        counts = np.zeros((ys.size, 1), dtype=np.float32)
        _accumulate_fast_neighbor(current_rgb, sums, counts, ys, xs, ys, xs - 1, xs > 0)
        _accumulate_fast_neighbor(current_rgb, sums, counts, ys, xs, ys, xs + 1, xs + 1 < width)
        _accumulate_fast_neighbor(current_rgb, sums, counts, ys, xs, ys - 1, xs, ys > 0)
        _accumulate_fast_neighbor(current_rgb, sums, counts, ys, xs, ys + 1, xs, ys + 1 < height)
        current_rgb[ys, xs] = sums / np.maximum(counts, 1.0)

    if method == "kl":
        current_rgb[ys, xs] = _fast_kl_adjustment(current_rgb[ys, xs], context_samples)
    if color_match_strength > 0.0:
        current_rgb[ys, xs] = _fast_match_pixels_to_context(current_rgb[ys, xs], context_samples, color_match_strength)
    if grain_strength > 0.0:
        current_rgb = _fast_reinject_grain(image_float[:, :, :3], current_rgb, repair_mask_bool, grain_strength)

    final_alpha = _fast_gui_blend_alpha(core_mask_bool, repair_mask_bool, int(feather_radius))
    output_float = image_float.copy()
    output_float[:, :, :3] = (
        image_float[:, :, :3] * (1.0 - final_alpha[:, :, None])
        + current_rgb * final_alpha[:, :, None]
    )
    if output_float.shape[2] == 4:
        output_float[:, :, 3] = image_float[:, :, 3]
    repaired = restore_dtype(output_float, original.dtype)
    outside = final_alpha <= 0.0
    repaired[outside, :] = original[outside, :]

    precise_frequency_metrics: dict[str, Any] | None = None
    override_pixel_count = 0
    if method == "defect_aware" and frequency_scope_mask is not None:
        scope_bool = normalize_frequency_scope_mask(frequency_scope_mask, core_mask_bool.shape)
        precise_cfg = RepairConfig(
            method="defect_aware",
            mask_channel="grayscale",
            threshold=0.5,
            dilate_radius=int(repair_expand_radius),
            feather_radius=int(feather_radius),
            strength=1.0,
            padding=16,
            max_component_area=GUI_MAX_COMPONENT_AREA,
            grain_reinject_strength=float(grain_strength),
            color_match_strength=float(color_match_strength),
            frequency_guided_enabled=True,
        )
        selection = prepare_frequency_selection(core_mask_bool, scope_bool, precise_cfg)
        if np.any(selection.selected_core_mask):
            precise_mask = selection.selected_core_mask.astype(np.uint8) * 255
            precise = repair_image(original, precise_mask, precise_cfg, frequency_scope_mask=scope_bool)
            precise_frequency_metrics = {
                key: value for key, value in precise.metrics.items() if key.startswith("frequency_")
            }
            override = (precise.blend_alpha > 0.0) if precise.blend_alpha is not None else np.zeros(core_mask_bool.shape, dtype=bool)
            if np.any(override):
                override_pixel_count = int(np.count_nonzero(override))
                repaired[override] = precise.repaired_image[override]
                final_alpha = np.where(override, precise.blend_alpha, final_alpha).astype(np.float32)
    elapsed_ms = (datetime.now() - started).total_seconds() * 1000.0
    metrics = _fast_gui_metrics(
        original,
        repaired,
        core_mask_bool,
        repair_mask_bool,
        elapsed_ms,
        method,
        iterations,
        repair_expand_radius,
        feather_radius,
        color_match_strength,
        grain_strength,
        final_alpha=final_alpha,
    )
    if method == "defect_aware" and frequency_scope_mask is not None:
        scope_bool = normalize_frequency_scope_mask(frequency_scope_mask, core_mask_bool.shape)
        precise_cfg = RepairConfig(method="defect_aware", frequency_guided_enabled=True)
        selection = prepare_frequency_selection(core_mask_bool, scope_bool, precise_cfg)
        if precise_frequency_metrics is not None:
            metrics.update(precise_frequency_metrics)
        metrics.update({
            "frequency_guided_enabled": True,
            "frequency_scope_mask_pixel_count": int(np.count_nonzero(scope_bool)),
            "frequency_selected_region_count": int(selection.selected_region_count),
            "frequency_selected_component_count": int(selection.selected_component_count),
            "frequency_selected_core_pixel_count": int(selection.selected_core_pixel_count),
            "frequency_fast_mode_override_count": 1 if override_pixel_count > 0 else 0,
            "frequency_fast_mode_selected_pixel_count": int(np.count_nonzero(selection.selected_core_mask)),
            "frequency_fast_mode_override_pixel_count": int(override_pixel_count),
        })
    return repaired, metrics

def _accumulate_fast_neighbor(
    current_rgb: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    neighbor_y: np.ndarray,
    neighbor_x: np.ndarray,
    valid: np.ndarray,
) -> None:
    if not np.any(valid):
        return
    sums[valid] += current_rgb[neighbor_y[valid], neighbor_x[valid]]
    counts[valid] += 1.0


def _fast_context_samples(current_rgb: np.ndarray, mask_bool: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    samples: list[np.ndarray] = []
    height, width = mask_bool.shape
    for neighbor_y, neighbor_x, valid in (
        (ys, xs - 1, xs > 0),
        (ys, xs + 1, xs + 1 < width),
        (ys - 1, xs, ys > 0),
        (ys + 1, xs, ys + 1 < height),
    ):
        valid_known = valid.copy()
        valid_indices = np.flatnonzero(valid_known)
        if valid_indices.size == 0:
            continue
        known = ~mask_bool[neighbor_y[valid_indices], neighbor_x[valid_indices]]
        if np.any(known):
            selected = valid_indices[known]
            samples.append(current_rgb[neighbor_y[selected], neighbor_x[selected]])
    if not samples:
        return np.empty((0, 3), dtype=np.float32)
    context = np.concatenate(samples, axis=0).astype(np.float32, copy=False)
    if context.shape[0] > _FAST_CONTEXT_SAMPLE_LIMIT:
        indices = np.linspace(0, context.shape[0] - 1, _FAST_CONTEXT_SAMPLE_LIMIT).astype(np.int64)
        context = context[indices]
    return context


def _fast_kl_adjustment(linear_pixels: np.ndarray, context_samples: np.ndarray) -> np.ndarray:
    if linear_pixels.size == 0 or context_samples.size == 0:
        return linear_pixels

    context_bins = _fast_color_bins(context_samples)
    sums = np.zeros((512, 3), dtype=np.float64)
    counts = np.bincount(context_bins, minlength=512).astype(np.int64)
    np.add.at(sums, context_bins, context_samples.astype(np.float64))
    nonempty = counts > 0
    representatives = np.zeros((512, 3), dtype=np.float32)
    representatives[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(np.float32)

    output = linear_pixels.copy()
    pixel_bins = _fast_color_bins(linear_pixels)
    has_direct = nonempty[pixel_bins]
    output[has_direct] = representatives[pixel_bins[has_direct]]

    missing_indices = np.flatnonzero(~has_direct)
    available = np.flatnonzero(nonempty)
    if missing_indices.size > 0 and available.size > 0:
        available_colors = representatives[available]
        for start in range(0, missing_indices.size, 10000):
            chunk = missing_indices[start : start + 10000]
            diffs = linear_pixels[chunk, None, :] - available_colors[None, :, :]
            nearest = np.argmin(np.sum(diffs * diffs, axis=2), axis=1)
            output[chunk] = available_colors[nearest]
    return output



def _fast_match_pixels_to_context(pixels: np.ndarray, context_samples: np.ndarray, strength: float) -> np.ndarray:
    if pixels.size == 0 or context_samples.size == 0 or strength <= 0.0:
        return pixels
    pixel_mean = np.mean(pixels, axis=0)
    context_mean = np.mean(context_samples, axis=0)
    pixel_std = np.std(pixels, axis=0)
    context_std = np.std(context_samples, axis=0)
    scale = np.where(pixel_std > 1.0e-5, context_std / np.maximum(pixel_std, 1.0e-5), 1.0)
    scale = np.clip(scale, 0.25, 4.0)
    matched = (pixels - pixel_mean) * scale + context_mean
    q_low = np.percentile(context_samples, 2.0, axis=0)
    q_high = np.percentile(context_samples, 98.0, axis=0)
    matched = np.clip(matched, q_low - 0.025, q_high + 0.025)
    amount = float(np.clip(strength, 0.0, 1.0))
    return np.clip(pixels * (1.0 - amount) + matched * amount, 0.0, 1.0).astype(np.float32)


def _fast_reinject_grain(original_rgb: np.ndarray, current_rgb: np.ndarray, repair_mask: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0 or not np.any(repair_mask):
        return current_rgb
    context = dilate_mask(repair_mask, 8) & ~repair_mask
    if int(np.count_nonzero(context)) < 16:
        return current_rgb
    residual = original_rgb - _fast_box_blur_rgb(original_rgb, 1)
    samples = residual[context]
    if samples.size == 0:
        return current_rgb
    ys, xs = np.nonzero(repair_mask)
    indices = ((ys.astype(np.int64) * 73856093) ^ (xs.astype(np.int64) * 19349663)) % samples.shape[0]
    output = current_rgb.copy()
    output[ys, xs] = np.clip(output[ys, xs] + samples[indices] * float(strength), 0.0, 1.0)
    return output


def _fast_gui_blend_alpha(core_mask_bool: np.ndarray, repair_mask_bool: np.ndarray, feather_radius: int) -> np.ndarray:
    core = np.asarray(core_mask_bool, dtype=bool)
    repair = np.asarray(repair_mask_bool, dtype=bool)
    alpha = np.zeros(repair.shape, dtype=np.float32)
    if not np.any(repair):
        return alpha
    shell = repair & ~core
    if np.any(shell):
        if feather_radius > 0:
            blurred = _fast_box_blur_scalar(core.astype(np.float32), max(1, int(feather_radius)))
            shell_alpha = np.clip(blurred, 0.15, 0.65)
            alpha[shell] = shell_alpha[shell]
        else:
            alpha[shell] = 0.35
    alpha[core] = 1.0
    return alpha


def _fast_box_blur_scalar(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.astype(np.float32, copy=False)
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
    result = np.zeros_like(image, dtype=np.float32)
    count = 0
    size = radius * 2 + 1
    for dy in range(size):
        for dx in range(size):
            result += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
            count += 1
    return result / float(count)


def _fast_box_blur_rgb(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.astype(np.float32, copy=False)
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    result = np.zeros_like(image, dtype=np.float32)
    count = 0
    size = radius * 2 + 1
    for dy in range(size):
        for dx in range(size):
            result += padded[dy : dy + image.shape[0], dx : dx + image.shape[1], :]
            count += 1
    return result / float(count)


def _fast_color_bins(pixels: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(pixels, dtype=np.float32), 0.0, 1.0)
    bins = np.minimum((values * 8.0).astype(np.int16), 7)
    return (bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]).astype(np.int16)


def _fast_gui_metrics(
    original: np.ndarray,
    repaired: np.ndarray,
    core_mask_bool: np.ndarray,
    repair_mask_bool: np.ndarray,
    processing_time_ms: float,
    method: str,
    iterations: int,
    repair_expand_radius: int,
    feather_radius: int,
    color_match_strength: float,
    grain_strength: float,
    final_alpha: np.ndarray | None = None,
) -> dict[str, Any]:
    original_float = as_float32(original)
    repaired_float = as_float32(repaired)
    diff = np.max(np.abs(repaired_float[:, :, :3] - original_float[:, :, :3]), axis=2)
    outside = ~repair_mask_bool
    shell = repair_mask_bool & ~core_mask_bool
    alpha = np.zeros(repair_mask_bool.shape, dtype=np.float32) if final_alpha is None else np.asarray(final_alpha, dtype=np.float32)
    core_alpha = alpha[core_mask_bool]
    shell_alpha = alpha[shell]
    metrics = {
        "method": method,
        "gui_fast_mode": True,
        "linear_iterations": int(iterations),
        "mask_expand_radius": int(repair_expand_radius),
        "feather_radius": int(feather_radius),
        "color_match_strength": float(color_match_strength),
        "grain_reinject_strength": float(grain_strength),
        "mask_pixel_count": int(np.count_nonzero(core_mask_bool)),
        "repair_mask_pixel_count": int(np.count_nonzero(repair_mask_bool)),
        "changed_pixel_count_core": int(np.count_nonzero((diff > 0.0) & core_mask_bool)),
        "changed_pixel_count_shell": int(np.count_nonzero((diff > 0.0) & shell)),
        "max_abs_diff_outside_mask": float(np.max(diff[outside])) if np.any(outside) else 0.0,
        "mean_abs_diff_core": float(np.mean(diff[core_mask_bool])) if np.any(core_mask_bool) else 0.0,
        "mean_abs_diff_shell": float(np.mean(diff[shell])) if np.any(shell) else 0.0,
        "gui_fast_core_alpha_min": float(np.min(core_alpha)) if core_alpha.size else 0.0,
        "gui_fast_core_alpha_mean": float(np.mean(core_alpha)) if core_alpha.size else 0.0,
        "gui_fast_core_alpha_max": float(np.max(core_alpha)) if core_alpha.size else 0.0,
        "gui_fast_core_alpha_below_full_count": int(np.count_nonzero(core_alpha < 0.999999)),
        "gui_fast_shell_alpha_mean": float(np.mean(shell_alpha)) if shell_alpha.size else 0.0,
        "processing_time_ms": float(processing_time_ms),
    }
    if method == "defect_aware":
        metrics.update(
            {
                "defect_aware": True,
                "defect_aware_version": 1,
                "defect_aware_fallback_method": "linear",
                "defect_strategy_counts": {"gui_fast_linear_fallback": 1 if np.any(repair_mask_bool) else 0},
                "gui_fast_fallback_method": "linear",
                "defect_core_alpha_min": metrics["gui_fast_core_alpha_min"],
                "defect_core_alpha_mean": metrics["gui_fast_core_alpha_mean"],
                "defect_core_alpha_max": metrics["gui_fast_core_alpha_max"],
                "defect_core_alpha_below_full_count": metrics["gui_fast_core_alpha_below_full_count"],
            }
        )
    return metrics

def _build_manifest(
    *,
    target_file: Path,
    inspection_file: Path,
    method: str,
    white_manifest: dict[str, Any],
    repair_metrics: dict[str, Any],
    repaired_path: Path,
    mask_path: Path,
    repair_mask_path: Path | None = None,
    repair_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "実行日": datetime.now().isoformat(timespec="seconds"),
        "version": f"KLComplementary {GUI_VERSION} / dust-mask-repair {_package_version()}",
        "対象写真": str(target_file),
        "マスク作成用写真": str(inspection_file),
        "補正方法": method,
        "正答率": None,
        "補正後画像": str(repaired_path),
        "生成マスク": str(mask_path),
        "補正対象マスク": str(repair_mask_path) if repair_mask_path is not None else None,
        "補正設定": _json_safe(repair_options or {}),
        "mask_pixels": int(white_manifest.get("final_mask_pixels", 0)),
        "white_dust": _json_safe(white_manifest),
        "repair_metrics": _json_safe(repair_metrics),
    }


def _write_processing_status(path: Path, **values: Any) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **{key: _json_safe(value) for key, value in values.items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _package_version() -> str:
    try:
        return version("dust-mask-repair")
    except PackageNotFoundError:
        return "local"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_gui_error(exc: Exception, output_dir: Path) -> str:
    log_path = _write_gui_error_log(output_dir, exc)
    message = str(exc)
    if log_path is not None:
        message = f"{message}\n\n詳細ログ: {log_path}"
    return message


def _write_gui_error_log(output_dir: Path, exc: Exception) -> Path | None:
    try:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "last_error.txt"
        _write_error_log(log_path, exc)
        return log_path
    except Exception:
        return None


def _write_error_log(path: Path, exc: Exception) -> None:
    path.write_text("".join(traceback.format_exception(exc)), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
