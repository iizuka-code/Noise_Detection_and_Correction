import json

import numpy as np

from dust_mask_repair import RepairConfig, repair_image
from dust_mask_repair.defects import DEFECT_CLASSIFIER_VERSION, classify_defects


def _constant_image(shape=(40, 48), value=(90, 115, 140), channels=3):
    image = np.zeros((*shape, channels), dtype=np.uint8)
    image[:, :, :3] = np.array(value, dtype=np.uint8)
    if channels == 4:
        image[:, :, 3] = 180
    return image


def _striped_image(shape=(64, 72)):
    height, width = shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    stripe = ((np.arange(width) // 3) % 2).astype(np.uint8)
    image[:, :, 0] = np.where(stripe[None, :] == 0, 50, 210)
    image[:, :, 1] = np.where(stripe[None, :] == 0, 80, 180)
    image[:, :, 2] = np.where(stripe[None, :] == 0, 120, 90)
    return image


def _features_for(image, mask):
    return classify_defects(image, mask, repair_mask=mask)


def test_one_pixel_defect_is_tiny_local():
    image = _constant_image()
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[18, 20] = True

    features = _features_for(image, mask)

    assert len(features) == 1
    assert features[0].recommended_strategy == "tiny_local"
    assert features[0].area == 1


def test_small_smooth_defect_is_small_local():
    image = _constant_image()
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[16:20, 18:22] = True

    features = _features_for(image, mask)

    assert len(features) == 1
    assert features[0].recommended_strategy == "small_local"


def test_thin_line_defect_is_directional():
    image = _constant_image(shape=(36, 52))
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[17, 10:36] = True

    features = _features_for(image, mask)

    assert len(features) == 1
    assert features[0].recommended_strategy == "directional"
    assert features[0].elongation >= 4.0


def test_textured_region_recommends_patch():
    image = _striped_image()
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[24:34, 30:40] = True

    features = _features_for(image, mask)

    assert len(features) == 1
    assert features[0].recommended_strategy == "patch"
    assert features[0].texture_score > 0.0


def test_border_context_shortage_uses_safe_strategy():
    image = _constant_image(shape=(28, 28))
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[:12, :12] = True

    features = _features_for(image, mask)

    assert len(features) == 1
    assert features[0].touches_border is True
    assert features[0].recommended_strategy in {"skip", "fast_inpaint"}


def test_classification_does_not_mutate_image_or_mask_and_ignores_alpha():
    image = _constant_image(channels=4)
    image[:, :, 3] = np.arange(image.shape[1], dtype=np.uint8)[None, :]
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[10:14, 11:15] = True
    image_before = image.copy()
    mask_before = mask.copy()

    features = _features_for(image, mask)

    assert len(features) == 1
    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)
    assert np.array_equal(image[:, :, 3], image_before[:, :, 3])


def test_defect_aware_repair_records_classification_metrics_and_debug_json(tmp_path):
    image = _constant_image(shape=(36, 44))
    damaged = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[12, 12] = 255
    mask[20, 18:34] = 255
    damaged[mask > 0] = [255, 255, 255]

    result = repair_image(
        damaged,
        mask,
        RepairConfig(
            method="defect_aware",
            mask_channel="grayscale",
            dilate_radius=0,
            feather_radius=0,
            padding=8,
            debug_dir=tmp_path,
        ),
    )

    assert result.metrics["defect_classification_enabled"] is True
    assert result.metrics["defect_classifier_version"] == DEFECT_CLASSIFIER_VERSION
    assert result.metrics["defect_component_count"] == 2
    assert sum(result.metrics["defect_strategy_counts"].values()) == 2
    assert result.metrics["defect_strategy_counts"]["tiny_local"] == 1
    assert result.metrics["defect_strategy_counts"]["directional"] == 1

    debug_path = tmp_path / "defect_components.json"
    assert debug_path.exists()
    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    assert payload["summary"]["defect_component_count"] == 2
    assert len(payload["components"]) == 2
    assert result.debug_paths["defect_components"] == str(debug_path)
