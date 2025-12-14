import os
import random
import shutil
import tempfile
import unittest

import cv2
import numpy as np

import utils.preprocessing as preprocessing
import utils.run_engine as engine_stage2
import utils.run_engine_stage1 as engine_stage1

Stage2Dataset = engine_stage2.DATASET
Stage1Dataset = engine_stage1.DATASET


def _make_toy_sample(root_dir: str, name: str = "sample", size: int = 96):
    os.makedirs(root_dir, exist_ok=True)
    image_path = os.path.join(root_dir, f"{name}.png")
    mask_path = os.path.join(root_dir, f"{name}_mask.png")

    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    rgb = np.stack([
        grid_x,
        grid_y,
        1.0 - grid_x
    ], axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 3, 255, -1)
    cv2.imwrite(mask_path, mask)

    return image_path, mask_path


class DualInputPreprocessingTest(unittest.TestCase):
    def test_triplet_tensor_shape_and_range(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, size=(128, 80, 3), dtype=np.uint8)

        tensor = preprocessing.build_triplet_tensor(image)

        self.assertEqual(tensor.shape, (3, 128, 80))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all((tensor >= 0.0) & (tensor <= 1.0)))

    def test_degradation_changes_tensor(self):
        base = np.full((3, 64, 64), 0.75, dtype=np.float32)
        random.seed(0)
        degraded = preprocessing.degrade_triplet(base, shadow_prob=1.0, occlusion_prob=1.0, deformation_prob=1.0)

        self.assertFalse(np.allclose(base, degraded))
        self.assertTrue(np.all((degraded >= 0.0) & (degraded <= 1.0)))


class DualInputDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="condseg_test_")
        cls.image_path, cls.mask_path = _make_toy_sample(cls.tmp_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_stage2_dataset_emits_degraded_pair(self):
        original_degrade = engine_stage2.degrade_triplet
        try:
            engine_stage2.degrade_triplet = lambda tensor: tensor * 0.5
            dataset = Stage2Dataset(
                [self.image_path],
                [self.mask_path],
                size=(64, 64),
                transform=None,
                dual_input=True
            )

            (primary, degraded), (mask, background) = dataset[0]
        finally:
            engine_stage2.degrade_triplet = original_degrade

        self.assertEqual(primary.shape, (3, 64, 64))
        self.assertEqual(degraded.shape, (3, 64, 64))
        self.assertEqual(mask.shape, (1, 64, 64))
        self.assertEqual(background.shape, (1, 64, 64))
        self.assertFalse(np.allclose(primary, degraded))
        self.assertTrue(np.all((primary >= 0.0) & (primary <= 1.0)))
        self.assertTrue(np.all((degraded >= 0.0) & (degraded <= 1.0)))
        self.assertTrue(np.all((mask >= 0.0) & (mask <= 1.0)))
        self.assertTrue(np.all((background >= 0.0) & (background <= 1.0)))

    def test_stage1_dataset_can_disable_dual_input(self):
        dataset = Stage1Dataset(
            [self.image_path],
            [self.mask_path],
            size=(64, 64),
            transform=None,
            dual_input=False
        )

        (primary, degraded), _ = dataset[0]
        self.assertTrue(np.allclose(primary, degraded))


if __name__ == "__main__":
    unittest.main()
