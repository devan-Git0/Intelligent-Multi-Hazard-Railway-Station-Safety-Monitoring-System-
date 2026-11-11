"""
STEP 2 -- Capture & Preprocess Frame
  2.1 Capture one frame from the CCTV camera
  2.2 Resize frame to model input size
  2.3 Normalize pixel values (0-1 range)
  2.4 Apply ROI mask if needed
  2.5 Pass frame to next step (returned to the caller)

Run this file directly to test preprocessing on a synthetic frame --
no real camera needed to verify the math is correct.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from config import IMG_SIZE


def open_camera(source=0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {source}")
    return cap


def capture_and_preprocess(
    cap: cv2.VideoCapture,
    img_size: int = IMG_SIZE,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (frame_norm, frame_resized) or (None, None) if the camera read
    failed -- the main loop should skip the cycle on None, not crash.

    frame_norm    : float32 RGB array in [0, 1] -- what the models consume
    frame_resized : uint8 BGR array -- for overlays, snapshots, dashboard
    """
    # 2.1 -- capture one frame
    ret, frame = cap.read()
    if not ret or frame is None:
        return None, None

    return preprocess_frame(frame, img_size=img_size, roi_mask=roi_mask)


def preprocess_frame(
    frame: np.ndarray,
    img_size: int = IMG_SIZE,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """The actual preprocessing math, split out from capture_and_preprocess
    so it can be unit-tested on a synthetic frame with no camera at all."""

    # 2.4 -- apply ROI mask if configured. This is a general "valid analysis
    # region" mask (e.g. blacking out a timestamp overlay or an area outside
    # the camera's real coverage) -- NOT the same as a hazard module's own
    # zone polygon, which is applied later, per-module, in hazards.py.
    if roi_mask is not None:
        frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

    # 2.2 -- resize to model input size
    frame_resized = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # 2.3 -- normalize to [0, 1], RGB order (Ultralytics expects RGB)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb.astype(np.float32) / 255.0

    # 2.5 -- pass both versions to the next step
    return frame_norm, frame_resized


if __name__ == "__main__":
    # Synthetic test frame -- 480p, random noise -- stands in for a real
    # camera frame so this file is testable with zero hardware.
    fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    frame_norm, frame_resized = preprocess_frame(fake_frame)
    print("Input shape :", fake_frame.shape)
    print("Resized shape:", frame_resized.shape, frame_resized.dtype)
    print("Normalized shape:", frame_norm.shape, frame_norm.dtype,
          "min/max:", frame_norm.min(), frame_norm.max())

    # Test with an ROI mask covering only the bottom 65% of frame
    mask = np.zeros(fake_frame.shape[:2], dtype=np.uint8)
    mask[int(0.35 * 480):, :] = 255
    frame_norm_roi, frame_resized_roi = preprocess_frame(fake_frame, roi_mask=mask)
    print("\nWith ROI mask applied -- top 35% should be black:")
    print("  top-left pixel (should be 0):", frame_resized_roi[0, 0])
    print("  bottom-left pixel (should vary):", frame_resized_roi[-1, 0])

    assert frame_resized.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert 0.0 <= frame_norm.min() and frame_norm.max() <= 1.0
    print("\nAll preprocessing checks passed.")
