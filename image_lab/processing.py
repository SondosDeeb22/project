"""Transformations + registry.

The UI is driven entirely by TRANSFORMS: formula, variables (sliders) and
whether histograms are shown. To add a technique, write a function and add one
Transform(...) entry. If you already have your own implementations, replace the
bodies of the functions below and keep the signatures (uint8 gray array in,
uint8 gray array out).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import cv2


@dataclass(frozen=True)
class Var:
    key: str
    label: str
    min: float
    max: float
    default: float
    step: float


@dataclass(frozen=True)
class Transform:
    key: str
    name: str
    group: str
    formula: str  # HTML
    vars: tuple
    func: Callable
    shows_histogram: bool = False


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# Grayscale Conversion Implementation
# ------------------------------------------------------------
def convert_to_grayscale(image):
    """Convert RGB image to grayscale using the standard weighted formula."""
    if len(image.shape) == 2:
        # Already grayscale
        return image

    # Split into B, G, R channels (OpenCV uses BGR format)
    B, G, R = cv2.split(image)
    # Apply grayscale formula: Gray = 0.299*R + 0.587*G + 0.114*B
    grayscale = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
    return grayscale


# ------------------------------------------------------------
# Negative Transformation Implementation
# ------------------------------------------------------------
def negative(img, **_):
    """Apply negative transformation: s = (L - 1) - r"""
    L = 256
    negative = (L - 1) - img
    return negative.astype(np.uint8)


# ------------------------------------------------------------
# Log Transformation Implementation
# ------------------------------------------------------------
def log_transform(img, c=46.0, **_):
    """Apply log transformation: s = c * log(1 + r)"""
    if c is None:
        # Default normalization: c = 255 / ln(1 + 255)
        c = 255 / np.log(1 + 255)
    
    # Calculate log transformation directly
    transformed = c * np.log(1 + img.astype(np.float64))
    # Round and constrain to 0-255
    transformed = np.round(transformed).astype(np.uint8)
    transformed = np.clip(transformed, 0, 255)
    return transformed


# ------------------------------------------------------------
# Power-Law (Gamma) Transformation Implementation
# ------------------------------------------------------------
def power_law(img, c=1.0, gamma=0.5, **_):
    """Apply power-law (gamma) transformation: s = c * r^gamma"""
    if c is None:
        # Default normalization: c = 255 / (255^gamma)
        c = 255 / (255 ** gamma)
    
    # Calculate power-law transformation directly
    transformed = c * (img.astype(np.float64) ** gamma)
    # Round and constrain to 0-255
    transformed = np.round(transformed).astype(np.uint8)
    transformed = np.clip(transformed, 0, 255)
    return transformed


# ------------------------------------------------------------
# Linear Contrast Stretching Implementation
# ------------------------------------------------------------
def linear_stretch(img, **_):
    """Apply linear contrast stretching: s = ((r - r_min) / (r_max - r_min)) * 255"""
    r_min = np.min(img)
    r_max = np.max(img)
    
    # Handle special case where all pixels have the same intensity
    if r_min == r_max:
        # Return the image as-is or a constant value
        return img.copy()
    
    # Apply linear contrast stretching
    stretched = ((img.astype(np.float64) - r_min) / (r_max - r_min)) * 255
    # Round and constrain to 0-255
    stretched = np.round(stretched).astype(np.uint8)
    stretched = np.clip(stretched, 0, 255)
    return stretched


# ------------------------------------------------------------
# Piecewise Linear Contrast Stretching Implementation
# ------------------------------------------------------------
def piecewise_stretch(img, r1=70, s1=20, r2=180, s2=235, **_):
    """Apply piecewise linear contrast stretching with control points (r1, s1) and (r2, s2)"""
    grayscale_float = img.astype(np.float64)
    output = np.zeros_like(grayscale_float)
    
    # Ensure valid control points
    r1 = max(1, min(r1, 254))
    r2 = max(r1 + 1, min(r2, 255))
    s1 = max(0, min(s1, 255))
    s2 = max(0, min(s2, 255))
    
    # Region 1: 0 <= r <= r1
    mask1 = grayscale_float <= r1
    if r1 > 0:
        output[mask1] = (s1 / r1) * grayscale_float[mask1]
    else:
        output[mask1] = 0
    
    # Region 2: r1 < r <= r2
    mask2 = (grayscale_float > r1) & (grayscale_float <= r2)
    if r2 > r1:
        output[mask2] = s1 + ((s2 - s1) / (r2 - r1)) * (grayscale_float[mask2] - r1)
    else:
        output[mask2] = s1
    
    # Region 3: r2 < r <= 255
    mask3 = grayscale_float > r2
    if r2 < 255:
        output[mask3] = s2 + ((255 - s2) / (255 - r2)) * (grayscale_float[mask3] - r2)
    else:
        output[mask3] = s2
    
    # Handle any NaN or inf values
    output = np.nan_to_num(output, nan=0.0, posinf=255.0, neginf=0.0)
    
    # Round and constrain to 0-255
    output = np.round(output).astype(np.uint8)
    output = np.clip(output, 0, 255)
    return output


# ------------------------------------------------------------
# Histogram Equalization Implementation
# ------------------------------------------------------------
def hist_equalize(img, **_):
    """Apply histogram equalization using CDF: s_k = round(255 * CDF(k))"""
    L = 256  # Number of intensity levels for 8-bit images
    
    # Calculate histogram
    histogram, _ = np.histogram(img.flatten(), bins=L, range=[0, L])
    
    # Calculate probability of each intensity
    total_pixels = img.size
    probability = histogram / total_pixels
    
    # Calculate cumulative distribution function (CDF)
    cdf = np.cumsum(probability)
    
    # Generate intensity mapping
    mapping = np.round((L - 1) * cdf).astype(np.uint8)
    
    # Apply mapping to original image
    equalized = mapping[img]
    
    return equalized

## ================================================================================================
## ================================================================================================

TRANSFORMS = [
    Transform("negative", "Image negative", "Intensity", "s = (L − 1) − r", (), negative),
    Transform("log", "Log transformation", "Intensity", "s = c · log(1 + r)",
              (Var("c", "c", 1.0, 100.0, 46.0, 0.5),), log_transform),
    Transform("gamma", "Power-law (gamma)", "Intensity", "s = c · r<sup>γ</sup>",
              (Var("c", "c", 0.1, 2.0, 1.0, 0.05), Var("gamma", "γ (gamma)", 0.1, 5.0, 0.5, 0.05)),
              power_law),
    Transform("linear", "Linear contrast stretching", "Contrast",
              "s = (r − r<sub>min</sub>) / (r<sub>max</sub> − r<sub>min</sub>) · (L − 1)", (), linear_stretch),
    Transform("piecewise", "Piecewise linear stretching", "Contrast",
              "s = T(r) through (r<sub>1</sub>, s<sub>1</sub>) and (r<sub>2</sub>, s<sub>2</sub>)",
              (Var("r1", "r1", 1, 253, 70, 1), Var("s1", "s1", 0, 255, 20, 1),
               Var("r2", "r2", 2, 254, 180, 1), Var("s2", "s2", 0, 255, 235, 1)),
              piecewise_stretch),
    Transform("equalize", "Histogram equalization", "Contrast", "s = (L − 1) · CDF(r)", (),
              hist_equalize, shows_histogram=True),
]
BY_KEY = {t.key: t for t in TRANSFORMS}


def get(key: str) -> Transform:
    return BY_KEY[key]
