import number
import random
import warnings 
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \ 
    CenterCrop, ColorJitter, Grayscale

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .utils import to_2tuple

@dataclass
class PreprocessCfg:
    size: Union[int, Tuple[int, int]] = 224
    mode: str = 'RGB'
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN
    std: Tuple[float, ...] = OPENAI_DATASET_STD
    interpolation: str = 'bicubic'
    resize_mode: str = 'shortest'
    fill_color: int = 0

    def __post_init__(self):
        assert self.mode in ('RGB',)
    
    @property
    def num_channels(self):
        return 3
    
    @property
    def input_size(self):
        return (self.num_channels,) + to_2tuple(self.size)

_PREPROCESS_KEYS = set(asdict(PreprocessCfg()).keys())

def merge_preprocess_dict(
        base: Union[PreprocessCfg, Dict],
        overlay: Dict,
):
    if isinstance(base, PreprocessCfg):
        base_clean = asdict(base)
    
    else:
        base_clean = {k: v for k, v in base.items() if k in _PREPROCESS_KEYS}

    if overlay: 
        overlay_clean = {k: v for k, v in overlay.items() if k in _PREPROCESS_KEYS and v is not None}
        base_clean.update(overlay_clean)
    
    return base_clean

def merge_preprocess_kwargs(base: PreprocessCfg, **kwargs):
    return merge_preprocess_dict(base, kwargs)

@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float], Tuple[float, float, float, float]]] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False

    color_jitter_prob: float = None
    gray_scale_prob: float = None

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)
    
    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]
    
    if len(size) != 2:
        raise ValueError(error_msg)
    
    return size

