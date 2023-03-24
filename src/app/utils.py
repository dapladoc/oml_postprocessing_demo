import asyncio
from typing import List

import cv2
import numpy as np

from src.app.data import Sample
from src.const import WHITE_COLOR


def pad_image_to_square(image, size, border_size=0, color=WHITE_COLOR):
    old_size = image.shape[:2]  # old_size is in (height, width) format

    ratio = float(size - border_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image


async def load_one_image_for_sample(sample: Sample) -> np.ndarray:
    return sample.load_image()


async def _load_images_for_samples(samples):
    return await asyncio.gather(*[load_one_image_for_sample(sample) for sample in samples])


def load_images_for_samples(samples: List[Sample]) -> List[np.ndarray]:
    return list(asyncio.run(_load_images_for_samples(samples)))
