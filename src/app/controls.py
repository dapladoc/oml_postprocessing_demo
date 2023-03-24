from typing import Dict, List

import cv2
import numpy as np
import streamlit as st
from utils import load_images_for_samples, pad_image_to_square

from data import GalleryDataset, QuerySample
from src.const import BLACK_COLOR, BORDER_SIZE, GREEN_COLOR, SIZE, RetrievalResultsType


class RetrievalResultsViewer:
    def __init__(self, n: int, more_info_flag: bool):
        self.more_info_flag = more_info_flag
        self.cols = st.columns(n + 1)  # 1 for query

    def show(self, images: List[np.ndarray], infos: List[Dict[str, str]]):
        for col, image, info in zip(self.cols, images, infos):
            show_image_card(col, image, info, self.more_info_flag)


def show_image_card(st_, image, info, show_info_flag: bool) -> None:
    st_.image(image)
    if show_info_flag:
        for k, v in info.items():
            st_.markdown(f"**{k}**: {v}")


def show_retrieval_results(
    top_k: int,
    more_info_flag: bool,
    query_sample: QuerySample,
    gallery_dataset: GalleryDataset,
    matching_type: RetrievalResultsType,
):

    top_k_images_ids = (
        query_sample.top_k_images_ids
        if matching_type == RetrievalResultsType.before_stir
        else query_sample.postprocessed_top_k_images_ids
    )
    top_k_scores = (
        query_sample.top_k_scores
        if matching_type == RetrievalResultsType.before_stir
        else query_sample.postprocessed_top_k_scores
    )
    gallery_samples = [gallery_dataset.find_sample_by_id(sample_id) for sample_id in top_k_images_ids]
    images = load_images_for_samples([query_sample, *gallery_samples])
    query_image = images[0]
    query_image = pad_image_to_square(query_image, SIZE - BORDER_SIZE)
    query_image = cv2.copyMakeBorder(
        query_image, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, cv2.BORDER_CONSTANT, value=BLACK_COLOR
    )
    images[0] = query_image
    infos = [{"Label": str(query_sample.label), "Category": query_sample.category}]
    for i, (gallery_sample, image, score) in enumerate(zip(gallery_samples, images[1:], top_k_scores)):
        if gallery_sample.label == query_sample.label:
            image = pad_image_to_square(image, SIZE - BORDER_SIZE)
            image = cv2.copyMakeBorder(
                image, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, cv2.BORDER_CONSTANT, value=GREEN_COLOR
            )
        else:
            image = pad_image_to_square(image, SIZE)
        images[i + 1] = image
        infos.append(
            {
                "Label": gallery_sample.label,
                "Category": gallery_sample.category,
                "Distance": score,
            }
        )
    viewer = RetrievalResultsViewer(top_k, more_info_flag)
    viewer.show(images, infos)
