from typing import Dict, List

import cv2
import numpy as np
import streamlit as st
from utils import pad_image_to_square

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
    sample: QuerySample,
    gallery_dataset: GalleryDataset,
    matching_type: RetrievalResultsType,
):
    query_image = sample.load_image()
    query_image = pad_image_to_square(query_image, SIZE - BORDER_SIZE)
    query_image = cv2.copyMakeBorder(
        query_image, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, cv2.BORDER_CONSTANT, value=BLACK_COLOR
    )
    images = []
    infos = []
    images.append(query_image)
    infos.append({"Label": str(sample.label), "Category": sample.category})
    top_k_images_ids = (
        sample.top_k_images_ids
        if matching_type == RetrievalResultsType.before_stir
        else sample.postprocessed_top_k_images_ids
    )
    top_k_scores = (
        sample.top_k_scores if matching_type == RetrievalResultsType.before_stir else sample.postprocessed_top_k_scores
    )
    for image_id, score in zip(top_k_images_ids, top_k_scores):
        gallery_sample = gallery_dataset.find_by_id(image_id)
        image = gallery_sample.load_image()
        if gallery_sample.label == sample.label:
            image = pad_image_to_square(image, SIZE - BORDER_SIZE)
            image = cv2.copyMakeBorder(
                image, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, cv2.BORDER_CONSTANT, value=GREEN_COLOR
            )
        else:
            image = pad_image_to_square(image, SIZE)
        images.append(image)
        infos.append(
            {
                "Label": gallery_sample.label,
                "Category": gallery_sample.category,
                "Distance": score,
            }
        )
    viewer = RetrievalResultsViewer(top_k, more_info_flag)
    viewer.show(images, infos)
