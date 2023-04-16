# flake8: noqa
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from collections import defaultdict
from typing import Dict, Tuple, Union

import streamlit as st
from controls import show_retrieval_results

from data import QueryDataset, load_gallery_dataset, load_query_dataset
from src.const import (
    DATASETS,
    IMPROVED_SUFFIX,
    MAIN_PAGE_ABSTRACT,
    MAIN_PAGE_IMAGE_PATH,
    MAIN_PAGE_TITLE,
    METRICS_TO_EXCLUDE_FROM_VIEWER,
    TOP_K,
    ImprovementFlags,
    RetrievalResultsType,
)

st.set_page_config(layout="wide", page_title="similarity-api")


def main():
    datasets = download_datasets(DATASETS)

    st.title(MAIN_PAGE_TITLE)
    abstract_text_column, abstract_image_column = st.columns([1, 1.5])
    abstract_text_column.markdown(MAIN_PAGE_ABSTRACT)

    abstract_image_column.image(cv2.imread(MAIN_PAGE_IMAGE_PATH)[..., ::-1])

    st.sidebar.subheader("Dataset")
    dataset_name = st.sidebar.selectbox("Dataset", datasets, label_visibility="collapsed")
    query_dataset = load_query_dataset(datasets[dataset_name]["query"], datasets[dataset_name]["zip"])
    gallery_dataset = load_gallery_dataset(datasets[dataset_name]["gallery"], datasets[dataset_name]["zip"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter by")
    filter_options = get_filter_options(query_dataset)
    filter_by = st.sidebar.selectbox(
        "Filter by",
        options=filter_options,
        label_visibility="collapsed",
    )
    filter_by = filter_options[filter_by]

    improvement_flag, query_dataset = filter_query_dataset_by_improvement_flag(filter_by, query_dataset)

    if len(query_dataset) == 0:
        st.markdown("There is no query to fulfill the filter requirements.")
        return
    st.sidebar.markdown("---")
    st.sidebar.subheader("Category")
    category_name = st.sidebar.selectbox("Category", query_dataset.categories, label_visibility="collapsed")
    category_dataset = query_dataset.filter("category", category_name)

    st.sidebar.markdown("---")
    more_info_flag = st.sidebar.checkbox(label="Show more info")
    set_session_state(dataset_name, category_name, filter_by, improvement_flag)

    sample = category_dataset[st.session_state.query_controller_position]

    st.title("Retrieval results")
    top_k = min(query_dataset.max_top_k, TOP_K)
    st.subheader("Baseline model")
    show_retrieval_results(
        top_k,
        more_info_flag,
        sample,
        gallery_dataset,
        matching_type=RetrievalResultsType.before_stir,
    )
    st.subheader("Baseline model + STIR postprocessing")
    show_retrieval_results(
        top_k,
        more_info_flag,
        sample,
        gallery_dataset,
        matching_type=RetrievalResultsType.after_stir,
    )
    prev, random, next_ = st.columns(9, gap="small")[3:6]
    prev.button("Previous query", on_click=_add_to_viewer_position, args=(-1,))
    random.button("Random query", on_click=_add_to_viewer_position, args=(np.random.randint(0, 1e10),))
    next_.button("Next query", on_click=_add_to_viewer_position, args=(1,))


def filter_query_dataset_by_improvement_flag(filter_by: str, query_dataset: QueryDataset) -> Tuple[str, QueryDataset]:
    improvement_flag = ""
    if filter_by:
        improved_query_dataset = query_dataset.filter(filter_by, 1)
        worsened_query_dataset = query_dataset.filter(filter_by, -1)
        improvement_flag_options = [
            f"{ImprovementFlags.improvements.value} ({len(improved_query_dataset)})",
            f"{ImprovementFlags.worsenings.value} ({len(worsened_query_dataset)})",
        ]
        improvement_flag = st.sidebar.radio(
            "Filter type",
            options=improvement_flag_options,
            disabled=not bool(filter_by),
        )
        if improvement_flag == improvement_flag_options[0]:
            query_dataset = improved_query_dataset
        else:
            query_dataset = worsened_query_dataset
    return improvement_flag, query_dataset


def _add_to_viewer_position(v: int):
    st.session_state.query_controller_position += v


@st.cache_resource(show_spinner=True)
def download_datasets(datasets: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
    import gdown

    output: Dict[str, Dict[str, str]] = defaultdict(dict)
    for dataset_name, dataset_info in datasets.items():
        for data_name, gdrive_id in dataset_info["gdrive_ids"].items():
            local_path = dataset_info["local_paths"][data_name]
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            if not Path(local_path).exists():
                gdown.download(id=gdrive_id, output=local_path, quiet=False)
            output[dataset_name][data_name] = dataset_info["local_paths"][data_name]
    return output


def set_session_state(dataset_name: str, category_name: str, filter_by: str, improvement_flag: str):
    """Set session state in order to keep track of a current query image number. Changing dataset,
    category or filter settings resets current query number to 0."""
    if "query_controller_position" not in st.session_state:
        st.session_state.query_controller_position = 0
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = dataset_name
    if "category" not in st.session_state:
        st.session_state.category = category_name
    if "filter_by" not in st.session_state:
        st.session_state.filter_by = filter_by
    if "improvement_flag" not in st.session_state:
        st.session_state.improvement_flag = improvement_flag
    if st.session_state.category != category_name:
        st.session_state.category = category_name
        st.session_state.query_controller_position = 0
    if st.session_state.dataset_name != dataset_name:
        st.session_state.dataset_name = dataset_name
        st.session_state.category = category_name
        st.session_state.query_controller_position = 0
    if st.session_state.filter_by != filter_by:
        st.session_state.filter_by = filter_by
        st.session_state.query_controller_position = 0
    if st.session_state.improvement_flag != improvement_flag:
        st.session_state.improvement_flag = improvement_flag
        st.session_state.query_controller_position = 0


def get_filter_options(dataset: QueryDataset) -> Dict[str, Union[None, str]]:
    options: Dict[str, Union[None, str]] = {
        c.split("_")[0]: c
        for c in dataset.columns
        if c.endswith(IMPROVED_SUFFIX) and c not in METRICS_TO_EXCLUDE_FROM_VIEWER
    }
    return options


if __name__ == "__main__":
    main()
