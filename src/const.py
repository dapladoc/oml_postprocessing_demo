from enum import Enum
from pathlib import Path
from typing import Union

DATASETS = {
    "DeepFashion InShop": {
        "gdrive_ids": {
            "query": "1ipcsWWvGEsHwr65m2UHmXSik6a1KRDjm",
            "gallery": "1t5mNvWqPSIhjiJ6_0NQvNiSwEv1FMbtp",
            "zip": "1ew1wkG3mub70THUozWQylgmbjyYQl1TK",
        },
        "local_paths": {
            "query": "tmp/data/InShop/filtered_query_df.csv",
            "gallery": "tmp/data/InShop/filtered_gallery_df.csv",
            "zip": "tmp/data/InShop/InShop_images.zip",
        },
    },
    "Stanford Online Products": {
        "gdrive_ids": {
            "query": "1vZYr2y0IMDDjOyLTAk1j2mAVN74yBxOq",
            "gallery": "1YGQ2TKM8NTWXdsAurKN5EYi9_BvXcXzz",
            "zip": "1KrJOVjKqupG8rEG8-cmfRHCg3If9ABUH",
        },
        "local_paths": {
            "query": "tmp/data/SOP/filtered_query_df.csv",
            "gallery": "tmp/data/SOP/filtered_gallery_df.csv",
            "zip": "tmp/data/SOP/SOP_images.zip",
        },
    },
}
LOCAL_DATASETS = {
    "Stanford Online Products": {
        "query": "data/SOP/filtered_query_df.csv",
        "gallery": "data/SOP/filtered_gallery_df.csv",
        "zip": "data/SOP/SOP_images.zip",
    },
    "DeepFashion InShop": {
        "query": "data/InShop/filtered_query_df.csv",
        "gallery": "data/InShop/filtered_gallery_df.csv",
        "zip": "data/InShop/InShop_images.zip",
    },
}
MAIN_PAGE_TITLE = r"[STIR: Siamese Transformer for Image Retrieval Postprocessing](https://arxiv.org/abs/2304.13393)"
MAIN_PAGE_ABSTRACT = """In this work, we first construct a simple Baseline model (trained with a triplet loss
with hard negatives mining) and then we introduce a Siamese Transformer for Image Retrieval (STIR) that
reranks top retrieval outputs.
It takes pairs of a query and each image in top outputs, concatenates them in pixel space, and re-estimate
the distance between them. The pixel-to-pixel comparison allows STIR to improve the original retrieval metrics of
the Baseline model.

Implemented in [OpenMetricLearning](https://github.com/OML-Team/open-metric-learning/tree/docs/pipelines/postprocessing/pairwise_postprocessing)
"""
MAIN_PAGE_IMAGE_PATH = "src/app/assets/pic_1.jpg"
LABELS_COLUMN = "label"
PATHS_COLUMN = "path"
SPLIT_COLUMN = "split"
IS_QUERY_COLUMN = "is_query"
IS_GALLERY_COLUMN = "is_gallery"
CATEGORIES_COLUMN = "category"
ID_COLUMN = "image_id"
IMPROVED_SUFFIX = "_improved"
IMAGE_ID_SUFFIX = "_image_id"
SIMPLE_IMPROVED_COLUMN = f"simple{IMPROVED_SUFFIX}"
SIZE = 256
BORDER_SIZE = 6
TOP_K_SCORE_COLUMN_TEMPLATE = "top_%i_score"
POSTPROCESSED_TOP_K_SCORE_COLUMN_TEMPLATE = "postprocessed_top_%i_score"
TOP_K_IMAGE_ID_COLUMN_TEMPLATE = f"top_%i{IMAGE_ID_SUFFIX}"
POSTPROCESSED_TOP_K_IMAGE_ID_COLUMN_TEMPLATE = f"postprocessed_top_%i{IMAGE_ID_SUFFIX}"
CMC_TOP_K_COLUMN_TEMPLATE = "cmc@%i"
POSTPROCESSED_CMC_TOP_K_COLUMN_TEMPLATE = "postprocessed_cmc@%i"
CMC_IMPROVED_COLUMN_TEMPLATE = f"cmc@%i{IMPROVED_SUFFIX}"
MAP_TOP_K_COLUMN_TEMPLATE = "map@%i"
POSTPROCESSED_MAP_TOP_K_COLUMN_TEMPLATE = "postprocessed_map@%i"
MAP_IMPROVED_COLUMN_TEMPLATE = f"map@%i{IMPROVED_SUFFIX}"
RED_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
IMPROVEMENT_FLAG_VALUE = 1
WORSENING_FLAG_VALUE = -1
WITHOUT_CHANGE_FLAG_VALUE = 0
PathType = Union[Path, str]
TOP_K = 5
FILTER_OPTION_NONE = "None"
METRICS_TO_EXCLUDE_FROM_VIEWER = [
    SIMPLE_IMPROVED_COLUMN,
]


class ImprovementFlags(Enum):
    all = "Show all"
    improvements = "Show only improvements"
    worsenings = "Show only worsenings"


class RetrievalResultsType(Enum):
    before_stir = "before stir"
    after_stir = "after stir"
