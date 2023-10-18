from typing import Dict, List, Optional, Union

from dataset_tools.templates import (
    AnnotationType,
    Category,
    CVTask,
    Domain,
    Industry,
    License,
    Research,
)

##################################
# * Before uploading to instance #
##################################
PROJECT_NAME: str = "DOTA"
PROJECT_NAME_FULL: str = (
    "DOTA v2.0: Dataset of Object deTection in Aerial images"
)
HIDE_DATASET = False  # set False when 100% sure about repo quality

##################################
# * After uploading to instance ##
##################################
LICENSE: License = License.Custom(url="https://captain-whu.github.io/DOTA/dataset.html")
APPLICATIONS: List[Union[Industry, Domain, Research]] = [Domain.Geospatial()]
CATEGORY: Category = Category.Aerial(extra=Category.Satellite())

CV_TASKS: List[CVTask] = [CVTask.ObjectDetection()]
ANNOTATION_TYPES: List[AnnotationType] = [AnnotationType.ObjectDetection()]

RELEASE_DATE: Optional[str] = "2021-02-25"  # e.g. "YYYY-MM-DD"
if RELEASE_DATE is None:
    RELEASE_YEAR: int = None

HOMEPAGE_URL: str = "https://captain-whu.github.io/DOTA/index.html"
# e.g. "https://some.com/dataset/homepage"

PREVIEW_IMAGE_ID: int = 7003618
# This should be filled AFTER uploading images to instance, just ID of any image.

GITHUB_URL: str = "https://github.com/dataset-ninja/dota"
# URL to GitHub repo on dataset ninja (e.g. "https://github.com/dataset-ninja/some-dataset")

##################################
### * Optional after uploading ###
##################################
DOWNLOAD_ORIGINAL_URL: Optional[
    Union[str, dict]
] = "https://captain-whu.github.io/DOTA/dataset.html"
# Optional link for downloading original dataset (e.g. "https://some.com/dataset/download")

CLASS2COLOR: Optional[Dict[str, List[str]]] = {
    "plane": [230, 25, 75],
    "ship": [60, 180, 75],
    "storage tank": [255, 225, 25],
    "baseball diamond": [0, 130, 200],
    "tennis court": [245, 130, 48],
    "basketball court": [145, 30, 180],
    "ground track field": [70, 240, 240],
    "harbor": [240, 50, 230],
    "bridge": [210, 245, 60],
    "large vehicle": [250, 190, 212],
    "small vehicle": [0, 128, 128],
    "helicopter": [220, 190, 255],
    "roundabout": [170, 110, 40],
    "soccer ball field": [255, 250, 200],
    "swimming pool": [128, 0, 0],
    "container crane": [170, 255, 195],
    "airport": [128, 128, 0],
    "helipad": [255, 215, 180],
}
# If specific colors for classes are needed, fill this dict (e.g. {"class1": [255, 0, 0], "class2": [0, 255, 0]})

# If you have more than the one paper, put the most relatable link as the first element of the list
# Use dict key to specify name for a button
PAPER: Optional[Union[str, List[str], Dict[str, str]]] = [
    "https://arxiv.org/abs/2102.12219",
    "https://ieeexplore.ieee.org/document/8578516",
    "https://arxiv.org/abs/1812.00155",
]
BLOGPOST: Optional[Union[str, List[str], Dict[str, str]]] = None
REPOSITORY: Optional[Union[str, List[str], Dict[str, str]]] = {"GitHub":"https://github.com/CAPTAIN-WHU/DOTA"}

CITATION_URL: Optional[str] = "https://captain-whu.github.io/DOTA/index.html"
AUTHORS: Optional[List[str]] = [
    "Jian Ding",
    "Nan Xue",
    "Gui-Song Xia",
    "Xiang Bai",
    "Wen Yang",
    "Micheal Ying Yang",
    "Serge Belongie",
    "Jiebo Luo",
    "Mihai Datcu",
    "Marcello Pelillo",
    "Liangpei Zhang",
]

ORGANIZATION_NAME: Optional[Union[str, List[str]]] = "CHI-NLD-USA-GER-ITL joint research group"
ORGANIZATION_URL: Optional[Union[str, List[str]]] = "https://captain-whu.github.io/DOTA/people.html"

# Set '__PRETEXT__' or '__POSTTEXT__' as a key with string value to add custom text. e.g. SLYTAGSPLIT = {'__POSTTEXT__':'some text}
SLYTAGSPLIT: Optional[Dict[str, Union[List[str], str]]] = {
    "__PRETEXT__": "Additionally, images contain meta-info about ***acquisition date***, ***image source***, and ***ground sample distance***, while every OBB has boolean ***difficult*** tag"
}
TAGS: Optional[List[str]] = None


SECTION_EXPLORE_CUSTOM_DATASETS: Optional[List[str]] = None

##################################
###### ? Checks. Do not edit #####
##################################


def check_names():
    fields_before_upload = [PROJECT_NAME]  # PROJECT_NAME_FULL
    if any([field is None for field in fields_before_upload]):
        raise ValueError("Please fill all fields in settings.py before uploading to instance.")


def get_settings():
    if RELEASE_DATE is not None:
        global RELEASE_YEAR
        RELEASE_YEAR = int(RELEASE_DATE.split("-")[0])

    settings = {
        "project_name": PROJECT_NAME,
        "project_name_full": PROJECT_NAME_FULL or PROJECT_NAME,
        "hide_dataset": HIDE_DATASET,
        "license": LICENSE,
        "applications": APPLICATIONS,
        "category": CATEGORY,
        "cv_tasks": CV_TASKS,
        "annotation_types": ANNOTATION_TYPES,
        "release_year": RELEASE_YEAR,
        "homepage_url": HOMEPAGE_URL,
        "preview_image_id": PREVIEW_IMAGE_ID,
        "github_url": GITHUB_URL,
    }

    if any([field is None for field in settings.values()]):
        raise ValueError("Please fill all fields in settings.py after uploading to instance.")

    settings["release_date"] = RELEASE_DATE
    settings["download_original_url"] = DOWNLOAD_ORIGINAL_URL
    settings["class2color"] = CLASS2COLOR
    settings["paper"] = PAPER
    settings["blog"] = BLOGPOST
    settings["repository"] = REPOSITORY
    settings["citation_url"] = CITATION_URL
    settings["authors"] = AUTHORS
    settings["organization_name"] = ORGANIZATION_NAME
    settings["organization_url"] = ORGANIZATION_URL
    settings["slytagsplit"] = SLYTAGSPLIT
    settings["tags"] = TAGS

    settings["explore_datasets"] = SECTION_EXPLORE_CUSTOM_DATASETS

    return settings
