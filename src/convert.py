# https://captain-whu.github.io/DOTA/dataset.html

import os

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dotenv import load_dotenv
from PIL import Image
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
)
from supervisely.io.json import load_json_file

Image.MAX_IMAGE_PIXELS = None

import os
import shutil
from urllib.parse import unquote, urlparse

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import get_file_name, get_file_size
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(desc=f"Downloading '{file_name_with_ext}' to buffer...", total=fsize) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "DOTA-v2.0"

    # TODO сделать корректный ковнертер без необходимости предварительного перетасования файлов
    train_anns_path = "/mnt/d/datasetninja-raw/dota/2_0/train/labelTxt-v2.0/DOTA-v2.0_train"
    val_anns_path = "/mnt/d/datasetninja-raw/dota/2_0/val/labelTxt-v2.0/DOTA-v2.0_val"
    val_images_path = "/mnt/d/datasetninja-raw/dota/2_0/val/images"
    train_images_path = "/mnt/d/datasetninja-raw/dota/2_0/train/images"

    val_images_meta_path = "/mnt/d/datasetninja-raw/dota/2_0/val/labelTxt-v2.0/val_meta/meta"
    train_images_meta_path = "/mnt/d/datasetninja-raw/dota/2_0/train/labelTxt-v2.0/train_meta/meta"

    batch_size = 5
    test_images_path = "/mnt/d/datasetninja-raw/dota/2_0/test-dev"
    test_images_meta_path = "/mnt/d/datasetninja-raw/dota/2_0/test-dev/test-dev_meta/meta"

    anns_ext = ".txt"
    meta_ext = ".txt"
    img_ext = ".png"

    name_to_path = {
        "train": (train_images_path, train_anns_path, train_images_meta_path),
        "val": (val_images_path, val_anns_path, val_images_meta_path),
        "test-dev": (test_images_path, None, test_images_meta_path),
    }

    def create_ann(image_path):
        labels = []
        tags = []

        image = Image.open(image_path)
        img_height = image.height
        img_wight = image.width

        if ds_name != "test-dev":
            ann = os.path.join(anns_path, get_file_name(image_path) + anns_ext)
            with open(ann) as f:
                content = f.read().split("\n")
                for curr_data in content:
                    if len(curr_data) == 0:
                        continue
                    curr_data = curr_data.split(" ")
                    class_name = curr_data[-2].replace("-", " ")
                    obj_class = meta.get_obj_class(class_name)
                    coords = list(map(float, curr_data[:-2]))

                    exterior = []
                    for i in range(0, len(coords), 2):
                        exterior.append([int(coords[i + 1]), int(coords[i])])
                    if len(exterior) < 3:
                        continue
                    poligon = sly.Polygon(exterior)

                    label_tags = [sly.Tag(tag_difficult, value=map_tag_difficult[curr_data[-1]])]
                    label_poly = sly.Label(poligon, obj_class, tags=label_tags)
                    labels.append(label_poly)

        met = os.path.join(meta_path, get_file_name(image_path) + meta_ext)
        with open(met) as f:
            content = f.read().split("\n")
            for curr_data in content:
                if len(curr_data) == 0:
                    continue
                curr_data = curr_data.split(":")
                key, value = curr_data[0], curr_data[1]
                if value not in ["null", "None"]:
                    try:
                        tags.append(sly.Tag(tagkey_to_tagmeta[key], value=value))
                    except:
                        tags.append(sly.Tag(tagkey_to_tagmeta[key], value=float(value)))
                else:
                    pass

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    obj_class_plane = sly.ObjClass("plane", sly.Polygon)
    obj_class_ship = sly.ObjClass("ship", sly.Polygon)
    obj_class_tank = sly.ObjClass("storage tank", sly.Polygon)
    obj_class_baseball = sly.ObjClass("baseball diamond", sly.Polygon)
    obj_class_tennis = sly.ObjClass("tennis court", sly.Polygon)
    obj_class_basketball = sly.ObjClass("basketball court", sly.Polygon)
    obj_class_track = sly.ObjClass("ground track field", sly.Polygon)
    obj_class_harbor = sly.ObjClass("harbor", sly.Polygon)
    obj_class_bridge = sly.ObjClass("bridge", sly.Polygon)
    obj_class_large = sly.ObjClass("large vehicle", sly.Polygon)
    obj_class_small = sly.ObjClass("small vehicle", sly.Polygon)
    obj_class_helicopter = sly.ObjClass("helicopter", sly.Polygon)
    obj_class_roundabout = sly.ObjClass("roundabout", sly.Polygon)
    obj_class_soccer = sly.ObjClass("soccer ball field", sly.Polygon)
    obj_class_pool = sly.ObjClass("swimming pool", sly.Polygon)
    obj_class_crane = sly.ObjClass("container crane", sly.Polygon)
    obj_class_airport = sly.ObjClass("airport", sly.Polygon)
    obj_class_helipad = sly.ObjClass("helipad", sly.Polygon)

    tag_acquisition_date = sly.TagMeta("acquisition date", sly.TagValueType.ANY_STRING)
    tag_imagesource = sly.TagMeta("image source", sly.TagValueType.ANY_STRING)
    tag_gsd = sly.TagMeta("ground sample distance", sly.TagValueType.ANY_NUMBER)

    tag_difficult = sly.TagMeta(
        "difficult", sly.TagValueType.ONEOF_STRING, possible_values=["True", "False"]
    )
    map_tag_difficult = {"0": "False", "1": "True"}

    tagkey_to_tagmeta = {
        "acquisition dates": tag_acquisition_date,
        "acquisition date": tag_acquisition_date,
        "imagesource": tag_imagesource,
        "gsd": tag_gsd,
    }

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[
            obj_class_plane,
            obj_class_ship,
            obj_class_tank,
            obj_class_baseball,
            obj_class_tennis,
            obj_class_basketball,
            obj_class_track,
            obj_class_harbor,
            obj_class_bridge,
            obj_class_large,
            obj_class_small,
            obj_class_helicopter,
            obj_class_roundabout,
            obj_class_soccer,
            obj_class_pool,
            obj_class_crane,
            obj_class_airport,
            obj_class_helipad,
        ],
        tag_metas=[tag_acquisition_date, tag_imagesource, tag_gsd, tag_difficult],
    )
    api.project.update_meta(project.id, meta.to_json())

    for ds_name in ["train", "val", "test-dev"]:
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        images_path = name_to_path[ds_name][0]
        anns_path = name_to_path[ds_name][1]
        meta_path = name_to_path[ds_name][2]

        images_names = [name for name in os.listdir(images_path) if name.endswith(img_ext)]

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [os.path.join(images_path, im_name) for im_name in img_names_batch]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(img_names_batch))
    return project
