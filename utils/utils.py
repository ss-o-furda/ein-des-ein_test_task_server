import os
from pathlib import Path
from typing import Iterable

from utils.aliases import Classes, Polygon, Polygons, ResObject
from utils.constants import (
    CLASSES,
    IMAGES_PREFIX,
    LABELS_PREFIX,
    TEST_FILES_PREFIX,
    TRAIN_FILES_PREFIX,
    VALID_FILES_PREFIX,
    GroupName,
)


def get_class_by_id(id: int) -> str:
    """Returns the class label corresponding to the given class ID.

    Args:
        id (int): The ID of the class.

    Returns:
        str: The class label corresponding to the given ID.
    """
    return CLASSES[id]


def get_class_id(line: str | list[str]) -> int:
    """Extracts the class ID from the given line.

    Args:
        line (str): A string representing a line of data.

    Returns:
        str: The class ID extracted from the line.
    """
    return int(line[0])


def get_dir_files(dir: str) -> list[os.DirEntry]:
    """Returns a list of files in the specified directory.

    Args:
        dir (str): The path in which you want to find the files.

    Returns:
        list[os.DirEntry]: List of DirEntry objects if the entry is a file.
    """
    return list(filter(lambda x: x.is_file(), os.scandir(dir)))


def get_prefix(group: GroupName) -> str:
    """Returns the prefix corresponding to the specified group.

    Args:
        group (GroupName): The group name.

    Returns:
        str: The prefix associated with the group.
    """
    return {
        GroupName.test: TEST_FILES_PREFIX,
        GroupName.train: TRAIN_FILES_PREFIX,
        GroupName.valid: VALID_FILES_PREFIX,
    }.get(group, "")


def is_image_class_in_filter_classes(
    image_classes: Classes, filter_classes: str
) -> bool:
    """Checks if there are intersections between a list of image classes and a string of filter classes.

    Args:
        image_classes (Classes): A list of strings representing the classes of an image.
        filter_classes (str): A string containing comma-separated classes for filtering.

    Returns:
        bool: True if there are intersections between the image classes and filter classes, False otherwise.
    """
    string_set = set(image_classes)
    single_set = set(filter_classes.split(","))

    return bool(string_set.intersection(single_set))


def build_polygon(row: str) -> Polygon:
    """Returns a polygon object based on input data (a line from a YOLOv4 TXT file).

    Args:
        row (str): A line from a text file.

    Returns:
        Polygon: A dictionary representing the polygon object.
    """
    parts = row.strip().split()
    class_idx = get_class_id(parts)
    class_label = get_class_by_id(class_idx)
    vertices = list(map(float, parts[1:]))
    return {
        "class": class_label,
        "vertices": [vertices[i : i + 2] for i in range(0, len(vertices), 2)],
    }


def build_res_object(
    filename: str,
    image_path: str,
    classes: Classes = [],
    polygons: Polygons = [],
) -> ResObject:
    """Builds and returns a result object.

    Args:
        filename (str): The filename.
        image_path (str): The path to the image.
        classes (Classes, optional): List of classes. Defaults to [].
        polygons (Polygons, optional): List of polygons. Defaults to [].

    Returns:
        ResObject: The result object.
    """
    return {
        "filename": filename,
        "image_path": image_path,
        "classes": classes,
        "polygons": polygons,
    }


def process_image(
    image: os.DirEntry, filter_classes: str | None, group: GroupName
) -> ResObject:
    """Processes an image and its corresponding label file.

    Args:
        image (os.DirEntry): A DirEntry object representing the image file.
        filter_classes (str | None, optional): If provided, only polygons belonging to the specified classes will be included in the result.

    Returns:
        ResObject: A dictionary containing the processed image data.
            It includes:
            - 'filename' (str): The filename.
            - 'image_path' (str): The path to the image.
            - 'classes' (Classes): A list of unique classes found in the polygons.
            - 'polygons' (Polygons): A list of dictionaries representing the polygons.
                Each dictionary contains:
                - 'class' (str): The class label of the polygon.
                - 'vertices' (Vertices): A list of vertex coordinates for the polygon.
    """
    prefix = get_prefix(group)

    filename = Path(image.name).stem
    image_path = f"{prefix}/{IMAGES_PREFIX}/{image.name}"
    image_labels_file = f".{prefix}/{LABELS_PREFIX}/{filename}.txt"

    if not os.path.exists(image_labels_file):
        return None

    with open(image_labels_file, "r") as f:
        content = f.readlines()

        if len(content) == 0:
            return build_res_object(filename=filename, image_path=image_path)
        else:
            polygons = list(map(lambda line: build_polygon(line), content))
            classes = list(set(map(lambda poly: poly["class"], polygons)))

            if (
                filter_classes
                and filter_classes.strip()
                and not is_image_class_in_filter_classes(classes, filter_classes)
            ):
                return None

            return build_res_object(
                filename=filename,
                image_path=image_path,
                classes=classes,
                polygons=polygons,
            )


def filter_empty(data: Iterable[ResObject]) -> Iterable[ResObject]:
    """Filters out empty items from the input list.

    Args:
        data (Iterable): Input list.

    Returns:
        Iterable: Filtered list without empty items.
    """
    return filter(lambda x: x is not None, data)


def sort_by_name(data: Iterable[ResObject]) -> Iterable[ResObject]:
    """Sorts a list of dictionaries by the 'filename' key.

    Args:
        data (Iterable): List of dictionaries.

    Returns:
        Iterable: Sorted list of dictionaries by the 'filename' key.
    """
    return sorted(data, key=lambda x: x["filename"])


def prepare_all_data(
    test_images: list[os.DirEntry],
    train_images: list[os.DirEntry],
    valid_images: list[os.DirEntry],
    classes: str,
) -> list[ResObject]:
    """Prepares data for all groups (test, train, and valid).

    Args:
        test_images (list[os.DirEntry]): List of test images.
        train_images (list[os.DirEntry]): List of train images.
        valid_images (list[os.DirEntry]): List of valid images.
        classes (str): Classes to filter the images.

    Returns:
        list[ResObject]:
            A sorted list containing processed images for all groups.
    """
    all_images = [
        *map(lambda image: process_image(image, classes, GroupName.test), test_images),
        *map(
            lambda image: process_image(image, classes, GroupName.train), train_images
        ),
        *map(
            lambda image: process_image(image, classes, GroupName.valid), valid_images
        ),
    ]
    return sort_by_name(filter_empty(all_images))


def prepare_group_data(
    classes: str, group: GroupName, images: list[os.DirEntry]
) -> list[ResObject]:
    """Prepares data for a specific group.

    Args:
        classes (str): Classes to filter the images.
        group (GroupName): The group type (test, train, or valid).
        images (list[os.DirEntry]): List of images for the specified group.

    Returns:
        list[ResObject]:
            A sorted list containing processed images for the specified group.
    """
    processed_images = map(lambda image: process_image(image, classes, group), images)
    filtered_images = filter_empty(processed_images)
    return sort_by_name(filtered_images)


def generate_response(group: GroupName, classes: str) -> list[ResObject]:
    """Generates response data based on the specified group and classes.

    Args:
        group (GroupName): The group type (test, train, valid, or all).
        classes (str): Classes to filter the images.

    Returns:
        list[ResObject]:
            Response data containing processed images based on the group and classes.
    """

    if group is GroupName.all:
        test_images = get_dir_files(f".{TEST_FILES_PREFIX}/{IMAGES_PREFIX}")
        train_images = get_dir_files(f".{TRAIN_FILES_PREFIX}/{IMAGES_PREFIX}")
        valid_images = get_dir_files(f".{VALID_FILES_PREFIX}/{IMAGES_PREFIX}")
        return prepare_all_data(test_images, train_images, valid_images, classes)

    images = get_dir_files(f".{get_prefix(group)}/{IMAGES_PREFIX}")
    return prepare_group_data(classes, group, images)
