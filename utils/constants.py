from enum import Enum


TEST_FILES_PREFIX = "/images/test"
TRAIN_FILES_PREFIX = "/images/train"
VALID_FILES_PREFIX = "/images/valid"

IMAGES_PREFIX = "images"
LABELS_PREFIX = "labels"

CLASSES = [
    "elbow positive",
    "fingers positive",
    "forearm fracture",
    "humerus fracture",
    "humerus",
    "shoulder fracture",
    "wrist positive",
]


class GroupName(str, Enum):
    all = "all"
    train = "train"
    test = "test"
    valid = "valid"
