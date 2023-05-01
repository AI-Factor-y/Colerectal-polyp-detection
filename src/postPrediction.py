import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
import os


def read_image(path: str) -> np.ndarray:
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x


def read_mask(path: str) -> np.ndarray:
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x


def mask_parse(mask: np.ndarray) -> np.ndarray:
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def load_data(path: str) -> list[np.ndarray]:
    images = sorted(glob(os.path.join(path, "images_prepped_test/*")))
    return images


if __name__ == "__main__":
    ## Dataset
    path = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "tmp",
        "training",
    )
    resultMaskDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "tmp",
    )

    actualMaskDir = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "tmp",
        "training",
        "annotations_prepped_test",
    )

    print(actualMaskDir)

    outputDir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "output")
    os.makedirs(outputDir, exist_ok=True)

    batch_size = 8

    data = load_data(path=path)

    print(data[0])

    for i, filename in tqdm(enumerate(data)):
        x = read_image(filename)
        y = read_mask(os.path.join(resultMaskDir, filename))
        y_orginal = read_mask(os.path.join(actualMaskDir, os.path.basename(filename)))
        y = np.where(y > 128, 255, 0)
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0

        all_images = [x * 255.0, white_line, mask_parse(y), white_line, mask_parse(y_orginal)]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(os.path.join(outputDir, f"{i}.png"), image)
