import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse

actualMaskDir = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
    "tmp",
    "training",
    "annotations_prepped_test",
)


def readImage(path: str) -> np.ndarray:
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x / 255.0
    return x


def readMask(path: str) -> np.ndarray:
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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


def smoothenMask(mask: np.ndarray) -> np.ndarray:
    blur = np.where(mask == 1, 255, 0)
    blur = cv2.GaussianBlur(blur.astype(np.uint8), (7, 7), 20)
    blur = np.where(blur >= 128, 1, 0)
    return blur


def applyMask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    blur = smoothenMask(mask).astype(np.uint8)
    b, g, r = cv2.split(image)
    b = cv2.multiply(b.astype(np.uint8), blur)
    g = cv2.multiply(g.astype(np.uint8), blur)
    r = cv2.multiply(r.astype(np.uint8), blur)
    result = cv2.merge((b, g, r))
    return result


def segmentImages(modelName: str, data: list) -> list:
    resultMaskDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "tmp",
        modelName,
    )

    results = []

    for i, filename in tqdm(enumerate(data)):
        x = readImage(filename)
        y = readMask(os.path.join(resultMaskDir, f"out_{os.path.basename(filename)}"))
        unique = np.unique(y)
        y = np.where(y == unique[0], 0, 255)
        masked_x_predicted = applyMask(
            image=x * 255.0, mask=np.where(y == 255, 1, 0)
        )
        y_orginal = readMask(os.path.join(actualMaskDir, os.path.basename(filename)))
        masked_x_orginal = applyMask(image=x * 255.0, mask=y_orginal)

        results.append(
            {
                "x": x,
                "original": masked_x_orginal,
                "predicted": masked_x_predicted,
            }
        )

    return results


def compareModels(data: list) -> None:
    compareModelDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "output",
        "compare",
    )
    os.makedirs(compareModelDir, exist_ok=True)

    models = ["unet", "vgg_unet", "resnet50_unet", "segnet"]

    # Output: original image, expected mask, segmented image of model 1, ... model2, ...

    finalResults = {}

    for model in models:
        results = segmentImages(modelName=model, data=data)
        if len(results) == 0:
            print("No test images found.")
            sys.exit(1)

        for i, result in enumerate(results):
            if i not in finalResults:
                finalResults[i] = {}
            
            finalResults[i]["x"] = result["x"]
            finalResults[i]["original"] = result["original"]
            finalResults[i][model] = result["predicted"]

    for i, key in enumerate(finalResults):
        h, w, _ = finalResults[key]["x"].shape
        white_line = np.ones((h, 10, 3)) * 255.0
        allImages = [
            finalResults[key]["x"] * 255.0,
            white_line,
            finalResults[key]["original"],
            white_line,
        ]
        for model in models:
            allImages.extend([finalResults[key][model], white_line])

        image = np.concatenate(allImages, axis=1)
        cv2.imwrite(os.path.join(compareModelDir, f"{i}.png"), image)

    print(models)


def __main(modelName: str) -> None:
    path = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "tmp",
        "training",
    )

    outputDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "output", modelName
    )
    os.makedirs(outputDir, exist_ok=True)

    data = load_data(path=path)

    if modelName == "compare":
        return compareModels(data)

    resultMaskDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "tmp",
        modelName,
    )

    for i, filename in tqdm(enumerate(data)):
        x = readImage(filename)
        y = readMask(os.path.join(resultMaskDir, f"out_{os.path.basename(filename)}"))
        bgrMask = readImage(os.path.join(resultMaskDir, f"out_{os.path.basename(filename)}"))
        unique = np.unique(y)
        y = np.where(y == unique[0], 0, 255)
        masked_x_predicted = applyMask(
            image=x * 255.0, mask=np.where(y == 255, 1, 0)
        )
        y_orginal = readMask(os.path.join(actualMaskDir, os.path.basename(filename)))
        masked_x_orginal = applyMask(image=x * 255.0, mask=y_orginal)
        y_orginal = np.where(y_orginal == 1, 255, 0)
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0
        allImages = [x * 255.0, white_line, bgrMask * 255.0]
        # allImages = [
        #     x * 255.0,
        #     white_line,
        #     mask_parse(y),
        #     white_line,
        #     mask_parse(y_orginal),
        #     white_line,
        #     masked_x_predicted,
        #     white_line,
        #     masked_x_orginal,
        # ]
        image = np.concatenate(allImages, axis=1)
        cv2.imwrite(os.path.join(outputDir, f"{i}.png"), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Postprocessing of output from the segmentation models"
    )
    parser.add_argument(
        "--model",
        metavar="model",
        type=str,
        help="Name of the model. Available models - unet, vgg_unet, resnet50_unet, segnet, compare(compares all models)",
    )
    args = parser.parse_args()
    __main(modelName=args.model)

    # batch_size = 8
