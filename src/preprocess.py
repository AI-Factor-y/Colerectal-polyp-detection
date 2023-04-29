import os
import cv2
import shutil
import random
import numpy as np
from tqdm import tqdm

dir = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "datasets", "Orginal"
)

trainImgDir = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "tmp", "training", "images_prepped_train"
)
trainMaskDir = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
    "tmp",
    "training",
    "annotations_prepped_train",
)

testImgDir = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "tmp", "training", "images_prepped_test"
)
testMaskDir = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
    "tmp",
    "training",
    "annotations_prepped_test",
)

frameSize = (608, 416)
ROTATION_LIMIT = (-20, 20)
TRANSLATION_LIMIT = (-30, 30)
MASK_THRESHOLD = 128 # pixels >= this will be mapped to 1 and < this will be mapped to 0

def rotateImage(image, angle) -> np.ndarray:
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotMat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rotMat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def translateImage(image, tx, ty) -> np.ndarray:
    translationMatrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    translatedImage = cv2.warpAffine(
        src=image, M=translationMatrix, dsize=image.shape[1::-1]
    )
    return translatedImage


def augmentImageAndSave(
    filename: str, srcImgPath: str, srcMaskPath: str, dstImgPath: str, dstMaskPath
) -> None:
    image = cv2.imread(srcImgPath)
    image = cv2.resize(image, frameSize)
    mask = cv2.imread(srcMaskPath)
    mask = cv2.resize(mask, frameSize)

    filename = filename.split(".")[0]
    flippedImg = cv2.flip(image, 1)
    flippedMask = cv2.flip(mask, 1)
    flippedMask = np.where(flippedMask >= MASK_THRESHOLD, 1, 0)

    saveImgPath = os.path.join(dstImgPath, f"{filename}_f.png")
    saveMaskPath = os.path.join(dstMaskPath, f"{filename}_f.png")

    cv2.imwrite(saveImgPath, flippedImg)
    cv2.imwrite(saveMaskPath, flippedMask)

    for angle in range(*ROTATION_LIMIT, 1):
        if angle == 0:
            continue
        rX = random.randint(*TRANSLATION_LIMIT)
        rY = random.randint(*TRANSLATION_LIMIT)
        translatedImg = translateImage(image, rX, rY)
        translatedMask = translateImage(mask, rX, rY)
        rotatedImg = rotateImage(translatedImg, angle)
        rotatedMask = rotateImage(translatedMask, angle)
        rotatedMask = np.where(rotatedMask >= MASK_THRESHOLD, 1, 0)

        saveImgPath = os.path.join(dstImgPath, f"{filename}_r{angle}.png")
        saveMaskPath = os.path.join(dstMaskPath, f"{filename}_r{angle}.png")
        cv2.imwrite(saveImgPath, rotatedImg)
        cv2.imwrite(saveMaskPath, rotatedMask)


def saveTestDataset(filenames: list[str]) -> None:
    imgPath = os.path.join(dir, "imgs")
    maskPath = os.path.join(dir, "masks")

    for filename in filenames:
        shutil.copy(os.path.join(imgPath, filename), testImgDir)
        shutil.copy(
            os.path.join(maskPath, filename), testMaskDir
        )


def main():
    srcImgPath = os.path.join(dir, "imgs")
    srcMaskPath = os.path.join(dir, "masks")

    saveBaseDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "tmp",
        "training",
    )

    saveDirs = [os.path.join(saveBaseDir,  "images_prepped_train"), os.path.join(saveBaseDir,  "images_prepped_test"), os.path.join(saveBaseDir,  "annotations_prepped_train"), os.path.join(saveBaseDir,  "annotations_prepped_test")]
    
    for saveDir in saveDirs:
        os.makedirs(saveDir, exist_ok=True)

    imgs = []
    for filename in os.listdir(srcImgPath):
        if not (filename.endswith(".png") or filename.endswith(".jpg")):
            continue
        imgs.append(filename)

    random.shuffle(imgs)
    testLen = round(len(imgs) * 0.2)
    test = imgs[:testLen]
    train = imgs[testLen:]

    saveTestDataset(filenames=test)

    for filename in tqdm(train):
        try:
            augmentImageAndSave(
                filename=filename,
                srcImgPath=os.path.join(srcImgPath, filename),
                srcMaskPath=os.path.join(srcMaskPath, filename),
                dstImgPath=trainImgDir,
                dstMaskPath=trainMaskDir,
            )
        except Exception:
            continue


if __name__ == "__main__":
    main()
