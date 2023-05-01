import os
import glob
import shutil
import cv2
import numpy as np

THRESHOLD = 128

dir = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "PNG"
)


def saveAnnotation(coords: tuple, filename: str, saveDir: os.path, width: int = 608, height: int = 416) -> None:
    (xMin, xMax, yMin, yMax) = coords

    annotation = f'''
    <annotation>
        <folder>pascalVOCAnnotations</folder>
        <filename>{filename}</filename>
        <path>{os.path.join(saveDir, filename)}</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>polyp</name>
            <pose>Frontal</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <occluded>0</occluded>
            <bndbox>
                <xmin>{xMin}</xmin>
                <xmax>{xMax}</xmax>
                <ymin>{yMin}</ymin>
                <ymax>{yMax}</ymax>
            </bndbox>
        </object>
    </annotation>
    '''

    saveFilename = filename.split('.')[0] + '.xml'
    with open(os.path.join(saveDir, saveFilename), "w") as file:
        file.write(annotation)

def countImagesinDir(src : str) -> int:
    count = 0
    for filename in os.listdir(src):
        if not (filename.endswith(".png") or filename.endswith(".jpg")):
            continue
        count +=1

    return count

def trainTestSplitInfo(srcImgPath: str, trainSplitPercent: int, saveDir: os.path) -> None:

    testLabelFile = "test.txt"
    trainLabelFile = "trainval.txt"

    # calculate the train test split size
    totalImages = countImagesinDir(srcImgPath)
    trainDataSize = int((trainSplitPercent/100) * totalImages)

    print(f'total {totalImages} images found \n \
          training images count : {trainDataSize} \n \
          test images count : {totalImages - trainDataSize} ')

    with open(os.path.join(saveDir, testLabelFile), "a") as testFile, \
        open(os.path.join(saveDir, trainLabelFile), "a") as trainFile:
        
        for idx,filename in enumerate(os.listdir(srcImgPath)):
            if not (filename.endswith(".png") or filename.endswith(".jpg")):
                continue

            if idx < trainDataSize:
                trainFile.write(filename.split(".")[0] + '\n')
            else:
                testFile.write(filename.split('.')[0] + '\n')
        

def copyImagesToDataset(srcImgPath: str, imageSaveDir: str) -> None:
    for jpgfile in glob.iglob(os.path.join(srcImgPath, "*.png")):
        shutil.copy(jpgfile, imageSaveDir)

def main() -> None:
    srcImgPath = os.path.join(dir, "Original")
    srcMaskPath = os.path.join(dir, "Ground_Truth")
    
    saveBaseDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "PolypDataset_SSD",
    )

    annotationsSaveDir = os.path.join(saveBaseDir,  "Annotations")
    os.makedirs(annotationsSaveDir, exist_ok=True)

    # directory for saving train test split info
    splitInfoSaveDir = os.path.join(saveBaseDir,  "ImageSets/Main")
    os.makedirs(splitInfoSaveDir, exist_ok=True)

    # directory for storing images
    imageSaveDir = os.path.join(saveBaseDir, "PngImages")
    os.makedirs(imageSaveDir, exist_ok=True)
    copyImagesToDataset(srcImgPath, imageSaveDir)
    
    # create the train test split info text file
    trainTestSplitInfo(srcImgPath, 80, splitInfoSaveDir)

    for filename in os.listdir(srcImgPath):
        if not (filename.endswith(".png") or filename.endswith(".jpg")):
            continue
        
        mask = cv2.imread(os.path.join(srcMaskPath, filename))
        possibleXs = [np.argmax(x > THRESHOLD) for x in mask]
        xMin = min(possibleXs)
        xMax = max(possibleXs)

        mask.transpose()
        possibleYs = [np.argmax(y > THRESHOLD) for y in mask]
        yMin = min(possibleYs)
        yMax = max(possibleYs)

        saveAnnotation((xMin, xMax, yMin, yMax), filename, annotationsSaveDir)

if __name__ == "__main__":
    main()