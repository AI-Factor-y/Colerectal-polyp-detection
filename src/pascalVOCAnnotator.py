import os
import glob
import shutil
import cv2
import numpy as np
import sys
from PIL import Image
from tqdm import tqdm
import argparse

THRESHOLD = 128

# dir = os.path.join(
#     os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "PNG"
# )

def setDatasetPath(relativeDirPath) -> str:
    dir = os.path.join(
        os.path.abspath(os.getcwd()), relativeDirPath
    )

    return dir

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

    print("writing test and train split info text files..")
    with open(os.path.join(saveDir, testLabelFile), "a") as testFile, \
        open(os.path.join(saveDir, trainLabelFile), "a") as trainFile:
        
        for idx,filename in tqdm(enumerate(os.listdir(srcImgPath))):
            if not (filename.endswith(".png") or filename.endswith(".jpg")):
                continue

            if idx < trainDataSize:
                trainFile.write(filename.split(".")[0] + '\n')
            else:
                testFile.write(filename.split('.')[0] + '\n')
        

def copyImagesToDataset(srcImgPath: str, imageSaveDir: str) -> None:
    for jpgfile in glob.iglob(os.path.join(srcImgPath, "*.png")):
        shutil.copy(jpgfile, imageSaveDir)



def convertJpgToPng(jpg_file_path : str) -> None:
    with Image.open(jpg_file_path) as im:
        # Convert the image to RGB mode (if necessary)
        if im.mode != "RGB":
            im = im.convert("RGB")
        
        # Get the file name and extension
        file_name, file_ext = os.path.splitext(jpg_file_path)
        
        # Get the parent directory of the JPG file
        parent_dir = os.path.dirname(jpg_file_path)
        
        # Create the "PNG" folder if it doesn't exist
        png_folder = os.path.join(parent_dir, "PNG")
        os.makedirs(png_folder, exist_ok=True)
        
        # Save the image as a PNG file in the "PNG" folder
        png_file_path = os.path.join(png_folder, os.path.basename(file_name) + ".png")

        im.save(png_file_path, "PNG")

def convertAllImagesToPng(src):
    for filename in tqdm(os.listdir(src)):
        if not (filename.endswith(".jpg")):
            continue
        convertJpgToPng(os.path.join(src,filename))

def rowBoundValue(imgMask):
    imageWidth= imgMask.shape[1]

    grey_values = np.mean(imgMask, axis=2)
    mask = (grey_values > THRESHOLD) & (imgMask[:, :, 0] != 0) & (imgMask[:, :, 0] != imageWidth)

    nonzero_indices = np.nonzero(mask)
    if nonzero_indices[0].size == 0:
        return -1, -1

    
    minX = np.min(nonzero_indices[1])
    maxX = np.max(nonzero_indices[1])

    return minX, maxX


def main() -> None:

    # Create a new ArgumentParser object
    parser = argparse.ArgumentParser()
    # Add the boolean argument
    parser.add_argument("--convert", action="store_const", const=True, default=False, help="convert jpg to png")
    parser.add_argument("-original", type=str, required=True, help="relative path of original image directory")
    parser.add_argument("-mask", type=str, required=True, help="relative path of mask image directory")
    # Parse the command line arguments
    args = parser.parse_args()

    srcImgPath = setDatasetPath(args.original)
    srcMaskPath = setDatasetPath(args.mask)

    print("image path  : ", srcImgPath)
    print("mask path  : ", srcMaskPath)
    isImageJpg = args.convert
    print("convert to png : ", isImageJpg)

    # if the images are not in png convert them to png first
    if isImageJpg:
        print("converting original images to png")
        convertAllImagesToPng(srcImgPath)
        srcImgPath = srcImgPath+ "/PNG"
        print("converting masks to png")
        convertAllImagesToPng(srcMaskPath)
        srcMaskPath = srcMaskPath + "/PNG"

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
    trainTestSplitInfo(srcImgPath, 90, splitInfoSaveDir)

    print("creating annotations xml files...")
    for filename in tqdm(os.listdir(srcImgPath)):
        if not (filename.endswith(".png") or filename.endswith(".jpg")):
            continue
        
        imageOrig = cv2.imread(os.path.join(srcImgPath, filename))
        imageHeight, imageWidth, _ = imageOrig.shape

        mask = cv2.imread(os.path.join(srcMaskPath, filename))
        xMin, xMax = rowBoundValue(mask)
        
        mask = np.transpose(mask, (1,0,2))
        yMin, yMax = rowBoundValue(mask)
        
        # print("bounds for ",filename," : ",xMin, xMax, yMin, yMax)
        saveAnnotation((xMin, xMax, yMin, yMax), filename, annotationsSaveDir, imageWidth, imageHeight)

if __name__ == "__main__":
    main()