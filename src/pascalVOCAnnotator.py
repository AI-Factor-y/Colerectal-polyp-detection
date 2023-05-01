import os
import cv2
import numpy as np

THRESHOLD = 128

dir = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "datasets", "Orginal"
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
            <name>21</name>
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

    saveFilename = filename.split('.')[0] + '.yaml'
    with open(os.path.join(saveDir, saveFilename), "w") as file:
        file.write(annotation)


def main() -> None:
    srcImgPath = os.path.join(dir, "imgs")
    srcMaskPath = os.path.join(dir, "masks")
    
    saveBaseDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "tmp",
        "training",
    )

    saveDir = os.path.join(saveBaseDir,  "pascalVOCAnnotations")
    os.makedirs(saveDir, exist_ok=True)

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

        saveAnnotation((xMin, xMax, yMin, yMax), filename, saveDir)

if __name__ == "__main__":
    main()