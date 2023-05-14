import os
import cv2
from tqdm import tqdm

def main():
    imagesDir = "images"
    maskDir = "masks"
    resultImg = "result_images"
    resultMask = "result_masks"

    os.makedirs(resultImg, exist_ok=True)
    os.makedirs(resultMask, exist_ok=True)

    for file in tqdm(os.listdir(imagesDir)):
        if not(file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".png")):
            continue
        image = cv2.imread(os.path.join(imagesDir, file))
        cv2.imwrite(os.path.join(resultImg, file.replace(".jpg", ".png").replace(".jpeg", ".png")), image)
    
    for file in tqdm(os.listdir(maskDir)):
        if not(file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".png")):
            continue
        image = cv2.imread(os.path.join(maskDir, file))
        cv2.imwrite(os.path.join(resultMask, file.replace(".jpg", ".png").replace(".jpeg", ".png")), image)
    

if __name__ == "__main__":
    main()