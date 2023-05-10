import os
import sys
import argparse
from keras_segmentation.models.unet import vgg_unet, unet, resnet50_unet
from keras_segmentation.models.segnet import segnet
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


def predict(model, modelName: str = ""):
    testImgDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "tmp",
        "training",
        "images_prepped_test",
    )

    if modelName == "" or modelName == None:
        modelName = model.name

    saveDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "tmp", modelName
    )

    os.makedirs(saveDir, exist_ok=True)

    for filename in os.listdir(testImgDir):
        if not filename.endswith(".png"):
            continue

        try:
            out = model.predict_segmentation(
                inp=f"../tmp/training/images_prepped_test/{filename}",
                out_fname=f"../tmp/{modelName}/out_{filename}",
            )
        except Exception as e:
            print(e)

    # evaluating the model
    print(
        model.evaluate_segmentation(
            inp_images_dir="../tmp/training/images_prepped_test/",
            annotations_dir="../tmp/training/annotations_prepped_test/",
        )
    )

def getModel(modelName: str):
    if modelName == "vgg_unet":
        return vgg_unet(n_classes=2 ,  input_height=256, input_width=256)
    if modelName == "unet":
        return unet(n_classes=2 ,  input_height=256, input_width=256)
    if modelName == "resnet50_unet":
        return resnet50_unet(n_classes=2 ,  input_height=256, input_width=256)
    if modelName == "segnet":
        return segnet(n_classes=2 ,  input_height=256, input_width=256)
    
    print("Error: Unsupported model.")
    sys.exit(1)

def __main(modelName: str, checkpoint: str) -> None:
    model = getModel(modelName)
    model.load_weights(f"checkpoints/{checkpoint}")

    predict(model, modelName)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation models")
    parser.add_argument("--model", metavar="model", type=str, help="Name of the model. Available models - unet, vgg_unet, resnet50_unet, segnet")
    parser.add_argument("--checkpoint", metavar="checkpoint", type=str, help="path to the checkpoint")
    args = parser.parse_args()
    __main(modelName=args.model, checkpoint=args.checkpoint)
