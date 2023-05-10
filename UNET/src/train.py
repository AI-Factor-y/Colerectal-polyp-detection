import tensorflow as tf
import sys
import argparse
from keras_segmentation.models.unet import vgg_unet, unet, resnet50_unet
from keras_segmentation.models.segnet import segnet
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from predict import predict
from keras.callbacks import Callback

class TestModel(Callback):
    def __init__(self, modelName: str):
        super().__init__()
        self.modelName = modelName
    
    def on_epoch_end(self, epoch, logs=None):
        result = self.model.evaluate_segmentation(
            inp_images_dir="../tmp/training/images_prepped_test/",
            annotations_dir="../tmp/training/annotations_prepped_test/",
        )
        logs["mean_IU"] = result["mean_IU"]
        logs["frequency_weighted_IU"] = result["frequency_weighted_IU"]

        print(f"Epoch: {epoch}, testing results: ", result)

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

def main(modelName: str, epochs: int = 50) -> None:
    model = getModel(modelName)
    tensorBoard = TensorBoard(log_dir=f"logs/{modelName}")
    callbacks = [
        ModelCheckpoint(
                    filepath="checkpoints/" + modelName + ".{epoch:05d}",
                    save_weights_only=True,
                    verbose=True
                ),
        TestModel(modelName=modelName),
        tensorBoard,
    ]

    model.train(
        train_images =  "../tmp/training/images_prepped_train/",
        train_annotations = "../tmp/training/annotations_prepped_train/",
        checkpoints_path = f"../tmp/{modelName}" , epochs=epochs,
        callbacks=callbacks,
    )

    predict(model, modelName=modelName)


parser = argparse.ArgumentParser(description="Train segmentation models")
parser.add_argument("--model", metavar="model", type=str, help="Name of the model. Available models - unet, vgg_unet, resnet50_unet, segnet")
parser.add_argument("--epochs", metavar="epochs", type=int, help="No of epochs")
args = parser.parse_args()
main(modelName=args.model, epochs=args.epochs)