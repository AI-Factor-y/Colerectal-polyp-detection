import os
from keras_segmentation.models.unet import vgg_unet, unet, resnet50_unet
from keras_segmentation.models.segnet import segnet
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

model = unet(n_classes=2 ,  input_height=416, input_width=608)

tensorBoard = TensorBoard(log_dir=f"logs/unet")
callbacks = [
    ModelCheckpoint(
                filepath="checkpoints/" + model.name + ".{epoch:05d}",
                save_weights_only=True,
                verbose=True
            ),
    tensorBoard
]

model.train(
    train_images =  "../tmp/training/images_prepped_train/",
    train_annotations = "../tmp/training/annotations_prepped_train/",
    checkpoints_path = "../tmp/unet" , epochs=2,
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

for filename in os.listdir(testImgDir):
    if not filename.endswith(".png"):
        continue

    try:
        out = model.predict_segmentation(
            inp=f"../tmp/training/images_prepped_test/{filename}",
            out_fname=f"../tmp/out_{filename}"
        )
    except Exception as e:
        print(e)

    # evaluating the model
    print(model.evaluate_segmentation( inp_images_dir="../tmp/training/images_prepped_test/"  , annotations_dir="../tmp/training/annotations_prepped_test/" ) )
