from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=2 ,  input_height=416, input_width=608  )

model.train(
    train_images =  "../tmp/training/images_prepped_train/",
    train_annotations = "../tmp/training/annotations_prepped_train/",
    checkpoints_path = "../tmp/vgg_unet_1" , epochs=10
)

for i in range(5):
    try:
        out = model.predict_segmentation(
            inp=f"../tmp/training/images_prepped_test/{i}.png",
            out_fname=f"../tmp/out_{i}.png"
        )
    except Exception as e:
        print(e)

    # import matplotlib.pyplot as plt
    # plt.imshow(out)

    # evaluating the model 
    print(model.evaluate_segmentation( inp_images_dir="../tmp/training/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )
