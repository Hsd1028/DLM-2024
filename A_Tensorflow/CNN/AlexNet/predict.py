import os
import json

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import AlexNet_v1, AlexNet_v2


def main():
    im_height = 224
    im_width = 224

    # load image
    img_path = "../../../B_data_pre/tulips.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # resize image to 224x224
    img = img.resize((im_width, im_height))
    plt.imshow(img)

    # scaling pixel value to (0-1)
    img = np.array(img) / 255.

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = 'class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = AlexNet_v1(num_classes=5)
    weighs_path = r"E:\DATA\LM_2024\trains_result\T-AlexNet.h5"
    # weighs_path = r"E:\DATA\LM_2024\trains_result\T-AlexNet.ckpt"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(weighs_path)
    model.load_weights(weighs_path)

    # prediction
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])
    plt.title(print_res)
    for i in range(len(result)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  result[i]))
    plt.show()


if __name__ == '__main__':
    main()