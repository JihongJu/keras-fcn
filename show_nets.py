from keras.utils import plot_model
from fcn import FCN


def vis_fcn_vgg16():
    input_shape = (500, 500, 3)
    fcn_vgg16 = FCN(input_shape=input_shape)
    plot_model(fcn_vgg16, to_file='fcn_vgg16.png')


if __name__ == "__main__":
    vis_fcn_vgg16()
