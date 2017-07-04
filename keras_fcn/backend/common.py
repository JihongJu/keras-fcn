import h5py
import keras.backend as K


def load_weights(model, weights_path):
    """Load weights from Caffe models."""
    print("Loading weights...")
    if h5py is None:
        raise ImportError('`load_weights` requires h5py.')
    f = h5py.File(weights_path, mode='r')

    # New file format.
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in model.layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            # Set values.
            for i in range(len(weight_values)):
                weight_value_tuples.append((symbolic_weights[i],
                                            weight_values[i]))
    K.batch_set_value(weight_value_tuples)

    return layer_names
