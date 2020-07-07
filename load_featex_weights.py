import h5py
import torch

def load_weights(weight_filepath, model):
    f = h5py.File(weight_filepath, 'r')
    f = f['Featex']
    for k in f.keys():
        layer_name = k[:-2]
        layer = getattr(model, layer_name) # b1c1, b1c2, etc.
        for sub_k in f[k].keys():
            param_name = sub_k[:-2]
            if param_name == 'kernel':
                param_name = 'weight'
            param = getattr(layer, param_name) # weight or bias
            weight = f[k][sub_k][:]
            # transposing weights for conv kernel
            if 'kernel' in sub_k:
                weight = weight.transpose(3, 2, 0, 1)
            assert  weight.shape == param.data.shape, \
                    f"{layer}.{param} Shape doesn't match. Got {param.data.shape} but needs {weight.shape}."
            param.data = torch.from_numpy(weight)
    return model
