import h5py
import torch
import numpy as np

def _load_Featex_weights(model, f):
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
                    f"Featex.{layer_name}.{param_name}: Shape doesn't match. Got {param.data.shape} but needs {weight.shape}."
            param.data = torch.from_numpy(weight)
    return model

def _load_bnorm_weights(layer, f):
    for sub_k in f.keys():
        param_name = sub_k
        if 'mean' in param_name:
            param_name = 'running_mean'
        else:
            param_name = 'running_var'
        param = getattr(layer, param_name)
        weight = f[sub_k][:]
        assert weight.shape == param.data.shape, \
                    f"bnorm.{param_name}: Shape doesn't match. Got {param.data.shape} but needs {weight.shape}."
        param.data = torch.from_numpy(weight)
    return layer

def _load_pred_weights(layer, f):
    for sub_k in f.keys():
        param_name = sub_k[:-2]
        if param_name == 'kernel':
                param_name = 'weight'
        param = getattr(layer, param_name)
        weight = f[sub_k][:]
        if 'kernel' in sub_k:
                weight = weight.transpose(3, 2, 0, 1)
        assert weight.shape == param.data.shape, \
                    f"pred.{param_name}: Shape doesn't match. Got {param.data.shape} but needs {weight.shape}."
        param.data = torch.from_numpy(weight)
    return layer

def _load_cLSTM_weights(layer, f):
    weights = {}
    for sub_k in f.keys():
        param_name = sub_k[:-2]
        weights[param_name] = f[sub_k][:]
    # bias
    param = layer.cell_list[0].conv.bias
    weight = weights['bias']
    assert weight.shape == param.data.shape, \
                f"cLSTM.bias: Shape doesn't match. Got {param.data.shape} but needs {weight.shape}."
    param.data = torch.from_numpy(weight)
    # weight
    param = layer.cell_list[0].conv.weight
    weight = np.concatenate((weights['kernel'], weights['recurrent_kernel']), axis=2)
    weight = weight.transpose(3, 2, 0, 1)
    assert weight.shape == param.data.shape, \
                f"cLSTM.weight: Shape doesn't match. Got {param.data.shape} but needs {weight.shape}."
    param.data = torch.from_numpy(weight)
    return layer

def _load_glbStd_weights(layer, f):
    for sub_k in f.keys():
        param_name = sub_k[:-2]
        param = getattr(layer, param_name)
        weight = f[sub_k][:]
        weight = weight.transpose(0, 3, 1, 2)
        assert weight.shape == param.data.shape, \
                    f"pred.{param_name}: Shape doesn't match. Got {param.data.shape} but needs {weight.shape}."
        param.data = torch.from_numpy(weight)
    return layer

def _load_outlierTrans_weights(layer, f):
    for sub_k in f.keys():
        param_name = sub_k[:-2]
        if param_name == 'kernel':
                param_name = 'weight'
        param = getattr(layer, param_name)
        weight = f[sub_k][:]
        if 'kernel' in sub_k:
                weight = weight.transpose(3, 2, 0, 1)
        assert weight.shape == param.data.shape, \
                    f"pred.{param_name}: Shape doesn't match. Got {param.data.shape} but needs {weight.shape}."
        param.data = torch.from_numpy(weight)
    return layer

def load_weights(weight_filepath, model):
    f = h5py.File(weight_filepath, 'r')
    # Featex
    model.Featex = _load_Featex_weights(model.Featex, f['Featex'])
    layer_names = ['bnorm', 'pred', 'cLSTM', 'glbStd', 'outlierTrans']
    load_weight_ftns = [_load_bnorm_weights,
                       _load_pred_weights,
                       _load_cLSTM_weights,
                       _load_glbStd_weights,
                       _load_outlierTrans_weights]
    for k, ftn in zip(layer_names, load_weight_ftns):
        layer_name = k
        layer = getattr(model, layer_name)
        for _ in f[k].keys():
            k_ext = _
        layer = ftn(layer, f[k][k_ext])
            
    return model