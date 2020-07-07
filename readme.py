convlstm = ConvLSTM(input_dim = 64, hidden_dim = 8, kernel_size = (7, 7), num_layers = 1, batch_first = True, bias = True, return_all_layers = False)
_, last_states = convlstm(devf5d)
devf = last_states[0][0]