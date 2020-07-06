class GlobatStd2D(nn.Module):
    """
    Custom pytorch layer to compute sample-wise feature deviation
    """
    def __init__(self, input_shape, min_std_val=1e-5,**kwargs):
        super().__init__()
        self.min_std_val = min_std_val
        
        #build min_std部份
        nb_feats = input_shape[-1]
        std_shape = (1,1,1,nb_feats)
        self.min_std_unconstraint = torch.nn.Parameter(torch.full(std_shape, self.min_std_val), requires_grad = True) #先沒加上constraint
        self.min_std = self.min_std_unconstraint.data.clamp(0,float("inf"))
    def forward(self, x):
        x_std = torch.std(x, dim = (1,2), keepdim=True)
        x_std = torch.max(x_std, self.min_std_val/10. + self.min_std)
        return x_std
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1, input_shape[-1])