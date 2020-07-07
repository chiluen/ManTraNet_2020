from sym_padding import *

class Featex_vgg16(nn.Module):
	def __init__(self, type=1):
	        super(Featex_vgg16, self).__init__()
	        self.type = type
	        self.relu = nn.ReLU(inplace=True)
	        base = 32
	        # block 1
	        nb_filters1 = base
	        outchannel = 32 if self.type in [0,1] else 16
	        self.CombinedConv = CombinedConv2D(in_channels = 3, out_channels = outchannel)
	        kernelsize = (3,3)
	        self.b1_sympadd = Conv2DSymPadding(in_channels = outchannel, out_channels = nb_filters1, kernel_size = kernelsize)
	        # block 2
	        nb_filters2 = 2 * base
	        self.b2_sympadd1 = Conv2DSymPadding(in_channels = nb_filters1, out_channels = nb_filters2, kernel_size = kernelsize)
			self.b2_sympadd2 = Conv2DSymPadding(in_channels = nb_filters2, out_channels = nb_filters2, kernel_size = kernelsize)
			# block 3
			nb_filters3 = 4 * base
	        self.b2_sympadd1 = Conv2DSymPadding(in_channels = nb_filters1, out_channels = nb_filters2, kernel_size = kernelsize)
			self.b2_sympadd2 = Conv2DSymPadding(in_channels = nb_filters2, out_channels = nb_filters2, kernel_size = kernelsize)


def create_featex_vgg16_base( type=1 ) :
    base = 32
    img_input = Input(shape=(None,None,3), name='image_in')
    # block 1
    bname = 'b1' # 32
    nb_filters = base
    x = CombinedConv2D( 32 if type in [0,1] else 16, activation='relu', bias=False, padding='same', name=bname+'c1')( img_input )
    x = Conv2DSymPadding( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    # block 2
    bname = 'b2'
    nb_filters = 2 * base # 64
    x = Conv2DSymPadding( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c1')( x )
    x = Conv2DSymPadding( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    # block 3
    bname = 'b3'
    nb_filters = 4 * base # 96
    x = Conv2DSymPadding( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c1')( x )
    x = Conv2DSymPadding( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    x = Conv2DSymPadding( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c3')( x )
    # block 4
    bname = 'b4'
    nb_filters = 8 * base # 128
    x = Conv2DSymPadding( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c1')( x )
    x = Conv2DSymPadding( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    x = Conv2DSymPadding( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c3')( x )
    # block 5/bottle-neck 
    bname = 'b5'
    x = Conv2DSymPadding( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c1')( x )
    x = Conv2DSymPadding( nb_filters, (3,3), activation='relu', padding='same',  name=bname+'c2')( x )
    activation=None if type >=1 else 'tanh'
    print ("INFO: use activation in the last CONV={}".format( activation ))
    sf = Conv2DSymPadding( nb_filters, (3,3),
                           activation=activation,
                          name='transform',
                          padding='same' )(x)
    sf = Lambda( lambda t : K.l2_normalize( t, axis=-1), name='L2')(sf)
    return Model( inputs= img_input, outputs=sf, name='Featex')