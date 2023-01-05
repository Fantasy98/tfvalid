def FCN_Skip_Padding(input_features,activation='elu',padding_layers=1):
    import tensorflow as tf
    import keras
    
    from keras.layers import InputSpec 
    from tensorflow.python.keras.utils import conv_utils

    class PeriodicPadding2D(keras.layers.Layer):
        def __init__(self, padding=1, **kwargs):
            super(PeriodicPadding2D, self).__init__(**kwargs)
            self.padding = conv_utils.normalize_tuple(padding, 1, 'padding')
            self.input_spec = InputSpec(ndim=3)

        def wrap_pad(self, input, size):
            M1 = tf.concat([input[:,:, -size:], input, input[:,:, 0:size]], 2)
            M1 = tf.concat([M1[:,-size:, :], M1, M1[:,0:size, :]], 1)
            return M1

        def compute_output_shape(self, input_shape):
            shape = list(input_shape)
            assert len(shape) == 3  
            if shape[1] is not None:
                length = shape[1] + 2*self.padding[0]
            else:
                length = None
            return tuple([shape[0], length, length])

        def call(self, inputs): 
            return self.wrap_pad(inputs, self.padding[0])

        def get_config(self):
            config = {'padding': self.padding}
            base_config = super(PeriodicPadding2D, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    input_list=[]
    reshape_list=[]
    weights=[128,256,256]
    
    for features in input_features:
        input=keras.layers.Input(shape=(256,256),name=features)
        input_list.append(input)
        pad=PeriodicPadding2D(padding=padding_layers)(input)
        reshape=keras.layers.Reshape((256+2*padding_layers,256+2*padding_layers,1))(pad)
        reshape_list.append(reshape)
    conc=keras.layers.Concatenate()(reshape_list)
    

    batch1=keras.layers.BatchNormalization(-1)(conc)
    cnn1=keras.layers.Conv2D(64,5,activation=activation)(batch1)
    
    batch2=keras.layers.BatchNormalization(-1)(cnn1)
    cnn2=keras.layers.Conv2D(weights[0],3,activation=activation)(batch2)

    batch3=keras.layers.BatchNormalization(-1)(cnn2)
    cnn3=keras.layers.Conv2D(weights[1],3,activation=activation)(batch3)

    batch4=keras.layers.BatchNormalization(-1)(cnn3)
    cnn4=keras.layers.Conv2D(weights[2],3,activation=activation)(batch4)
    
    batch5=keras.layers.BatchNormalization(-1)(cnn4)
    
    conc1=keras.layers.Concatenate()([cnn4,batch5])
    cnn5=keras.layers.Conv2DTranspose(weights[0],3,activation=activation)(conc1)
    batch6=keras.layers.BatchNormalization(-1)(cnn5)
    
    conc2=keras.layers.Concatenate()([cnn3,batch6])
    cnn6=keras.layers.Conv2DTranspose(weights[1],3,activation=activation)(conc2)
    batch7=keras.layers.BatchNormalization(-1)(cnn6)
    
    conc3=keras.layers.Concatenate()([cnn2,batch7])
    cnn7=keras.layers.Conv2DTranspose(weights[2],3,activation=activation)(conc3)
    batch8=keras.layers.BatchNormalization(-1)(cnn7)
    
    conc4=keras.layers.Concatenate()([cnn1,batch8])
    cnn8=tf.keras.layers.Conv2DTranspose(64,5,activation=activation)(conc4)
    batch9=keras.layers.BatchNormalization(-1)(cnn8)

    conc5=keras.layers.Concatenate()([conc,batch9])
    output=tf.keras.layers.Conv2DTranspose(1,1)(conc5)
    output=keras.layers.Cropping2D(cropping=padding_layers)(output)

    model = keras.Model(inputs=input_list, outputs=output)
    return model