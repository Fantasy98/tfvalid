#%%
# Test to use model predict a single snap shot 
# The model is far more to be trained
import os
from tensorflow import keras
from keras import layers
import tensorflow as tf
import wandb
import numpy as np
import matplotlib.pyplot as plt
from DataHandling.features.slices import read_tfrecords,feature_description,slice_loc
from wandb.keras import WandbCallback
from DataHandling.features import slices
from DataHandling import utility
# from utils.prediction import predict
from utils.plots import Plot_2D_snapshots
from utils.metrics import RMS_error, Glob_error
os.environ['WANDB_DISABLE_CODE']='True'
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[-1], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass
#%%
model_path = "/home/yuning/thesis/tfvalid/y50_pr02flux_ut500"
print(model_path)

overwrite = False
var=['u_vel',"v_vel","w_vel","pr0.2"]
target=['pr0.2_flux']
normalized=False
y_plus=50
# %%
from keras import models
model = models.load_model(model_path)
print(model.summary())

#%%
from utils.data import LoadTF
test_dl = LoadTF("test",y_plus,var,target,normalized,batch_s=1)
#%%

test_sample= iter(test_dl).next()
x = test_sample[0]
y = test_sample[1]
y = y.numpy().squeeze()

pred = model.predict(x)
pred = pred.squeeze()

# %%
from utils.metrics import RMS_error,Glob_error,Fluct_error

error = Glob_error(pred,y)
print(error)

error = RMS_error(pred,y)
print(error)



error = Fluct_error(pred,y)
print(error)


#%%
from utils.plots import Plot_2D_snapshots
Plot_2D_snapshots(y,"target")
Plot_2D_snapshots(pred,"pred")
# %%
