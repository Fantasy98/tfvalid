#%%
import tensorflow as tf 
from tensorflow import keras
from utils.data import LoadTF
from utils.networks import FCN_Skip_Padding
import wandb
from wandb.keras import WandbMetricsLogger,WandbEvalCallback,WandbModelCheckpoint
import os 

#%%
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
#%%
train_dl = LoadTF("train",y_plus,var,target,normalized,batch_s=2)
val_dl = LoadTF("validation",y_plus,var,target,normalized,batch_s=2)
# %%
# %%
wandb.login()
# %%
import json
with open("wandb_config.json","r") as f:
    configs = json.load(f)
f.close()

from wandb.keras import WandbCallback
run = wandb.init(project="valid",entity="yuning98",config=configs)


# %%
config = wandb.config

#%%
wandb_callback = WandbCallback(monitor="loss",)
wandb_eval = WandbMetricsLogger("batch")
callbacks = [wandb_callback,wandb_eval]
# %%
model = FCN_Skip_Padding(var,padding_layers=config.padding)
model.summary()
# %%
model.compile("adam","mean_squared_error")
# %%
hist = model.fit(train_dl,validation_data=val_dl,
                  callbacks=callbacks)
# %%

# %%
