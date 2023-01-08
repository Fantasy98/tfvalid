#%%
import tensorflow as tf 
from tensorflow import keras
from utils.data import LoadTF
from utils.networks import FCN_Skip_Padding
import wandb
from wandb.keras import WandbMetricsLogger,WandbEvalCallback,WandbModelCheckpoint
import os 

#%%
var=['u_vel',"v_vel","w_vel","pr0.2"]
target=['pr0.2_flux']
normalized=False
y_plus=50
#%%
train_dl = LoadTF("train",y_plus,var,target,normalized,batch_s=2)
val_dl = LoadTF("validation",y_plus,var,target,normalized,batch_s=2)
# %%
model = FCN_Skip_Padding(var,padding_layers=8)
model.summary()
model.compile("adam","mean_squared_error")
# %%
from tqdm import tqdm
Loss = []
import time
N_STEP = 1000
Time = time.time()
for i in tqdm(range(N_STEP)):
    t = time.time()
    print("On Step {}".format(i))
    train_sub = train_dl.take(1).cache()
    val_sub = val_dl.take(1).cache()
    history= model.fit(x = train_sub,
                        epochs=2,
                        validation_data=val_sub
                        # callbacks= [backup_cb, early_stopping_cb]
                        )
    loss = history.history["loss"]
    Loss.append(loss)
    print("Train loss ={}".format(loss))
    interval = time.time() - t 
    print("Time used {}".format(interval))


End_time = time.time() - Time
print("The time totally used {}".format(End_time)) 
# %%
model.save("y50_pr02flux_ut{}".format(N_STEP))

# %%
