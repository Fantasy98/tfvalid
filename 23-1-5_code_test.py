
#%%
from utils.data import LoadTF
from utils.networks import FCN_Skip_Padding
data_type ="train"
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
#%%
dataset = LoadTF(data_type,y_plus,var,target,normalized,batch_s=2)
# %%
sample = next(iter(dataset))
# x is a dict and y is numpy
x,y = sample
# %%
# print(x.numpy().shape())
print(y.get_shape())
print(y.dtype)
print(y.device)
#%%
# %%
from keras import optimizers,losses
model = FCN_Skip_Padding(var,"elu",8)
model.summary()
model.compile(optimizer =optimizers.Adam(),loss="mean_squared_error")
# %%
loss_hist  =[]
for i in range(500):
    batch = dataset.take(1)
    
    hist= model.fit(batch,epochs=2,verbose=0)
    loss = hist.history["loss"]
    print(loss)
    loss_hist.append(loss)
