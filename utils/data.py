def feature_description(save_loc):
    """Loads the json file descriping the file format for parsing the tfRecords. For now only array_serial has been implemented!
       Furthermore the last entry is allways the target.

    Args:
        save_loc (string): The file path to the folder where the data is saved

    Returns:
        dict: dict used to read tfRecords
    """
    import os
    import tensorflow as tf
    import json
    feature_format={}

    with open(os.path.join(save_loc,"format.json"),'r') as openfile:
        format_json=json.load(openfile)
    


    for key in list(format_json.keys()):
        if format_json[key] =="array_serial":
            feature_format[key]= tf.io.FixedLenFeature([], tf.string, default_value="")
        else:
            print("other features than array has not yet been implemented!")
    return feature_format

def read_tfrecords(serial_data,format,target):
    """Reads the tfRecords and converts them to a tuple where first entry is the features and the second is the targets

    Args:
        serial_data (TFrecordDataset): The output of the function tf.data.TFRecordDataset
        format (dict): dict used to parse the TFrecord example format

    Returns:
        tuple: tuple of (features,labels)
    """
    import tensorflow as tf
    features=tf.io.parse_single_example(serial_data, format)
    dict_for_dataset={}
    #Loops through the features and saves them into a dict
    for key, value in features.items():
        if value.dtype == tf.string:
            dict_for_dataset[key]=tf.io.parse_tensor(value,tf.float64)
        else:
            print("only arrays have been implemented")
       
    if len(target)==1:
        target_array=dict_for_dataset[target[0]]    
    else:
        target_array_list=[]
        for i in range(len(target)):
            target_array_i=dict_for_dataset[target[i]]
            target_array_list.append(target_array_i)
        target_array=tf.stack(target_array_list,axis=2)
    #Removes the target from the dict
    for i in target:
        dict_for_dataset.pop(i)
    return (dict_for_dataset,target_array)

def LoadTF(data_type,y_plus,var,target,normalized,root_path="/home/yuning/thesis/data",repeat=10,shuffle_size=100,batch_s=10):
    """Load TFrecord to scratch and loads the data from there
        
    Args:
        data_type(str): train,validation,test
        y_plus (int): the y_plus value to load data from
        var (list): list of features
        target (list): list of targets. only 1 for now
        repeat (int, optional): number of repeats of the dataset for each epoch. Defaults to 10.
        shuffle_size (int, optional): the size of the shuffle buffer. Defaults to 100.
        batch_s (int, optional): the number of snapshots in each buffer. Defaults to 10.

    Returns:
        [type]: [description]
    """
      
    import tensorflow as tf
    import os
    import shutil
    import xarray as xr
    save_loc=slice_loc(y_plus,var,target,normalized,False,root_path)
    print(f"Load data from {save_loc}")
    if not os.path.exists(save_loc):
        raise Exception("data does not exist. Make som new")

    features_dict=feature_description(save_loc)

    name = data_type
    data_loc=os.path.join(save_loc,name)
    dataset = tf.data.TFRecordDataset([data_loc],compression_type='GZIP',buffer_size=100,num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset=dataset.map(lambda x: read_tfrecords(x,features_dict,target),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset=dataset.shuffle(buffer_size=shuffle_size)
    dataset=dataset.repeat(repeat)
    dataset=dataset.batch(batch_size=batch_s)
    dataset=dataset.prefetch(3) 
    
    return dataset

def slice_loc(y_plus,var,target,normalized,test=False,root_path = "/home/yuning/thesis/data"):
    """where to save the slices

    Args:
        y_plus (int): y_plus value of slice
        var (list): list of variables
        target (list): list of targets
        normalized (bool): if the data is normalized or not
    
    Returns:
        str: string of file save location
    """
    import os

    var_sort=sorted(var)
    var_string="_".join(var_sort)
    target_sort=sorted(target)
    target_string="|".join(target_sort)
    if test==False:
        if normalized==True:
            slice_loc=os.path.join(root_path,'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string+"-normalized")
        else:
            slice_loc=os.path.join(root_path,'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string)
    else:
        if normalized==True:
            slice_loc=os.path.join(root_path +"_test",'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string+"-normalized")
        else:
            slice_loc=os.path.join(root_path +"_test",'y_plus_'+str(y_plus)+"-VARS-"+var_string+"-TARGETS-"+target_string)

    return slice_loc