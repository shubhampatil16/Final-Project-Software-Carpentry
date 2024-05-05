import os
import numpy as np
import torch
import pydicom
import matplotlib.pyplot as plt
from tcia_utils import nbia #cancer imaging archive library
from monai.bundle import ConfigParser, download
from monai.transforms import LoadImage, LoadImaged, Orientation, Orientationd, EnsureChannelFirst, EnsureChannelFirstd, Compose
import json

datadirect = 'D:\Shubham\Johns Hopkins University\Spring 2024\Software Carpentry\Final Project\Data'

'''
Part 1: Open CT Image
First, we will download the CT data into the data directory.
This data was obtained from the cancer imaging archive.
'''

name_cart = "nbia-56581714340225512"
data_cart = nbia.getSharedCart(name_cart)
df = nbia.downloadSeries(data_cart, format="df", path = datadirect)

#combine data directory with name of the folder
CT_storedfolder = os.path.join(datadirect, '1.3.6.1.4.1.14519.5.2.1.1600.1201.312087933416130338402619297252')


'''
Two ways of opening data in python:
Option 1: Using pydicom
'''

dataset = pydicom.read_file(os.path.join(CT_storedfolder, '1-215.dcm'))

#we can obtain pixel data by accessing the pixel_array attribute
imagegen = dataset.pixel_array
imagegen.shape

#Note that the image is a 2D array. Typically the pixel values are stored in a scaled format so we should adjust them:
imagegen = dataset.RescaleSlope * imagegen + dataset.RescaleIntercept

plt.pcolormesh(imagegen, cmap='Greys_r')
plt.colorbar(label='Hounsfield Units (HU)')
plt.axis('off')
plt.show()

'''
-1000 = Air
0 = Water
>1000 = Bone
'''

'''
Option 2: Using monai
MONAI stands for "Medical Open Network for Artificial Intelligence" and is essentially an extension of PyTorch for machine learning with medical data, containing many many many important functions.
'''

#MONAI can be used to open up medical data
image_loader = LoadImage(image_only=True)
CT_monai = image_loader(CT_storedfolder)

#The CT_monai contains both the pixel data (for all slices) and the image metadata
CT_monai.meta

#Now, we plot any plane of the CT image
CT_monai_coronal_slice = CT_monai[:,256].cpu().numpy()

#View CT image
plt.figure(figsize=(3,8))
plt.pcolormesh(CT_monai_coronal_slice.T, cmap='Greys_r')
plt.colorbar(label='Hounsfield Units (HU)')
plt.axis('off')
plt.show()

#This generates an image upside-down! We have to manually reverse the axis, or we can use MONAI functionality to modify the CT.

#Firstly we add a channel dimension, since this is required for most AI applications
CT_monai.shape

channel_transform = EnsureChannelFirst()
CT_monai = channel_transform(CT_monai)
CT_monai.shape

#Now we reorient the CT image
orientation_transform = Orientation(axcodes=('LPS'))
CT_monai = orientation_transform(CT_monai)

#Now obtain the coronal slice
CT_monai_coronal_slice = CT_monai[0,:,256].cpu().numpy()

#Now plot again
plt.figure(figsize=(3,8))
plt.pcolormesh(CT_monai_coronal_slice.T, cmap='Greys_r')
plt.colorbar(label='Hounsfield Units (HU)')
plt.axis('off')
plt.show()

#Alternatively, we can combine all these transforms in one go when we open the image data
preprocessing_pipeline = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes='LPS')
    ])  

#And we can open using this preprocessing pipeline
CT_monai = preprocessing_pipeline(CT_storedfolder)
CT_monai_coronal_slice = CT_monai[0,:,256].cpu().numpy()

#And plot
plt.figure(figsize=(3,8))
plt.pcolormesh(CT_monai_coronal_slice.T, cmap='Greys_r')
plt.colorbar(label='Hounsfield Units (HU)')
plt.axis('off')
plt.show()

'''
One other option (which is typically done) is to use the "dictionary" version of all the transforms above. This is done by adding a d to the end of the transforms, such as LoadImaged.
'''
#These transforms take in a dictionary with keys-value pairs
dicdata = {'image': CT_storedfolder, 'some_other_key': 42}
preprocessing_pipeline = Compose([
    LoadImaged(keys='image', image_only=True),
    EnsureChannelFirstd(keys='image'),
    Orientationd(keys='image',axcodes='LPS')
    ])
dicdata = preprocessing_pipeline(dicdata)

'''
Part 2: Segmentation Model
First we download the segmentation model which is obtained from https://monai.io/model-zoo.html
'''
segmodel_name = "wholeBody_ct_segmentation"
download(name=segmodel_name, bundle_dir=datadirect)

#We first set the paths of where we downloaded the model parameters (model.pt) and a file called inference.json
segmodel_path = os.path.join(datadirect, 'wholeBody_ct_segmentation', 'models', 'model_lowres.pt')
segconfig_path = os.path.join(datadirect, 'wholeBody_ct_segmentation', 'configs', 'inference.json')

#From this we create a config instance which lets us read from the json file
config = ConfigParser()
config.read_config(segconfig_path)

'''
Preprocessing Pipeline
From this we can extract the preprocessing pipeline specified by the inference.json file
'''
#These are all the operations applied to the data before feeding it to the model
preprocessing = config.get_parsed_content("preprocessing")

#This preprocessing pipeline uses LoadImaged instead of LoadImage. 
#The d at the end refers to the fact that everything should be fed in as a dictionary. 
#The keys argument are the keys of the dictionary by which to apply the transform to.
dicdata = preprocessing({'image': CT_storedfolder})

'''
Model
Now we can obtain the model using the 'network' key from the json file
'''
segmodel = config.get_parsed_content("network")

#At the moment, the model is initialized with random parameters. 
#We need to configure it with the parameters given by the model.pt file.
#Since we won't be training it (only use it for evaluation), we'll use the eval() function
segmodel.load_state_dict(torch.load(segmodel_path))
segmodel.eval();

'''
Inferer
The "inferer" pipeline takes in the data and the model, and returns model output. 
It contains some extra processing steps (in this case it breaks the data into 96x96x96 chunks before feeding it into the model)
'''
inferer = config.get_parsed_content("inferer")

'''
Postprocessing
Finally, once the model has finished running, there will be postprocessing that needs to be done on the data
'''
postprocessing = config.get_parsed_content("postprocessing")
#dicdata['image'].unsqueeze(0).shape

'''
Prediction Time
We can now combine all these pipelines to obtain organ masks for our data
'''
dicdata = preprocessing({'image': CT_storedfolder}) # returns a dictionary
#Compute mask prediction, add it to dictionary
with torch.no_grad():
    #Have to add additional batch dimension to feed into model
    dicdata['pred'] = inferer(dicdata['image'].unsqueeze(0), network=segmodel)
# Remove batch dimension in image and prediction
dicdata['pred'] = dicdata['pred'][0]
dicdata['image'] = dicdata['image'][0]
# Apply postprocessing to data
dicdata = postprocessing(dicdata)
segmentationvar = torch.flip(dicdata['pred'][0], dims=[2])
segmentationvar = segmentationvar.cpu().numpy()

segslice_idx = 250
CT_monai_coronal_slice = CT_monai[0,:,segslice_idx].cpu().numpy()
segmentation_monai_coronal_slice = segmentationvar[:,segslice_idx]

#plot
plt.subplots(1,2,figsize=(6,8))
plt.subplot(121)
plt.pcolormesh(CT_monai_coronal_slice.T, cmap='Greys_r')
plt.axis('off')
plt.subplot(122)
plt.pcolormesh(segmentation_monai_coronal_slice.T, cmap='nipy_spectral')
plt.axis('off')
plt.show()