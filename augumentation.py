#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary functions 
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img 
   
# Initialising the ImageDataGenerator class. 
# We will pass in the augmentation parameters in the constructor. 
datagen = ImageDataGenerator( 
        rotation_range = 40, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True, 
        brightness_range = (0.5, 1.5)) 
src=r'F:\bike_car\bike'+'\\'
ext='.jpg'
for j in range(1,619):    
 # Loading a sample image  
    img = load_img(src+str(j)+ext)  
    # Converting the input sample image to an array 
    x = img_to_array(img) 
    # Reshaping the input image 
    x = x.reshape((1, ) + x.shape)  
   
    # Generating and saving 5 augmented samples  
    # using the above defined parameters.  
    i = 0
    for batch in datagen.flow(x, batch_size = 1, 
                          save_to_dir ='F:\\bike_car\\aug_bike',  
                          save_prefix ='image', save_format ='jpg'): 
        i += 1
        if i > 5: 
            break


# In[3]:


# Importing necessary functions 
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img 
   
# Initialising the ImageDataGenerator class. 
# We will pass in the augmentation parameters in the constructor. 
datagen = ImageDataGenerator( 
        rotation_range = 40, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True, 
        brightness_range = (0.5, 1.5)) 
src=r'F:\bike_car\car'+'\\'
ext='.jpg'
for j in range(1,621):    
 # Loading a sample image  
    img = load_img(src+str(j)+ext)  
    # Converting the input sample image to an array 
    x = img_to_array(img) 
    # Reshaping the input image 
    x = x.reshape((1, ) + x.shape)  
   
    # Generating and saving 5 augmented samples  
    # using the above defined parameters.  
    i = 0
    for batch in datagen.flow(x, batch_size = 1, 
                          save_to_dir ='F:\\bike_car\\aug_car',  
                          save_prefix ='image', save_format ='jpg'): 
        i += 1
        if i > 5: 
            break



