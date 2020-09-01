#!/usr/bin/env python
# coding: utf-8

# In[268]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[269]:


tf.__version__


# In[270]:


training=r'C:\Users\DELL\Desktop\Section+40+-+Convolutional+Neural+Networks+(CNN)\Section 40 - Convolutional Neural Networks (CNN)\dataset\training_set'
test=r'C:\Users\DELL\Desktop\Section+40+-+Convolutional+Neural+Networks+(CNN)\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set'
                                                                                             


# In[271]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory(training,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[272]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(test,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[273]:


cnn = tf.keras.models.Sequential()


# In[274]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# In[275]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[276]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[277]:


cnn.add(tf.keras.layers.Flatten())


# In[278]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[279]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[280]:


cnn.add(tf.keras.layers.Dropout(0.5))


# In[281]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[282]:


opt = tf.keras.optimizers.Adam(learning_rate=0.0002)


# In[283]:


cnn.compile(optimizer =opt , loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[284]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 50)


# In[214]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'C:\Users\DELL\Desktop\DG\dog19.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'


# In[215]:


print(prediction)


# In[ ]:




