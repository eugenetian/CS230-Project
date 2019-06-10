# In[1]:
# Adapted From https://www.kaggle.com/sanwal092/intro-to-cnn-using-keras-to-predict-pneumonia

import sys
#get_ipython().system('{sys.executable} -m pip install matplotlib')
#get_ipython().system('{sys.executable} -m pip install pillow')
#get_ipython().system('{sys.executable} -m pip install sklearn')
import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import matplotlib
import os
# os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from PIL import Image
print(os.listdir("./splitdata"))
#print(os.listdir("./tb_data"))

# Keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
mainDIR = os.listdir('./splitdata/')
#mainDIR = os.listdir('./tb_data/')
print(mainDIR)


# In[3]:


train_folder = './splitdata/train/'
val_folder = './splitdata/val/'
test_folder = './splitdata/test/'
#test_folder = './tb_data/test/'


# In[4]:


print(os.listdir(train_folder))
train_n = train_folder+'nofinding/'
train_p = train_folder+'abnormality/'


# In[5]:


# Normal Picture
print(len(os.listdir(train_n)))
rand_index = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_index]
print('No Finding Picture Name: ', norm_pic)

norm_pic_address = train_n+norm_pic

# Pneumonia Picture
rand_index = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_index]
print('Abnormal Picture Name: ', sic_pic)

sic_pic_address = train_p+sic_pic

# Loading Images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_pic_address)

# Plotting Random Two Images
f = plt.figure(figsize = (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal Lungs')

a2 = f.add_subplot(1,2,2)
img_plot = plt.imshow(sic_load)
a2.set_title('Abnormal')


# In[6]:


image_size = 224
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D
from keras.utils import multi_gpu_model

def modified_resnet():
    resnet = ResNet50(weights='imagenet', include_top=False)
    
    result = resnet.output
    result = GlobalAveragePooling2D()(result)
    result = Dense(512, activation='relu')(result)
    
    predictions = Dense(2, activation='sigmoid')(result)
    
    transfer_model = Model(inputs=resnet.input, outputs=predictions)
    transfer_model = multi_gpu_model(transfer_model, gpus=4)
    return transfer_model

cnn = modified_resnet()                


# In[7]:


cnn.summary()
weights_path = './weights/modified_resnet.h5'
cnn = load_model(weights_path)


# In[9]:


# Fit CNN to Images

# Training image normalization settings
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range=20,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip = True)

# Test/Validation image normalization settings
test_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

train_batchsize = 10
val_batchsize = 10
test_batchsize = 10

train_generator = train_datagen.flow_from_directory(train_folder,
                                                  target_size = (image_size, image_size),
                                                  batch_size = train_batchsize,
                                                  class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(val_folder,
     target_size=(image_size, image_size),
     batch_size=val_batchsize,
     class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_folder,
                                            target_size = (image_size, image_size),
                                            batch_size = test_batchsize,
                                            class_mode = 'categorical',
                                            shuffle = False)


# In[10]:


from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

save_path = "saved-weights-{epoch:02d}-{val_acc:.2f}.hdf5"
my_opt = Adam(lr=0.001, decay=1e-5)
early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)
periodic_saving = ModelCheckpoint(save_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
cnn.compile(loss='binary_crossentropy', 
            optimizer=my_opt,
            metrics=['accuracy'])


# In[11]:


cnn_model = cnn.fit_generator(train_generator,
                        epochs = 15,
                        steps_per_epoch=train_generator.samples/train_generator.batch_size,
                        validation_steps=validation_generator.samples/validation_generator.batch_size,
                        validation_data = validation_generator,
                        callbacks=[early_stop, periodic_saving],
                        verbose=1)


# In[17]:


test_accu = cnn.evaluate_generator(test_generator, steps = test_generator.samples/test_generator.batch_size, verbose=1)


# In[18]:


print('The testing accuracy is :',test_accu[1]*100, '%')


# In[19]:


plt.plot(cnn_model.history['acc'])
plt.plot(cnn_model.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(cnn_model.history['val_loss'])
plt.plot(cnn_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()


# In[88]:


# Confusion Matrix
Y_pred = cnn.predict_generator(test_generator, steps = test_generator.samples/test_generator.batch_size, verbose=1)


# In[89]:


y_pred = np.argmax(Y_pred, axis=1)
print("This is y_pred and Y_pred")
print(y_pred)
print(Y_pred)

print ("Len of y_pred is ")
print(len(y_pred))
print ("Len of Y_pred is ")
print(len(Y_pred))


# In[16]:

real_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())


cm1 = confusion_matrix(real_classes, y_pred)
print("This is confusion_matrix")
print(cm1)
print('Classification Report')
target_names = ['NO FINDING', 'ABNORMAL']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

print("This is test_generator class indices")
print(test_generator.class_indices)


# In[15]:


# Adapted from SciKit documentation 

from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                         normalize=False,
                         title=None,
                         cmap=plt.cm.Blues):
   if not title:
       if normalize:
           title = 'Normalized confusion matrix'
       else:
           title = 'Confusion matrix, without normalization'

    Compute confusion matrix
   cm = confusion_matrix(y_true, y_pred)
    Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
   if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       print("Normalized confusion matrix")
   else:
       print('Confusion matrix, without normalization')

   print(cm)

   fig, ax = plt.subplots()
   im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
   ax.figure.colorbar(im, ax=ax)
    We want to show all ticks...
   ax.set(xticks=np.arange(cm.shape[1]),
          yticks=np.arange(cm.shape[0]),
           ... and label them with the respective list entries
          xticklabels=classes, yticklabels=classes,
          title=title,
          ylabel='True label',
          xlabel='Predicted label')

    Rotate the tick labels and set their alignment.
   plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    Loop over data dimensions and create text annotations.
   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i in range(cm.shape[0]):
       for j in range(cm.shape[1]):
           ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
   fig.tight_layout()
   return ax


np.set_printoptions(precision=2)

Plot non-normalized confusion matrix
plot_confusion_matrix(real_classes, y_pred, classes=target_names,
                     title='Confusion matrix, without normalization')

Plot normalized confusion matrix
plot_confusion_matrix(real_classes, y_pred, classes=target_names, normalize=True,
                     title='Normalized confusion matrix')

plt.show()



