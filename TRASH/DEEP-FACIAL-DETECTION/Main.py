import os
import time
import uuid
import cv2

# In[ ]:


IMAGES_PATH = os.path.join('data', 'images')
number_images = 30

# In[ ]:


cap = cv2.VideoCapture(1)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# ### 1.3 Annotate Images with LabelMe

# In[ ]:



# # 2. Review Dataset and Build Image Loading Function

# ### 2.1 Import TF and Deps

# In[ ]:


import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt

# ### 2.2 Limit GPU Memory Growth

# In[ ]:


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# In[ ]:


tf.config.list_physical_devices('GPU')

# ### 2.3 Load Image into TF Data Pipeline

# In[ ]:


images = tf.data.Dataset.list_files('data\\images\\*.jpg')

# In[ ]:


images.as_numpy_iterator().next()


# In[ ]:


def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


# In[ ]:


images = images.map(load_image)

# In[ ]:


images.as_numpy_iterator().next()

# In[ ]:


type(images)

# ### 2.4 View Raw Images with Matplotlib

# In[ ]:


image_generator = images.batch(4).as_numpy_iterator()

# In[ ]:


plot_images = image_generator.next()

# In[ ]:


fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()

# # 3. Partition Unaugmented Data

# ### 3.1 MANUALLY SPLT DATA INTO TRAIN TEST AND VAL

# In[ ]:


90 * .7  # 63 to train

# In[ ]:


90 * .15  # 14 and 13 to test and val

# ### 3.2 Move the Matching Labels

# In[ ]:


for folder in ['train', 'test', 'val']:
    for file in os.listdir(os.path.join('data', folder, 'images')):

        filename = file.split('.')[0] + '.json'
        existing_filepath = os.path.join('data', 'labels', filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join('data', folder, 'labels', filename)
            os.replace(existing_filepath, new_filepath)

        # # 4. Apply Image Augmentation on Images and Labels using Albumentations

# ### 4.1 Setup Albumentations Transform Pipeline

# In[ ]:


import albumentations as alb

# In[ ]:


augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                        bbox_params=alb.BboxParams(format='albumentations',
                                                   label_fields=['class_labels']))

# ### 4.2 Load a Test Image and Annotation with OpenCV and JSON

# In[ ]:


img = cv2.imread(os.path.join('data', 'train', 'images', 'ffd85fc5-cc1a-11ec-bfb8-a0cec8d2d278.jpg'))

# In[ ]:


with open(os.path.join('data', 'train', 'labels', 'ffd85fc5-cc1a-11ec-bfb8-a0cec8d2d278.json'), 'r') as f:
    label = json.load(f)

# In[ ]:


label['shapes'][0]['points']

# ### 4.3 Extract Coordinates and Rescale to Match Image Resolution

# In[ ]:


coords = [0, 0, 0, 0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

# In[ ]:


coords

# In[ ]:


coords = list(np.divide(coords, [640, 480, 640, 480]))

# In[ ]:


coords

# ### 4.4 Apply Augmentations and View Results

# In[ ]:


augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])

# In[ ]:


augmented['bboxes'][0][2:]

# In[ ]:


augmented['bboxes']

# In[ ]:


cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2], [450, 450]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [450, 450]).astype(int)),
              (255, 0, 0), 2)

plt.imshow(augmented['image'])

# # 5. Build and Run Augmentation Pipeline

# ### 5.1 Run Augmentation Pipeline

# In[ ]:


for partition in ['train', 'test', 'val']:
    for image in os.listdir(os.path.join('data', partition, 'images')):
        img = cv2.imread(os.path.join('data', partition, 'images', image))

        coords = [0, 0, 0.00001, 0.00001]
        label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640, 480, 640, 480]))

        try:
            for x in range(60):
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'),
                            augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0, 0, 0, 0]
                    annotation['class'] = 0

                with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)

# ### 5.2 Load Augmented Images to Tensorflow Dataset

# In[ ]:


train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
train_images = train_images.map(lambda x: x / 255)

# In[ ]:


test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120, 120)))
test_images = test_images.map(lambda x: x / 255)

# In[ ]:


val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120, 120)))
val_images = val_images.map(lambda x: x / 255)

# In[ ]:


train_images.as_numpy_iterator().next()


# # 6. Prepare Labels

# ### 6.1 Build Label Loading Function

# In[ ]:


def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']


# ### 6.2 Load Labels to Tensorflow Dataset

# In[ ]:


train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

# In[ ]:


test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

# In[ ]:


val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

# In[ ]:


train_labels.as_numpy_iterator().next()

# # 7. Combine Label and Image Samples

# ### 7.1 Check Partition Lengths

# In[ ]:


len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels)

# ### 7.2 Create Final Datasets (Images/Labels)

# In[ ]:


train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)

# In[ ]:


test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)

# In[ ]:


val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)

# In[ ]:


train.as_numpy_iterator().next()[1]

# ### 7.3 View Images and Annotations

# In[ ]:


data_samples = train.as_numpy_iterator()

# In[ ]:


res = data_samples.next()

# In[ ]:


fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = res[0][idx]
    sample_coords = res[1][1][idx]

    cv2.rectangle(sample_image,
                  tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                  (255, 0, 0), 2)

    ax[idx].imshow(sample_image)

# # 8. Build Deep Learning using the Functional API

# ### 8.1 Import Layers and Base Network

# In[ ]:


from keras.models import Model
from keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from keras.applications import VGG16

# ### 8.2 Download VGG16

# In[ ]:


vgg = VGG16(include_top=False)

# In[ ]:


vgg.summary()


# ### 8.3 Build instance of Network

# In[ ]:


def build_model():
    input_layer = Input(shape=(120, 120, 3))

    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker


# ### 8.4 Test out Neural Network

# In[ ]:


facetracker = build_model()

# In[ ]:


facetracker.summary()

# In[ ]:


X, y = train.as_numpy_iterator().next()

# In[ ]:


X.shape

# In[ ]:


classes, coords = facetracker.predict(X)

# In[ ]:


classes, coords

# # 9. Define Losses and Optimizers

# ### 9.1 Define Optimizer and LR

# In[ ]:


batches_per_epoch = len(train)
lr_decay = (1. / 0.75 - 1) / batches_per_epoch

# In[ ]:


opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)


# ### 9.2 Create Localization Loss and Classification Loss

# In[ ]:


def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))

    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]

    h_pred = yhat[:, 3] - yhat[:, 1]
    w_pred = yhat[:, 2] - yhat[:, 0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    return delta_coord + delta_size


# In[ ]:


classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

# ### 9.3 Test out Loss Metrics

# In[ ]:


localization_loss(y[1], coords)

# In[ ]:


classloss(y[0], classes)

# In[ ]:


regressloss(y[1], coords)


# # 10. Train Neural Network

# ### 10.1 Create Custom Model Class

# In[ ]:


class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

            total_loss = batch_localizationloss + 0.5 * batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)


# In[ ]:


model = FaceTracker(facetracker)

# In[ ]:


model.compile(opt, classloss, regressloss)

# ### 10.2 Train

# In[ ]:


logdir = 'logs'

# In[ ]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# In[ ]:


hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

# ### 10.3 Plot Performance

# In[ ]:


hist.history

# In[ ]:


fig, ax = plt.subplots(ncols=3, figsize=(20, 5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()

# # 11. Make Predictions

# ### 11.1 Make Predictions on Test Set

# In[ ]:


test_data = test.as_numpy_iterator()

# In[ ]:


test_sample = test_data.next()

# In[ ]:


yhat = facetracker.predict(test_sample[0])

# In[ ]:


fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]

    if yhat[0][idx] > 0.9:
        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)

    ax[idx].imshow(sample_image)

# ### 11.2 Save the Model

# In[ ]:


from keras.models import load_model

# In[ ]:


facetracker.save('facetracker.h5')

# In[ ]:


facetracker = load_model('facetracker.h5')

# ### 11.3 Real Time Detection

# In[ ]:


cap = cv2.VideoCapture(1)
while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:500, 50:500, :]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        # Controls the main rectangle
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                      (255, 0, 0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                   [0, -30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                   [80, 0])),
                      (255, 0, 0), -1)

        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                                [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('EyeTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# In[ ]:
