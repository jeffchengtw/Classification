import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.models import load_model

import cv2
import numpy as np

img_name_arr = []
predict_arr=[]
answer_arr=[]
img_arr = []
label_idx = []

PREDICT_MODE = 'test'
def get_key(dict, value):
    return [k for k, v in dict.items() if v==value]    


if PREDICT_MODE == 'train':
    with open('txt/training_labels.txt') as f:
            for line in f.readlines():
                line = line.strip('\n')
                answer_arr.append(line)
                line = line.split(' ')
                image_name = line[0]
                class_name = line[1][4:]

                img_name_arr.append(image_name)
else:
    with open('txt/testing_img_order.txt', 'r') as f:
        for line in f:
            img_name_arr.append(line.strip('\n').split(',')[0])



train_gen = ImageDataGenerator(

)
train_ds = train_gen.flow_from_directory(
    'dataset/train', 
    target_size=(480,480),
    batch_size=32,
    class_mode='categorical'
)

model = load_model('bestModelPath', compile=True)
model.summary()

label_idx = train_ds.class_indices

with open('answer/answer.txt', 'w') as f:
    for i in range(len(img_name_arr)):
        path ='src' + '/' + PREDICT_MODE + '/' + img_name_arr[i]
        #path = 'dataset/ttt' + '/' + img_name_arr[i]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (480,480))
        img = img /255.
        img = img.astype('float32')
        img = tf.reshape(img, (-1,480,480,3))
        result_class = np.argmax(model.predict(img))
        res_str = get_key(label_idx, result_class)
        answer_str = img_name_arr[i] + ' ' + res_str[0]
        f.write(answer_str + '\n')
        predict_arr.append(answer_str)
        print(answer_str)

if PREDICT_MODE == 'train':
    correct_num =0.
    error_num=0.
    for i in range(len(predict_arr)):
        if predict_arr[i] == answer_arr[i]:
            print("correct!")
            correct_num+=1
        else:
            print("wrong!")
            error_num+=1

    print(correct_num / (correct_num+error_num))
