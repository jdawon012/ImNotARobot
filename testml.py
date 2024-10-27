import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# 1. 데이터셋 준비하기
train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, # 검증 데이터 분할 추가
                                   rotation_range=20,                   # 이미지 회전 최대 20도
                                   width_shift_range=0.2,               # 수평이동 20%
                                   height_shift_range=0.2,              # 수직이동 20%
                                   shear_range=0.2,                     # 왜곡 20%
                                   zoom_range=0.2,                      # 확대축소 20%
                                   horizontal_flip=True)                # 수평 반전 추가  

test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    'image_data/train',
    target_size=(150, 150),
    batch_size=64,
    class_mode='categorical',
    subset='training')  # 훈련 데이터

validation_generator = train_datagen.flow_from_directory(
    'image_data/train',
    target_size=(150, 150),
    batch_size=64,
    class_mode='categorical',
    subset='validation')  # 검증 데이터

test_generator = test_datagen.flow_from_directory(
    'image_data/test',
    target_size=(150, 150),
    batch_size=64,
    class_mode='categorical')

# 2) 모델 구성하기
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='softmax'))



# 3) 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy',
              optimizer='adam',  # SGD, adam
              metrics=['accuracy'])

# 4) 모델 학습시키기
hist = model.fit(train_generator, epochs=50,
                           validation_data=validation_generator)

# 모델 저장하기
#model.save('my_model_final.h5')


# 6) 저장된 모델 불러오기
#model = load_model('my_model_2.h5')      # 80%     epochs = 70
#model = load_model('my_model.h5')        # 72%     epochs = 30
#model = load_model('my_model_final.h5')  # 76%     epochs = 50


# 6) 저장된 모델 불러오기
model = load_model('my_model_2.h5')

# 5) 모델 학습과정 살펴보기
plt.rcParams['figure.figsize'] = (10, 6)
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train_loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val_loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train_accuracy')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val_accuracy')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='lower left')
acc_ax.legend(loc='upper left')
plt.show()


# 7) 모델 평가하기
score = model.evaluate(test_generator)
print('손실:', score[0])
print('정확도: %.2f%%' % (score[1] * 100))

# 8) 모델 사용하기 (테스트셋 예측)

# 시각화
plt_row = 5
plt_col = 5

# 맞춘 이미지와 못맞춘 이미지를 저장할 리스트
correct_images = []
correct_labels = []
correct_predictions = []

incorrect_images = []
incorrect_labels = []
incorrect_predictions = []

# 테스트 제너레이터로부터 이미지와 레이블을 가져와 예측
for i in range(len(test_generator)):
    x_test, y_test = next(test_generator)
    y_hat = model.predict(x_test)
    
    for j in range(len(y_test)):
        if np.argmax(y_test[j]) == np.argmax(y_hat[j]):
            if len(correct_images) < 25:
                correct_images.append(x_test[j])            # 맞춘 이미지 저장
                correct_labels.append(y_test[j])            # 맞춘 이미지의 실제 라벨 저장
                correct_predictions.append(y_hat[j])        # 맞춘 이미지의 예측 라벨 저장
        else:
            if len(incorrect_images) < 25:
                incorrect_images.append(x_test[j])          # 틀린 이미지 저장
                incorrect_labels.append(y_test[j])          # 틀린 이미지의 실제 라벨 저장
                incorrect_predictions.append(y_hat[j])      # 틀린 이미지의 예측 라벨 저장
        
        if len(correct_images) >= 25 and len(incorrect_images) >= 25:
            break
    if len(correct_images) >= 25 and len(incorrect_images) >= 25:
        break

# 맞춘 이미지 시각화
plt.rcParams['figure.figsize'] = (10, 10)
fig, ax_arr = plt.subplots(plt_row, plt_col)
name = ['bus', 'car', 'truck', 'train', 'ship', 'air', 'bic', 'mot']

for i in range(plt_row * plt_col):
    sub_plt = ax_arr[i // plt_col, i % plt_col]
    sub_plt.imshow(correct_images[i])
    sub_plt.set_title(
        'R: {} / P: {}'.format(name[np.argmax(correct_labels[i])], name[np.argmax(correct_predictions[i])]))
    sub_plt.axis('off')
plt.suptitle('Correct Predictions')
plt.show()

# 못맞춘 이미지 시각화
fig, ax_arr = plt.subplots(plt_row, plt_col)
for i in range(plt_row * plt_col):
    sub_plt = ax_arr[i // plt_col, i % plt_col]
    sub_plt.imshow(incorrect_images[i])
    sub_plt.set_title(
        'R: {} / P: {}'.format(name[np.argmax(incorrect_labels[i])], name[np.argmax(incorrect_predictions[i])]))
    sub_plt.axis('off')
plt.suptitle('Incorrect Predictions')
plt.show()
