
import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# 간단한 Sequential 모델을 정의합니다
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model


# 기본 모델 객체를 만듭니다
model = create_model()

# 모델을 평가합니다
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("훈련되지 않은 모델의 정확도: {:5.2f}%".format(100*acc))

# 가중치 로드
model.load_weights(checkpoint_path)

# 모델 재평가
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))


# 체크포인트 콜백 매개변수
# 이 콜백 함수는 몇 가지 매개변수를 제공합니다. 체크포인트 이름을 고유하게 만들거나 체크포인트 주기를 조정할 수 있습니다.
# 새로운 모델을 훈련하고 다섯 번의 에포크마다 고유한 이름으로 체크포인트를 저장해 보겠습니다:

# 파일 이름에 에포크 번호를 포함시킵니다(`str.format` 포맷)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 다섯 번째 에포크마다 가중치를 저장하기 위한 콜백을 만듭니다
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)

# 새로운 모델 객체를 만듭니다
model = create_model()

# `checkpoint_path` 포맷을 사용하는 가중치를 저장합니다
model.save_weights(checkpoint_path.format(epoch=0))

# 새로운 콜백을 사용하여 모델을 훈련합니다
model.fit(train_images,
          train_labels,
          epochs=50,
          callbacks=[cp_callback],
          validation_data=(test_images,test_labels),
          verbose=0)


latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)


# 새로운 모델 객체를 만듭니다
model = create_model()
# 이전에 저장한 가중치를 로드합니다
model.load_weights(latest)
# 모델을 재평가합니다
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))


# 새로운 모델 객체를 만들고 훈련합니다
model = create_model()
model.fit(train_images, train_labels, epochs=5)

loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print('복원된 모델의 정확도: {:5.2f}%'.format(100*acc))
# SavedModel로 전체 모델을 저장합니다
model.save('saved_model/my_model')


new_model2 = tf.keras.models.load_model('saved_model/my_model')

# 모델 구조를 확인합니다
new_model2.summary()


# 복원된 모델을 평가합니다
loss, acc = new_model2.evaluate(test_images,  test_labels, verbose=2)
print('복원된 모델의 정확도: {:5.2f}%'.format(100*acc))


# Create a new model instance
model = create_model()
# Train the model
model.fit(train_images, train_labels, epochs=5)
# Save the entire model to a HDF5 file
model.save('my_model.h5')