import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import Image

warnings.filterwarnings('ignore')
SEED = 34

mnist = keras.datasets.mnist
((train_images, train_labels) , (test_images, test_labels)) = mnist.load_data()

# 2. 데이터의 shape을 출력
print(f"train_images{train_images.shape}")
print(f"train_labels{train_labels.shape}")
print(f"test_images{test_images.shape}")
print(f"test_labels{test_labels.shape}")

# BHW Batch / Height / Width
# (60000, 28, 28)
# (60000,)
# (10000, 28, 28)
# (10000,)

# 3. (28, 28) 형태의 이미지를 plt를 이용해서 출력
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(True)
print(train_labels[0])

# 4. train_images에서 0이 아닌 값들을 출력해보세요

# 1차원으로 변경함
train_images[0].reshape(-1)
sorted(list(filter(lambda x:x != 0, train_images[0].reshape(-1))))[:10]

# 5. train_images의 dtype을 출력해보세요
print(f"train_images ?: {train_images.dtype}")
print(f"train_labels ?: {train_labels.dtype}")
print(f"test_images ?: {test_images.dtype}")
print(f"test_labels ?: {test_labels.dtype}")

"""
    Step 2. 전처리
    문제 6. train/test 이미지 데이터의 범위 확인
    문제 7. train/test 이미지 데이터의 최소/최대값을 출력
    문제 8. 정수형을 실수형으로 변경 후 dtype으로 비교
    문제 9. 데이터 0-1 노말라이즈 수행
    문제 10. 0-1 노말라이즈 후 데이터의 값이 변경되었는지 문제 6, 7의 방법을 이용하여 확인하세요.
"""

# 문제 6. train/test 이미지 데이터의 범위 확인
# test_images의 shape과 dtype, 0이 아닌 숫자를 출력하는 코드를 작성하세요.

print(f"train_images ?: {train_images.shape}")
print(f"train_labels ?: {train_labels.shape}")
print(f"test_images ?: {test_images.shape}")
print(f"test_labels ?: {test_labels.shape}")

print(list(filter(lambda x:x != 0, train_images[0].reshape(-1)))[:10])
print(list(filter(lambda x:x != 0, train_labels.reshape(-1)))[:10])
print(list(filter(lambda x:x != 0, test_images[0].reshape(-1)))[:10])
print(list(filter(lambda x:x != 0, test_labels.reshape(-1)))[:10])

# 문제 7. train/test 이미지 데이터의 최소/최대값을 출력
# train/test 전체 데이터에서 각 images, labels의 min, max를 출력하는 코드를 작성하세요.
print(f"train_images ?: {max(train_images.reshape(-1)), min(train_images.reshape(-1)) }")
print(f"train_images ?: {max(test_images.reshape(-1)), min(test_images.reshape(-1)) }")

# 문제 8. 정수형을 실수형으로 변경 후 dtype으로 비교
# train/test 데이터의 타입을 dtype으로 확인해보고 실수형으로 전환하는 코드를 작성하세요.
print(train_images.astype(np.float64))
print(test_images.astype(np.float64))

train_images = train_images.astype(np.float64)
test_images = test_images.astype(np.float64)

# 문제 9. 데이터 0-1 노말라이즈 수행
# images의 값이 0-1사이의 값을 같도록 코드를 작성해보세요.
train_images = (train_images / 255)
test_images = (test_images / 255)

# 문제 10. 0-1 노말라이즈 후 데이터의 값이 변경되었는지 문제 6, 7의 방법을 이용하여 확인하세요.
# 노말라이즈 후 min/max, shape, 0이 아닌 값, dtype 등을 확인하는 코드를 작성해보세요.
print(list(filter(lambda x: x != 0, train_images[0].reshape(-1)))[:10])
print(list(filter(lambda x: x != 0, train_labels.reshape(-1)))[:10])
print(list(filter(lambda x: x != 0, test_images[0].reshape(-1)))[:10])
print(list(filter(lambda x: x != 0, test_labels.reshape(-1)))[:10])

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
print(train_images.dtype, train_labels.dtype, test_images.dtype, test_labels.dtype)

"""
    Step 3. 시각화 방법
    문제 11. train_image의 이미지를 5장 획득하여 (5, 28, 28)의 shape을 출력하세요.
    문제 12. 획득한 5장의 의미지를 (28, 28 * 5)의 shape으로 변경해줍니다.
    문제 13. np.hstack은 tensorflow에서 제공하는 방법이 아니므로 transpose 함수를 이용하여 (28, 28 * 5)로 shape을 변경해보세요.
    문제 14. (28, 140)이 된 tensor를 plt로 출력해보세요.
    문제 15. (28, 140)이 된 tensor를 plt로 흑백으로 출력해보세요. 또한, 해당되는 labels의 값도 print로 출력하세요.
"""

# 문제 11. train_image의 이미지를 5장 획득하여 (5, 28, 28)의 shape을 출력하세요.
# (60000, 28, 28)인 train_images에서 (5, 28, 28)을 획득하는 코드를 작성하세요.
print(train_images[:5].shape)

# 문제 12. 획득한 5장의 의미지를 (28, 28 * 5)의 shape으로 변경해줍니다
# np.hstack은 height 방향의 배열을 풀어서 width 방향으로 연결해줍니다.
# 해당 기능을 쓰면 (height, image_height, image_width)의 shape을 (image_height, image_width * height)으로 바꿔 줄 수 있습니다.
# 코드를 작성해보세요.
# 의도한대로 shape를 맞추는게 중요하다.
img_data = np.hstack(train_images[:5])
print(img_data.shape)
plt.imshow(img_data)

# 문제 13. np.hstack은 tensorflow에서 제공하는 방법이 아니므로 transpose 함수를 이용하여 (28, 28 * 5)로 shape을 변경해보세요.
# transpose는 tensor의 axis를 섞는 기능을 합니다.
# 해당 기능을 이용해서 (28, 140)을 작성하는 코드를 작성해보세요.
img_trans = train_images[:5].transpose(( 1, 0, 2)).reshape(28, -1)

# 문제 14. (28, 140)이 된 tensor를 plt로 출력해보세요.
# (28, 140)의 이미지를 plt로 출력해보세요.
plt.imshow(img_trans)
plt.show()

# 문제 15. (28, 140)이 된 tensor를 plt로 흑백으로 출력해보세요. 또한, 해당되는 labels의 값도 print로 출력하세요.
# images와 labels 5개를 출력하는 코드를 작성하세요.
plt.imshow(img_trans, cmap="gray")
plt.colorbar()
plt.show()
print(train_labels[:5])


"""
    Step 4. Data augmentation - Noise 추가 방법
        문제 16. np.random.random 함수를 이용하여 0-1 사이의 랜덤값을 3회 print로 출력해보시오.
        문제 17. np.random.random 함수와 shape 파라매터를 (28, 28)의 랜덤 노이즈를 생성해보세요
        문제 18. 생성된 random (28, 28) 노이즈를 plt를 통하여 확인해보세요.
        문제 19. 가우시안 노이즈 함수를 사용하여 평균 0, 표준편차 0.1, 사이즈 1로 랜덤 값을 3번 출력하세요. (np.random.normal)
        문제 20. 가우시안 노이즈 함수의 옵션을 문제 19과 동일하지만 평균이 3.0인 경우, 표준 편차가 0.01인 경우로 각각 3회씩 출력해보세요.
        문제 21. 가우시안 노이즈 함수를 문제 19의 옵션으로 size를 (28, 28)로 생성 후 plt로 출력해보세
        문제 22. train_images의 5번째 이미지와 가우시안 노이즈 (28, 28)를 생성 한 뒤 각각 tensor를 더한 뒤 noisy_image 변수에 할당 해보세요.
        문제 23. noisy_image를 plt를 통해서 출력해보세요.
        문제 24. 노이지 이미지를 생성했지만, max가 1이 넘습니다. max값을 1로 조절해보세요.
        문제 25. 위의 방법을 전부 활용하여 train_images와 test_images 데이터에 랜덤 노이즈를 추가한 train_noisy_images와 test_noisy_images를 생성해보세요.
        문제 26. labels에 onehot 인코딩을 적용하여 (배치 사이즈, 클래스 개수)의 shape으로 변경해보세요.
"""
# 문제 16. np.random.random 함수를 이용하여 0-1 사이의 랜덤값을 3회 print로 출력해보시오.
# np.random.ranodm을 이용해서 0-1 사의의 랜덤값을 3회 출력하는 코드를 작성하시오.
print(np.random.random())
print(np.random.random())
print(np.random.random())

# 문제 17. np.random.random 함수와 shape 파라매터를 (28, 28)의 랜덤 노이즈를 생성해보세요
# np.random.randpm 함수와 shape 파라매터를 이용하여 (28, 28)의 랜덤 노이즈를 생성하는 코드를 작성하시오.
np.random.random((28,28)).shape

# 문제 18. 생성된 random (28, 28) 노이즈를 plt를 통하여 확인해보세요.
# plt를 통하여 random (28, 28) 노이즈를 2회 출력하는 코드를 작성하시고, 이미지가 다른지 확인하세요.
plt.imshow(np.random.random((28,28)), cmap="gray")
plt.colorbar()
plt.show()

# 문제 19. 가우시안 노이즈 함수를 사용하여 평균 0, 표준편차 0.1, 사이즈 1로 랜덤 값을 3번 출력하세요. (np.random.normal)
# 가우시안 노이즈 함수를 이용하여 mu 0.1 std 0.1 size 1인 랜덤값을 3번 출력하는 코드를 작성하세요.
print(np.random.normal(0.0, 0.1, 1))
print(np.random.normal(0.0, 0.1, 1))
print(np.random.normal(0.0, 0.1, 1))

# 문제 20. 가우시안 노이즈 함수의 옵션을 문제 19과 동일하지만 평균이 3.0인 경우, 표준 편차가 0.01인 경우로 각각 3회씩 출력해보세요.
# 가우시안 노이즈 함수 mu = 3.0, std = 0.1, size = 1을 3 회 출력, mu = 0.0, std = 0.01, size = 1을 3회 출력하는 코드를 작성해보세요.
print(np.random.normal(3.0, 0.01, 1))
print(np.random.normal(3.0, 0.01, 1))
print(np.random.normal(3.0, 0.01, 1))

# 문제 21. 가우시안 노이즈 함수를 문제 19의 옵션으로 size를 (28, 28)로 생성 후 plt로 출력해보세요
# 가우시안 노이즈 함수를 문제 17의 옵션으로 size를 (28, 28)로 생성 후 plt로 출력하는 코드를 작성해보세요. 단, std = 0.01
plt.imshow(np.random.normal(0.0, 0.01, (28, 28)), cmap="gray")
plt.colorbar()
plt.show()

# 문제 22. train_images의 5번째 이미지와 가우시안 노이즈 (28, 28)를 생성 한 뒤 각각 tensor를 더한 뒤 noisy_image 변수에 할당 해보세요.
# train_images[5]와 가우시안 노이즈 (28, 28)을 더한 뒤 noisy_image 변수에 할당하는 코드를 작성하세요. mu = 0.5, std = 0.1
noisy_image = train_images[5] + np.random.normal(0.5, 0.1, (28, 28))

# 문제 23. noisy_image를 plt를 통해서 출력해보세요.
# noisy_image를 plt로 출력하는 코드를 작성해보세요.
plt.imshow(noisy_image, cmap="gray")
plt.colorbar()
plt.show()

# 문제 24. 노이지 이미지를 생성했지만, max가 1이 넘습니다. max값을 1로 조절해보세요.
# max가 1을 초과하지 않게 noisy_image를 수정한 뒤 plt로 출력해보세요.
noisy_image[noisy_image > 1.0] = 1.0
plt.imshow(noisy_image, cmap="gray")
plt.colorbar()
plt.show()

# 문제 25. 위의 방법을 전부 활용하여 train_images와 test_images 데이터에 랜덤 노이즈를 추가한 train_noisy_images와 test_noisy_images를 생성해보세요.
# train_noisy_image와 test_nosiy_images를 생성하는 코드를 작성하세요.
train_noisy_images = train_images + np.random.normal(0.5, 0.1, train_images.shape)
train_noisy_images[train_noisy_images > 1.0] = 1.0

test_noisy_images = test_images + np.random.normal(0.5, 0.1, test_images.shape)
test_noisy_images[test_noisy_images > 1.0] = 1.0

plt.imshow(train_noisy_images[:5].transpose(( 1, 0, 2)).reshape(28, -1), cmap="gray")
plt.show()


# 문제 26. labels에 onehot 인코딩을 적용하여 (배치 사이즈, 클래스 개수)의 shape으로 변경해보세요.
# train/test labels에 onehot encoding을 적용하여 shape을 (배치사이즈,) 에서 (배치사이즈, 클래스 개수)로 변경하는 코드를 작성하세요.
from tensorflow.keras.utils import to_categorical

print(train_labels.shape, test_labels.shape)
train_labels = to_categorical( train_labels, 10)
test_labels = to_categorical( test_labels, 10)
print(train_labels.shape, test_labels.shape)


"""
    Step 5. 모델링
        문제 27. 해당 학습셋을 처리하는 이미지 classification 모델을 작성하세요.
        문제 28. 모델 요약 정보를 출력해보세요.
        문제 29. 27에서 만든 모델에 로스와 옵티마이저, 메트릭을 설정하세요
        문제 30. 만든 모델에 train_noisy_images를 학습시켜 보세요.
        문제 31. 학습 진행 사항을 plt으로 출력하세요.
"""

# 문제 27. 해당 학습셋을 처리하는 이미지 classification 모델을 작성하세요.
# input (28, 28), 64 unit의 simple RNN, 10 unit의 fully-conntect를 가진 모델을 작성하세요.
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# 시계열로 처리함 위에서부터 차례대로 스캔하는 개념임
inputs = Input(shape=(28, 28))
x1 = SimpleRNN(64, activation="tanh")(inputs)
x2 = Dense(10, activation="softmax")(x1)
model = Model(inputs, x2)

# 문제 28. 모델 요약 정보를 출력해보세요.
# 모델 요약 정보를 출력해보세요.
print(model.summary())

# 문제 29. 27에서 만든 모델에 로스와 옵티마이저, 메트릭을 설정하세요.
# 만든 모델에 loss는 categorical_crossentropy, optimizer는 adam 매트릭은 accuracy으로 설정하는 코드를 작성하세요.
# loss란 ? 모델이 예측하는 값과 label값이 차이가 없으면 없을수록 조금씩 줄이는 방향으로 학습을 진행한다.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# 문제 30. 만든 모델에 train_noisy_images를 학습시켜 보세요.
# train_noisy_images를 학습시키고 5 epochs을 돌리고 그 진행 사항을 hist에 저장하는 코드를 작성하세요.
# epochs(에포크)는 학습의 전체 반복 주기
# verbose는 학습의 진행 상황을 보여줄 것인지 지정을 하는데 verbose를 1로 세팅하면 학습이 되는 모습을 볼 수 있다.
hist = model.fit(train_noisy_images, train_labels, validation_data=(test_noisy_images, test_labels), epochs=5, verbose=2)

# 문제 31. 학습 진행 사항을 plt으로 출력하세요.
# hist의 accuracy plt의 plot을 이용하여 출력하는 코드를 작성하세요.
plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend(loc='upper left')
plt.show()

"""
    Step 6. 결과 확인
        문제 32. 완성된 모델에서 test_noisy_image를 1장 넣고 결과를 res 변수에 저정하세요.
        문제 33. test_noisy_images[0], test_images[0]를 width 방향으로 결합하여 plt로 출력하세요
        문제 34. res와 test_labels[0]의 결과를 plt.bar로 확인하세요.
"""

# 문제 32. 완성된 모델에서 test_noisy_image를 1장 넣고 결과를 res 변수에 저정하세요.
# 모델에 test_noisy_images 중 1장을 넣고 결과를 받는 코드를 작성하세요.
res = model.predict( test_noisy_images[3:4] )
print(res.shape)

# 문제 33. test_noisy_images[0], test_images[0]를 width 방향으로 결합하여 plt로 출력하세요
# test_noisy_images[0], test_images[0]를 width 방향으로 결합하여 (28, 28 * 2) 의 이미지를 만들어 plt로 출력하는 코드를 작성하세요.
plt.imshow(np.concatenate([test_noisy_images[3], test_images[3]], axis=1) , cmap="gray")
plt.show()

# 문제 34. res와 test_labels[0]의 결과를 plt.bar로 확인하세요.
# res와 test_labels[0]의 결과를 plt.bar로 확인하세요.
plt.bar(range(10), res[0], color='red')
plt.bar(np.array(range(10)) + 0.35, test_labels[3])
plt.show()

"""
    Step 7. 모델 저장 및 로드, 다운  
        문제 35. 모델을 저장하세요.
        문제 36. 모델 파일을 새로운 모델에 로드하세요.
        문제 37. 로드한 모델을 test 데이터로 평가해보세요.
        문제 38. 모델을 내 컴퓨터에 저장해보세요
"""


# 문제 35. 모델을 저장하세요.
# 모델을 저장하는 코드를 작성하세요.
model.save("./lecture001.h5")

# 문제 36. 모델 파일을 새로운 모델에 로드하세요.
# 모델을 로드하는 코드를 작성하세요.
new_model = tf.keras.models.load_model('./lecture001.h5')
res = new_model.predict( test_noisy_images[3:4] )
res.shape
plt.bar(range(10), res[0], color='red')
plt.bar(np.array(range(10)) + 0.35, test_labels[3])
plt.show()

# 문제 37. 로드한 모델을 test 데이터로 평가해보세요.
# 로드한 모델을 test 데이터로 평가해보세요.
loss, acc = new_model.evaluate(test_noisy_images, test_labels, verbose=2)
print(loss, acc)
loss, acc = model.evaluate(test_noisy_images, test_labels, verbose=2)
print(loss, acc)


# 문제 37. 로드한 모델을 test 데이터로 평가해보세요.
# 문제 38. 모델을 내 컴퓨터에 저장해보세요

# 1. 탑 컨퍼런스
# 인공지능, 컴퓨터비전 NIPS, ICML, ICLR, CVPR, ICCV
# 정보보안 분야 : S&P, CCS, USENIX Security, NDSS
# 컴퓨터 아키텍처 분야 : MICRO, HPCA
# 자연어 처리 : ACL, NACCL

# 2. 아카이브

# 3. 유명 연구실 혹은 기업
