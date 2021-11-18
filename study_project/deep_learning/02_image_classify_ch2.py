import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import Image

warnings.filterwarnings('ignore')

SEED = 34
"""
    Step 1. 도입전 기본 처리
        문제 1. tfds를 이용하여 데이터셋 사용하기
        문제 2. celeba의 정보중 이용할 데이터만 추출

"""
# 문제 1. tfds를 이용하여 데이터셋 사용하기
# tfds를 이용하여 데이터셋 사용하기
import tensorflow_datasets as tfds
tfds.list_builders()

# 문제 2. celeba의 정보중 이용할 데이터만 추출
# celeba의 정보중 이용할 데이터만 추출
celeb_a = tfds.load('celeb_a')

# 문제 3. 데이터량 축소


# 데이터를 train에서 사람 이미지 한장과 label 정보 한개를 불러오는 코드를 작성해주세요. (x, y로 변수 대입)
# 문제 5. celeba_small 데이터 살펴보기
celeba_small = np.load('DATA_SET/celeba_small.npz')
x = celeba_small['train_images'][3]
y = celeba_small['train_labels'][3]

# 문제 6. x와 y의 shape을 출력해보세요.
print(x.shape, y.shape)

# 문제 7. x를 각각 plt를 통하여 출력하세요.
# x를 plt를 이용하여 출력하는 코드를 작성해보세요
plt.imshow(x)
plt.colorbar()
plt.show()
print(y)

# 문제 8. celeba_small.npz 데이터에서 학습, 테스트 데이터를 로드하세요.
# celeba_small.npz에서 train_images, test_images, train_labels, test_labels를 np array로 로드하세요.

celeba_small = np.load('DATA_SET/celeba_small.npz')
train_images = celeba_small['train_images']
train_labels = celeba_small['train_labels']

test_images = celebga_small['test_images']
test_labels = celeba_small['test_labels']

# 문제 9. train_images에서 0이 아닌 값들을 출력해보세요.
print(train_images[train_images != 0][:10])

# 문제 10. train_images의 dtype을 출력해보세요.
print(train_images.dtype)