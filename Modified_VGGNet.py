# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.layers import Resizing, BatchNormalization, Activation
from keras.optimizers import SGD
from tensorflow.keras.activations import swish

# csv 파일 읽기
DataFrame = pd.read_csv('/content/drive/MyDrive/Data/HAM10000_metadata.csv')

# 'dx' 값에 대한 카운트를 내림차순으로 정렬하고 그래프로 시각화
order = DataFrame['dx'].value_counts().index
plt.figure(figsize=(4, 4))
ax = sns.countplot(x='dx', data=DataFrame, palette='hsv', order=order)

# 각 막대 위에 해당 클래스의 수를 표시
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center',
                va = 'center',
                xytext = (0, 10),  # 간격 조정
                textcoords = 'offset points')
plt.show()

# CSV 파일 읽기
data = pd.read_csv('/content/drive/MyDrive/Data/hmnist_28_28_RGB.csv')

# ADASYN 객체 생성 및 데이터 증강
adasyn = ADASYN()
X = data.drop('label', axis=1)
y = data['label']
X_res, y_res = adasyn.fit_resample(X, y)

# 증강된 데이터를 CSV 파일로 저장
resampled_data = pd.concat([X_res, y_res], axis=1)
resampled_data.to_csv('/content/drive/MyDrive/Preprocessing/adasyn.csv', index=False)

# 증강된 데이터의 라벨 분포를 막대 그래프로 시각화
value_counts = pd.Series(y_res).value_counts().sort_index()
plt.figure(figsize=(5, 5))
bars = plt.bar(value_counts.index, value_counts.values)# 막대 그래프 위에 텍스트 추가
plt.title('After adasyn', pad=20)
plt.xlabel('Class', labelpad=15)
plt.ylabel('Count', labelpad=15)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, yval, ha='center', va='bottom')  # 텍스트 위치 설정
plt.show()

# 데이터 로드 및 전처리
df = pd.read_csv('/content/drive/MyDrive/Preprocessing/adasyn.csv')
X = df.drop(['label'],axis=1)/255
y = pd.get_dummies(df['label'])  # 원핫 인코딩 적용

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
X_train = X_train.values.reshape(-1,28,28,3)
X_test= X_test.values.reshape(-1,28,28,3)
y_train = y_train.values
y_test = y_test.values

# ADASYN을 통해 이미지가 잘 생성되었는지 확인하는 단계
plt.figure(figsize=(22, 32)) #출력 이미지 크기를 설정
for i in range(15): #이미지 수는 15
    plt.subplot(7, 5, i + 1)
    k = np.random.randint(0, len(X)-1)
    plt.imshow(np.array(X.values[k]).reshape(28,28,3)) #x.values[k]는 x에서 k번째 행을 선택하며, 이를 reshape(28,28,3)를 통해 28x28 크기의 3 채널 이미지로 변형함
    img_label = y.iloc[k].idxmax()  # 가장 큰 값을 가진 열의 이름을 가져옴 (원핫 인코딩된 라벨 데이터에서 실제 라벨을 가져오기 위한 것)
    plt.title(img_label) #이미지 위에 라벨 표시
    plt.axis("off")
plt.show()

# F1 Score, Recall, Precision 지표 정의
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def F1_Score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# 모델 정의
def model_fn():
    model = Sequential()
    model.add(Resizing(56, 56, interpolation="bilinear"))

    model.add(Conv2D(64, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))

    model.add(Conv2D(128, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))

    model.add(Conv2D(256, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Conv2D(256, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))

    model.add(Conv2D(512, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Conv2D(512, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))

    model.add(Conv2D(512, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Conv2D(512, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))

    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='he_uniform', activation=swish))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=swish))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))

    return model

# 5-Fold 교차검증 설정
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

# 평가 지표를 저장할 리스트 초기화
acc_per_fold = []
loss_per_fold = []
f1_score_per_fold = []
recall_per_fold = []
val_acc_per_fold = []
precision_per_fold = []

# 가장 높은 정확도를 가진 모델과 그 정확도를 저장하기 위한 변수 초기화
best_model = None
best_accuracy = 0

# K-Fold 교차검증 실행
for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
    print(f'------------------------------------------------------------------------')
    print(f'Training for fold {fold+1} ...')  # 현재 fold 번호 출력
    kf_X_train, kf_X_test = X_train[train_index], X_train[test_index]
    kf_y_train, kf_y_test = y_train[train_index], y_train[test_index]

    model = model_fn()  # 모델 생성
    opt=SGD(learning_rate=0.001,momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', F1_Score, recall_m, precision_m])
    history = model.fit(kf_X_train, kf_y_train, batch_size=256, epochs=10, validation_data=(kf_X_test, kf_y_test),verbose=2)

    # 각 에포크에서의 검증 Accuracy를 기록
    val_acc_per_fold.append(history.history['val_accuracy'])

    # 성능 지표 저장
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'scores for fold {fold+1}: {scores}')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    f1_score_per_fold.append(scores[2])
    recall_per_fold.append(scores[3])
    precision_per_fold.append(scores[4])

    # 현재 폴드의 정확도가 이전 폴드의 정확도보다 높은 경우, 모델과 정확도를 업데이트
    if scores[1] * 100 > best_accuracy:
        best_accuracy = scores[1] * 100
        best_model = model

# 가장 높은 정확도를 가진 모델 저장
best_model_path = "/content/drive/MyDrive/best_model.h5"
best_model.save(best_model_path)

# 평균 성능 지표 출력
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold):.4f} (+- {np.std(acc_per_fold):.4f})')
print(f'> Loss: {np.mean(loss_per_fold):.4f}')
print(f'> F1 Score: {np.mean(f1_score_per_fold):.4f}')
print(f'> Recall: {np.mean(recall_per_fold):.4f}')
print(f'> Precision: {np.mean(precision_per_fold):.4f}')
print('------------------------------------------------------------------------')

# 원본데이터로 평가
X_2 = data.drop(['label'],axis=1)/255
y_2 = pd.get_dummies(data['label'])  # 원핫 인코딩 적용
X_2 = X_2.values.reshape(-1,28,28,3)

# 데이터 분할
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, test_size=0.20, random_state=42)
X_2_test= X_2_test.reshape(-1,28,28,3)
y_2_test = y_2_test.values

# 테스트 데이터에 대한 평가
print("Evaluate on test data")
results = best_model.evaluate(X_2_test, y_2_test, batch_size=256)

# 각 fold의 검증 정확도 그래프 그리기
plt.figure(figsize=(5, 5))
for i in range(n_splits):
    plt.plot(range(1, len(val_acc_per_fold[i]) + 1), val_acc_per_fold[i], label=f'Fold {i+1} Validation Accuracy')
plt.title('Validation Accuracy per Fold')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()

# 손실 함수 그래프 그리기
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_title('Loss over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(5, 5))

# 정확도 그래프 그리기
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# F1 점수 그래프 그리기
axes[1].plot(history.history['F1_Score'], label='Training F1 Score')
axes[1].plot(history.history['val_F1_Score'], label='Validation F1 Score')
axes[1].set_title('F1 Score over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1 Score')
axes[1].legend()

plt.tight_layout()
plt.show()


