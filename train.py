import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout, 
                                     BatchNormalization, Activation, 
                                     Multiply, Reshape, Conv2D, Layer, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                       ReduceLROnPlateau, LearningRateScheduler)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import top_k_accuracy_score
import random
import shutil
import math
import json
from PIL import Image

# 设置随机种子保证可复现性
np.random.seed(42)
tf.random.set_seed(42)

# 数据集路径
dataset_dir = '/kaggle/input/nba-players/NBA'  # 替换为您的数据集路径

# 参数设置
BATCH_SIZE = 24  # 由于使用更大模型和分辨率，可能需要根据显存调整
IMG_SIZE = (300, 300)  # 增加输入分辨率
EPOCHS = 100
NUM_CLASSES = len(os.listdir(dataset_dir))  # 球星数量
TRAIN_VAL_SPLIT = 0.2  # 20%作为验证集
LABEL_SMOOTHING = 0.1  # 标签平滑正则化

# 创建训练和验证目录
base_dir = '/kaggle/working/nba_players'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# 复制文件并划分训练集/验证集
for player in os.listdir(dataset_dir):
    player_dir = os.path.join(dataset_dir, player)
    images = os.listdir(player_dir)
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_VAL_SPLIT)
    
    # 创建目标目录
    os.makedirs(os.path.join(train_dir, player), exist_ok=True)
    os.makedirs(os.path.join(val_dir, player), exist_ok=True)
    
    # 复制训练文件
    for img in images[split_idx:]:
        src = os.path.join(player_dir, img)
        dst = os.path.join(train_dir, player, img)
        shutil.copy(src, dst)
    
    # 复制验证文件
    for img in images[:split_idx]:
        src = os.path.join(player_dir, img)
        dst = os.path.join(val_dir, player, img)
        shutil.copy(src, dst)

# 改进的数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # 增加旋转范围
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  # 添加垂直翻转
    brightness_range=[0.7, 1.3],  # 亮度调整
    channel_shift_range=50.0,  # 通道偏移
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 创建数据生成器
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # 验证集不需要打乱
)

# 通道注意力层
class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        
    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = Dense(
            channel // self.ratio,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=False
        )
        self.shared_layer_two = Dense(
            channel,
            kernel_initializer='he_normal',
            use_bias=False
        )
        super(ChannelAttention, self).build(input_shape)
        
    def call(self, inputs):
        # 平均池化路径
        avg_pool = GlobalAveragePooling2D()(inputs)
        avg_pool = Reshape((1, 1, inputs.shape[-1]))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)
        
        # 最大池化路径
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)
        
        # 合并路径
        cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)
        
        # 应用注意力
        return Multiply()([inputs, cbam_feature])
    
    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'ratio': self.ratio})
        return config

# 加载更大的预训练模型
base_model = EfficientNetV2L(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_preprocessing=False  # 手动添加归一化
)

# 冻结预训练模型的层
base_model.trainable = False

# 创建模型
inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs)

# 添加通道注意力
x = ChannelAttention(ratio=8)(x)

# 更深的分类头
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(2048, kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(1024, kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.4)(x)

predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

# 自定义学习率调度（余弦衰减 + 预热）
def lr_schedule(epoch, lr):
    warmup_epochs = 5
    total_epochs = EPOCHS
    
    if epoch < warmup_epochs:
        # 线性预热
        return lr * (epoch + 1) / warmup_epochs
    else:
        # 余弦衰减
        decay_epoch = epoch - warmup_epochs
        decay_total = total_epochs - warmup_epochs
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_epoch / decay_total))
        return lr * cosine_decay

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # 初始学习率
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
)

# 回调函数
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,  # 增加耐心值
    verbose=1,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    '/kaggle/working/best_model_phase1.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

# 第一阶段训练（冻结基础模型）
print("\n第一阶段训练（冻结基础模型）...")
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
    validation_data=val_generator,
    validation_steps=max(1, val_generator.samples // BATCH_SIZE),
    epochs=20,  # 第一阶段训练轮数
    callbacks=[early_stop, model_checkpoint, reduce_lr, lr_scheduler],
    verbose=1
)

# 第二阶段：解冻模型顶层进行微调
print("\n第二阶段训练（微调顶层）...")
# 解冻最后30%的层
for layer in base_model.layers[-int(len(base_model.layers)*0.3):]:
    layer.trainable = True

# 重新编译模型，使用更小的学习率
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
)

# 更新模型检查点路径
model_checkpoint_phase2 = ModelCheckpoint(
    '/kaggle/working/best_model_phase2.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# 第二阶段训练
history_fine = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
    validation_data=val_generator,
    validation_steps=max(1, val_generator.samples // BATCH_SIZE),
    epochs=EPOCHS,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=[early_stop, model_checkpoint_phase2, reduce_lr],
    verbose=1
)

# 合并训练历史
for key in history_fine.history:
    history.history[key] += history_fine.history[key]

# 保存训练历史
history_df = pd.DataFrame(history.history)
history_df.to_excel('/kaggle/working/training_history.xlsx', index=False)

# 绘制训练曲线
plt.figure(figsize=(16, 8))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['top3_accuracy'], label='Training Top-3 Acc')
plt.plot(history.history['val_top3_accuracy'], label='Validation Top-3 Acc')
plt.title('Top-3 Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Top-3 Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png')
plt.show()

# 加载最佳模型
model.load_weights('/kaggle/working/best_model_phase2.h5')

# 计算加权Top-3准确率
def weighted_top_k_accuracy(y_true, y_pred, k=3, weights=[1.0, 0.8, 0.6]):
    top_k = np.argsort(y_pred, axis=1)[:, ::-1][:, :k]
    scores = []
    
    for i in range(len(y_true)):
        true_label = np.argmax(y_true[i])
        for j in range(k):
            if top_k[i, j] == true_label:
                scores.append(weights[j])
                break
        else:
            scores.append(0.0)
    
    return np.mean(scores)

# 预测验证集
val_generator.reset()
y_true = val_generator.classes
y_pred = model.predict(val_generator, steps=val_generator.samples // BATCH_SIZE + 1)

# 确保预测结果与真实标签长度一致
y_pred = y_pred[:len(y_true)]

# 计算加权准确率
weighted_acc = weighted_top_k_accuracy(
    to_categorical(y_true, num_classes=NUM_CLASSES),
    y_pred
)
print(f"\nWeighted Top-3 Accuracy: {weighted_acc:.4f}")

# 计算标准Top-1和Top-3准确率
top1_acc = top_k_accuracy_score(y_true, y_pred, k=1)
top3_acc = top_k_accuracy_score(y_true, y_pred, k=3)
print(f"Top-1 Accuracy: {top1_acc:.4f}")
print(f"Top-3 Accuracy: {top3_acc:.4f}")

# 随机抽取10个验证样本可视化 - 修改：展示原始尺寸图片
sample_indices = random.sample(range(len(y_true)), 10)
class_names = list(train_generator.class_indices.keys())

# 创建更大的画布
plt.figure(figsize=(20, 25))

for i, idx in enumerate(sample_indices):
    # 获取图像路径
    img_path = val_generator.filepaths[idx]
    
    # 使用PIL加载原始尺寸图片
    img = Image.open(img_path)
    
    # 真实标签
    true_label = class_names[y_true[idx]]
    
    # 预测结果
    top3_indices = np.argsort(y_pred[idx])[::-1][:3]
    top3_labels = [class_names[j] for j in top3_indices]
    top3_probs = y_pred[idx][top3_indices]
    
    # 可视化 - 使用原始尺寸图片
    plt.subplot(5, 2, i+1)
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {top3_labels[0]} ({top3_probs[0]:.2f})", fontsize=12)
    plt.axis('off')
    
    # 打印Top-3预测
    print(f"Sample {i+1}:")
    print(f"  Image: {os.path.basename(img_path)}")
    print(f"  True: {true_label}")
    print(f"  Top-3 Predictions:")
    for j in range(3):
        print(f"    {j+1}. {top3_labels[j]}: {top3_probs[j]:.4f}")

plt.tight_layout()
plt.savefig('/kaggle/working/sample_predictions_original_size.png', dpi=150)
plt.show()

# ====================== 保存模型和词表映射 ====================== #
# 保存完整模型
model.save('/kaggle/working/full_model.keras')

# 保存类别映射（词表）
class_mapping = {
    'class_to_index': train_generator.class_indices,
    'index_to_class': {v: k for k, v in train_generator.class_indices.items()}
}

# 保存为JSON文件
with open('/kaggle/working/class_mapping.json', 'w') as f:
    json.dump(class_mapping, f, indent=4)

print("\n模型和词表映射已保存:")
print(f"- 完整模型: /kaggle/working/full_model.keras")
print(f"- 类别映射: /kaggle/working/class_mapping.json")