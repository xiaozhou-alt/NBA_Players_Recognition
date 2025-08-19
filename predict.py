import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt

# 重新定义自定义层（必须与训练代码中的定义完全一致）
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        
    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = tf.keras.layers.Dense(
            channel // self.ratio,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=False
        )
        self.shared_layer_two = tf.keras.layers.Dense(
            channel,
            kernel_initializer='he_normal',
            use_bias=False
        )
        super(ChannelAttention, self).build(input_shape)
        
    def call(self, inputs):
        # 平均池化路径
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        avg_pool = tf.keras.layers.Reshape((1, 1, inputs.shape[-1]))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)
        
        # 最大池化路径
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)
        
        # 合并路径
        cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
        cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)
        
        # 应用注意力
        return tf.keras.layers.Multiply()([inputs, cbam_feature])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'ratio': self.ratio})
        return config

# 配置路径
MODEL_PATH = r'H:\project\NBA\output\model\best_model_phase2.h5'  # 替换为您的模型路径
CLASS_MAPPING_PATH = r'H:\project\NBA\output\class_mapping.json'  # 替换为您的类别映射路径
TEST_IMAGE_DIR = r'H:\project\NBA\test'  # 替换为您的测试图像目录

# 加载模型
print("加载模型中...")
try:
    model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'ChannelAttention': ChannelAttention},
    compile=False,
    )
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    if "Unknown layer" in str(e):
        print("⚠️ 请确保自定义层 'ChannelAttention' 已正确定义")
    exit()

# 加载类别映射
print("\n加载类别映射中...")
try:
    with open(CLASS_MAPPING_PATH, 'r') as f:
        class_mapping = json.load(f)
    print("✅ 类别映射加载成功")
    
    # 打印前5个类别
    print("\n类别映射示例:")
    for i, (key, value) in enumerate(class_mapping['index_to_class'].items()):
        if i >= 5: break
        print(f"索引 {key} -> {value}")
except Exception as e:
    print(f"❌ 类别映射加载失败: {e}")
    exit()

# 图像预处理函数
def preprocess_image(image_path, target_size=(300, 300)):
    """预处理图像用于模型预测"""
    try:
        img = Image.open(image_path)
        # 保留原始图像用于显示
        original_img = img.copy()
        # 调整大小为模型输入尺寸
        img = img.resize(target_size)
        img_array = np.array(img)
        
        # 处理图像通道
        if len(img_array.shape) == 2:  # 灰度图
            img_array = np.stack((img_array,) * 3, axis=-1)
            print(f"⚠️ 图像为灰度图，已转换为RGB: {os.path.basename(image_path)}")
        elif img_array.shape[2] == 4:  # RGBA转RGB
            img_array = img_array[..., :3]
            print(f"⚠️ 图像为RGBA格式，已转换为RGB: {os.path.basename(image_path)}")
        
        img_array = img_array.astype('float32') / 255.0
        return np.expand_dims(img_array, axis=0), original_img
    except Exception as e:
        print(f"❌ 图像预处理失败: {e}")
        return None, None

# 预测函数
def predict_player(image_path, top_k=3):
    """预测图像中的球员并显示结果"""
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return None, None
    
    print(f"\n处理图像: {os.path.basename(image_path)}")
    processed_img, original_img = preprocess_image(image_path)
    if processed_img is None:
        return None, None
    
    try:
        # 进行预测
        predictions = model.predict(processed_img)[0]
        
        # 获取top-k预测结果
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_indices = [int(idx) for idx in top_indices]  # 确保索引是整数
        
        # 获取球员名称
        top_players = [
            class_mapping['index_to_class'].get(str(idx), f"未知球员_{idx}") 
            for idx in top_indices
        ]
        top_probs = predictions[top_indices]
        
        # 显示结果（使用原始图像）
        plt.figure(figsize=(10, 8))
        plt.imshow(original_img)
        plt.axis('off')
        
        # 创建标题文本
        title_text = f"预测结果: {top_players[0]} ({top_probs[0]:.2f})"
        plt.title(title_text, fontsize=14)
        
        # 添加预测概率信息
        info_text = "\n".join([f"{i+1}. {player}: {prob:.4f}" 
                              for i, (player, prob) in enumerate(zip(top_players, top_probs))])
        plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=12, 
                    bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # 为底部文本留出空间
        plt.show()
        
        # 打印结果
        print("\n预测结果:")
        for i, (player, prob) in enumerate(zip(top_players, top_probs)):
            print(f"{i+1}. {player}: {prob:.4f}")
        
        return top_players, top_probs
    except Exception as e:
        print(f"❌ 预测过程中发生错误: {e}")
        return None, None

# 主程序
if __name__ == "__main__":
    print("\nNBA球员识别系统 - 本地预测")
    print("=" * 50)
    
    # 检查测试图像目录
    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"❌ 测试图像目录不存在: {TEST_IMAGE_DIR}")
        print("请创建目录并添加测试图像")
        exit()
    
    # 获取测试图像
    image_files = [f for f in os.listdir(TEST_IMAGE_DIR) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("❌ 测试图像目录中没有找到支持的图像文件")
        print("支持的格式: .png, .jpg, .jpeg")
        exit()
    
    print(f"找到 {len(image_files)} 张测试图像")
    
    # 预测所有图像
    for img_file in image_files:
        img_path = os.path.join(TEST_IMAGE_DIR, img_file)
        predict_player(img_path)
    
    print("\n预测完成!")