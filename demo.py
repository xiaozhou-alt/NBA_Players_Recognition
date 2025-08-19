import sys
import os
import json
import random
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QFileDialog, QStackedWidget,
                            QGroupBox, QRadioButton, QButtonGroup, QMessageBox, QFrame)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QMovie, QColor, QPainter, QBrush, QLinearGradient
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve

# 重新定义自定义层
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

class AnimatedButton(QPushButton):
    """带有悬停动画的按钮"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Arial", 14, QFont.Bold))
        self.setMinimumHeight(50)
        self.setCursor(Qt.PointingHandCursor)
        
        # 初始样式
        self.normal_style = """
            background-color: #1e3c72;
            color: white;
            border: 2px solid #2a5298;
            border-radius: 25px;
            padding: 10px 20px;
        """
        
        self.hover_style = """
            background-color: #2a5298;
            color: white;
            border: 2px solid #3a6bc4;
            border-radius: 25px;
            padding: 10px 20px;
        """
        
        self.setStyleSheet(self.normal_style)
        
    def enterEvent(self, event):
        # 鼠标进入时动画
        self.animate_size(50, 55)
        self.setStyleSheet(self.hover_style)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        # 鼠标离开时动画
        self.animate_size(55, 50)
        self.setStyleSheet(self.normal_style)
        super().leaveEvent(event)
        
    def animate_size(self, start, end):
        # 创建尺寸动画
        self.animation = QPropertyAnimation(self, b"minimumHeight")
        self.animation.setDuration(200)
        self.animation.setStartValue(start)
        self.animation.setEndValue(end)
        self.animation.setEasingCurve(QEasingCurve.OutBack)
        self.animation.start()

class GradientWidget(QWidget):
    """带有渐变背景的组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.color1 = QColor(30, 60, 114)  # 深蓝色
        self.color2 = QColor(42, 82, 152)  # 中蓝色
        self.color3 = QColor(58, 107, 196)  # 浅蓝色
        self.color4 = QColor(142, 45, 65)  # 红色（NBA主题）
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 创建渐变
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, self.color1)
        gradient.setColorAt(0.5, self.color2)
        gradient.setColorAt(0.7, self.color3)
        gradient.setColorAt(1.0, self.color4)
        
        # 填充背景
        painter.fillRect(self.rect(), QBrush(gradient))

class NBAApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置全局字体
        font = QFont("Times New Roman", 12)
        QApplication.setFont(font)
        
        # 应用设置
        self.setWindowTitle("NBA球星识别系统")
        self.setGeometry(100, 100, 1000, 750)
        # 初始化球员图像字典
        self.player_images = {}  # 修复：添加初始化
        
        # 创建主堆叠窗口
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # 创建主菜单页面
        self.main_menu = self.create_main_menu()
        self.stacked_widget.addWidget(self.main_menu)
        
        # 创建球星识别页面
        self.recognition_page = self.create_recognition_page()
        self.stacked_widget.addWidget(self.recognition_page)
        
        # 创建小游戏页面
        self.game_page = self.create_game_page()
        self.stacked_widget.addWidget(self.game_page)
        
        # 创建游戏结果页面
        self.result_page = self.create_result_page()
        self.stacked_widget.addWidget(self.result_page)
        
        # 创建游戏问题页面
        self.game_question_page = self.create_game_question_page()
        self.stacked_widget.addWidget(self.game_question_page)
        
        # 加载模型和类别映射
        self.load_resources()
        
        # 显示主菜单
        self.stacked_widget.setCurrentIndex(0)
    
    def resource_path(self, relative_path):
        """获取资源的绝对路径"""
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    
    def load_resources(self):
        """加载模型和类别映射"""
        try:
            # 加载模型 - 使用相对路径
            model_path = self.resource_path("output/model/best_model_phase2.h5")
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'ChannelAttention': ChannelAttention},
                    compile=False
                )
                print("✅ 模型加载成功")
            else:
                print(f"❌ 模型文件不存在: {model_path}")
            
            # 加载类别映射
            mapping_path = self.resource_path("output/class_mapping.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.class_mapping = json.load(f)
                print("✅ 类别映射加载成功")
                
                # 获取球员列表（将下划线替换为空格）
                self.player_names = [
                    name.replace('_', ' ') 
                    for name in self.class_mapping['class_to_index'].keys()
                ]
            else:
                print(f"❌ 类别映射文件不存在: {mapping_path}")
            
            # 加载球员图像（用于小游戏）
            self.load_player_images()
            
        except Exception as e:
            print(f"❌ 资源加载失败: {e}")
    
    def load_player_images(self):
        """加载球员图像（用于小游戏）"""
        if not hasattr(self, 'player_names') or not self.player_names:
            print("⚠️ 球员名称列表未初始化或为空")
            return
            
        # 球员文件夹路径
        nba_dir = self.resource_path("data")
        if not os.path.exists(nba_dir):
            print(f"❌ data文件夹不存在: {nba_dir}")
            return
            
        # 确保 player_images 字典已初始化
        if not hasattr(self, 'player_images'):
            self.player_images = {}
            print("ℹ️ player_images 字典已初始化")
            
        for player_folder in os.listdir(nba_dir):
            player_path = os.path.join(nba_dir, player_folder)
            if os.path.isdir(player_path):
                # 获取球员名称（替换下划线）
                player_name = player_folder.replace('_', ' ')
                
                # 获取该球员的所有图像
                images = [
                    os.path.join(player_path, img) 
                    for img in os.listdir(player_path)
                    if img.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                
                if images:
                    self.player_images[player_name] = images
    
    def create_main_menu(self):
        """创建主菜单页面"""
        widget = GradientWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(30)
        layout.setContentsMargins(50, 50, 50, 50)
        
        # 标题区域
        title_frame = QFrame()
        title_frame.setStyleSheet("background-color: rgba(0, 0, 0, 100); border-radius: 20px;")
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(30, 20, 30, 30)
        
        # 标题
        title = QLabel("NBA球星识别系统")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 36, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #FFFFFF; text-shadow: 2px 2px 4px #000000;")
        
        # 副标题
        subtitle = QLabel("探索篮球传奇，认识超级球星")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle_font = QFont("Arial", 18)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: #FFD700;")
        
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        
        # 按钮容器
        button_frame = QFrame()
        button_frame.setStyleSheet("background-color: rgba(0, 0, 0, 120); border-radius: 20px;")
        button_layout = QVBoxLayout(button_frame)
        button_layout.setSpacing(20)
        button_layout.setContentsMargins(40, 30, 40, 30)
        
        # 按钮
        btn_recognition = AnimatedButton("球星识别")
        btn_recognition.clicked.connect(self.show_recognition)
        
        btn_game = AnimatedButton("球星认识小游戏")
        btn_game.clicked.connect(self.show_game)
        
        btn_exit = AnimatedButton("退出系统")
        btn_exit.clicked.connect(self.close)
        
        button_layout.addWidget(btn_recognition)
        button_layout.addWidget(btn_game)
        button_layout.addWidget(btn_exit)
        
        # 添加篮球装饰
        basketball_label = QLabel()
        pixmap = QPixmap(self.resource_path("assets/basketball.png"))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(700, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            basketball_label.setPixmap(pixmap)
        basketball_label.setAlignment(Qt.AlignCenter)
        
        # 添加组件
        layout.addStretch(1)
        layout.addWidget(title_frame)
        layout.addStretch(1)
        layout.addWidget(button_frame)
        layout.addStretch(1)
        layout.addWidget(basketball_label)
        
        return widget
    
    def create_recognition_page(self):
        """创建球星识别页面"""
        widget = GradientWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(25)
        layout.setContentsMargins(30, 20, 30, 30)
        
        # 标题栏
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # 返回按钮
        btn_back = AnimatedButton("返回")
        btn_back.setIcon(QIcon(self.resource_path("assets/back_icon.png")))
        btn_back.setIconSize(QSize(24, 24))
        btn_back.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        btn_back.setMaximumWidth(120)
        
        # 标题
        title = QLabel("球星识别")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 28, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #FFFFFF; text-shadow: 1px 1px 3px #000000;")
        
        header_layout.addWidget(btn_back)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # 主内容区域
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content.setStyleSheet("""
            background-color: rgba(0, 0, 0, 80);
            border-radius: 20px;
            padding: 20px;
        """)
        
        # 图像显示区域
        self.recog_image_label = QLabel()
        self.recog_image_label.setAlignment(Qt.AlignCenter)
        self.recog_image_label.setMinimumSize(500, 350)
        self.recog_image_label.setStyleSheet("""
            background-color: rgba(30, 30, 40, 150);
            border: 2px solid #FFD700;
            border-radius: 15px;
            padding: 10px;
        """)
        
        # 按钮容器
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(20)
        
        # 上传按钮
        btn_upload = AnimatedButton("上传图片")
        btn_upload.setIcon(QIcon(self.resource_path("assets/upload_icon.png")))
        btn_upload.setIconSize(QSize(24, 24))
        btn_upload.clicked.connect(self.upload_image)
        
        # 识别按钮
        btn_recognize = AnimatedButton("识别球星")
        btn_recognize.setIcon(QIcon(self.resource_path("assets/scan_icon.png")))
        btn_recognize.setIconSize(QSize(24, 24))
        btn_recognize.clicked.connect(self.recognize_player)
        self.btn_recognize = btn_recognize
        
        button_layout.addWidget(btn_upload)
        button_layout.addWidget(btn_recognize)
        
        # 结果区域
        self.recog_result_label = QLabel("上传图片后点击识别按钮")
        self.recog_result_label.setAlignment(Qt.AlignCenter)
        self.recog_result_label.setFont(QFont("Arial", 14))
        self.recog_result_label.setWordWrap(True)
        self.recog_result_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 120);
            color: #FFD700;
            border-radius: 15px;
            padding: 20px;
            min-height: 100px;
        """)
        
        # 添加内容
        content_layout.addWidget(self.recog_image_label, 1)
        content_layout.addWidget(button_container)
        content_layout.addWidget(self.recog_result_label, 1)
        
        # 添加装饰
        decoration = QLabel()
        pixmap = QPixmap(self.resource_path("assets/nba_logo.png"))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            decoration.setPixmap(pixmap)
        decoration.setAlignment(Qt.AlignCenter)
        
        # 添加组件
        layout.addWidget(header)
        layout.addWidget(content, 1)
        layout.addWidget(decoration)
        
        return widget
    
    def create_game_page(self):
        """创建小游戏页面"""
        widget = GradientWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(25)
        layout.setContentsMargins(30, 20, 30, 30)
        
        # 标题栏
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # 返回按钮
        btn_back = AnimatedButton("返回")
        btn_back.setIcon(QIcon(self.resource_path("assets/back_icon.png")))
        btn_back.setIconSize(QSize(24, 24))
        btn_back.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        btn_back.setMaximumWidth(120)
        
        # 标题
        title = QLabel("球星认识小游戏")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 28, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #FFFFFF; text-shadow: 1px 1px 3px #000000;")
        
        header_layout.addWidget(btn_back)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # 主内容区域
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(30)
        content_layout.setContentsMargins(50, 30, 50, 30)
        content.setStyleSheet("""
            background-color: rgba(0, 0, 0, 80);
            border-radius: 20px;
        """)
        
        # 游戏说明
        instruction = QLabel("测试你对NBA球星的了解程度！\n选择难度级别，开始挑战吧！")
        instruction.setAlignment(Qt.AlignCenter)
        instruction.setFont(QFont("Arial", 16))
        instruction.setStyleSheet("color: #FFD700;")
        
        # 游戏选项区域
        group = QGroupBox("选择游戏难度")
        group.setFont(QFont("Arial", 16, QFont.Bold))
        group.setStyleSheet("""
            QGroupBox {
                color: #FFD700;
                border: 2px solid #FFD700;
                border-radius: 15px;
                margin-top: 10px;
            }
        """)
        group_layout = QHBoxLayout(group)
        group_layout.setSpacing(30)
        
        self.difficulty_group = QButtonGroup()
        btn_easy = QRadioButton("简单 (5位球星)")
        btn_medium = QRadioButton("中等 (10位球星)")
        btn_hard = QRadioButton("困难 (20位球星)")
        
        # 设置单选按钮样式
        radio_style = """
            QRadioButton {
                color: #FFFFFF;
                font-size: 16px;
                padding: 15px;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
                border-radius: 10px;
                border: 2px solid #FFD700;
            }
            QRadioButton::indicator:checked {
                background-color: #FFD700;
            }
        """
        btn_easy.setStyleSheet(radio_style)
        btn_medium.setStyleSheet(radio_style)
        btn_hard.setStyleSheet(radio_style)
        
        self.difficulty_group.addButton(btn_easy, 5)
        self.difficulty_group.addButton(btn_medium, 10)
        self.difficulty_group.addButton(btn_hard, 20)
        
        group_layout.addWidget(btn_easy)
        group_layout.addWidget(btn_medium)
        group_layout.addWidget(btn_hard)
        
        # 开始游戏按钮
        btn_start = AnimatedButton("开始游戏")
        btn_start.setIcon(QIcon(self.resource_path("assets/start_icon.png")))
        btn_start.setIconSize(QSize(24, 24))
        btn_start.clicked.connect(self.start_game)
        btn_start.setFont(QFont("Arial", 16, QFont.Bold))
        
        # 添加装饰
        trophy_label = QLabel()
        pixmap = QPixmap(self.resource_path("assets/trophy.png"))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            trophy_label.setPixmap(pixmap)
        trophy_label.setAlignment(Qt.AlignCenter)
        
        # 添加内容
        content_layout.addStretch(1)
        content_layout.addWidget(instruction)
        content_layout.addStretch(1)
        content_layout.addWidget(group)
        content_layout.addStretch(1)
        content_layout.addWidget(btn_start, 0, Qt.AlignCenter)
        content_layout.addStretch(1)
        content_layout.addWidget(trophy_label)
        
        # 添加组件
        layout.addWidget(header)
        layout.addWidget(content, 1)
        
        return widget
    
    def create_game_question_page(self):
        """创建游戏问题页面"""
        widget = GradientWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 20, 30, 30)
        
        # 标题栏
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # 返回按钮
        btn_back = AnimatedButton("返回")
        btn_back.setIcon(QIcon(self.resource_path("assets/back_icon.png")))
        btn_back.setIconSize(QSize(24, 24))
        btn_back.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        btn_back.setMaximumWidth(120)
        
        # 标题
        self.game_title = QLabel()
        self.game_title.setAlignment(Qt.AlignCenter)
        self.game_title.setFont(QFont("Arial", 22, QFont.Bold))
        self.game_title.setStyleSheet("color: #FFFFFF; text-shadow: 1px 1px 3px #000000;")
        
        header_layout.addWidget(btn_back)
        header_layout.addWidget(self.game_title)
        header_layout.addStretch()
        
        # 主内容区域
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(30, 20, 30, 20)
        content.setStyleSheet("background-color: rgba(0, 0, 0, 80); border-radius: 20px;")
        
        # 图像显示区域
        self.game_image_frame = QFrame()
        self.game_image_frame.setStyleSheet("""
            background-color: rgba(30, 30, 40, 150);
            border: 2px solid #FFD700;
            border-radius: 15px;
        """)
        image_layout = QVBoxLayout(self.game_image_frame)
        image_layout.setContentsMargins(10, 10, 10, 10)
        
        self.game_image_label = QLabel()
        self.game_image_label.setAlignment(Qt.AlignCenter)
        self.game_image_label.setMinimumSize(400, 300)
        image_layout.addWidget(self.game_image_label)
        
        # 选项组
        options_group = QGroupBox("请选择正确的球星名字")
        options_group.setFont(QFont("Arial", 16, QFont.Bold))
        options_group.setStyleSheet("""
            QGroupBox {
                color: #FFD700;
                border: 2px solid #FFD700;
                border-radius: 15px;
                margin-top: 10px;
            }
        """)
        option_layout = QVBoxLayout(options_group)
        option_layout.setSpacing(15)
        
        self.option_group = QButtonGroup()
        self.option_buttons = []
        
        # 创建选项按钮
        for i in range(4):
            btn = QRadioButton()
            btn.setFont(QFont("Arial", 14))
            btn.setStyleSheet("""
                QRadioButton {
                    color: #FFFFFF;
                    padding: 15px;
                    background-color: rgba(30, 30, 40, 150);
                    border-radius: 10px;
                }
                QRadioButton:hover {
                    background-color: rgba(50, 50, 70, 200);
                }
                QRadioButton::indicator {
                    width: 20px;
                    height: 20px;
                    border-radius: 10px;
                    border: 2px solid #FFD700;
                }
                QRadioButton::indicator:checked {
                    background-color: #FFD700;
                }
            """)
            self.option_buttons.append(btn)
            self.option_group.addButton(btn, i)
            option_layout.addWidget(btn)
        
        # 提交按钮
        btn_submit = AnimatedButton("提交答案")
        btn_submit.setIcon(QIcon(self.resource_path("assets/submit_icon.png")))
        btn_submit.setIconSize(QSize(24, 24))
        btn_submit.clicked.connect(self.check_answer)
        
        # 进度标签
        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.progress_label.setStyleSheet("color: #FFD700;")
        
        # 添加内容
        content_layout.addWidget(self.game_image_frame, 1)
        content_layout.addWidget(options_group, 1)
        content_layout.addWidget(btn_submit, 0, Qt.AlignCenter)
        content_layout.addWidget(self.progress_label)
        
        # 添加组件
        layout.addWidget(header)
        layout.addWidget(content, 1)
        
        return widget
    
    def create_result_page(self):
        """创建游戏结果页面"""
        widget = GradientWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(50, 30, 50, 30)
        
        # 标题
        title = QLabel("游戏结果")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 36, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #FFFFFF; text-shadow: 2px 2px 4px #000000;")
        
        # 结果容器
        result_container = QWidget()
        result_layout = QVBoxLayout(result_container)
        result_layout.setSpacing(20)
        result_layout.setContentsMargins(40, 30, 40, 30)
        result_container.setStyleSheet("""
            background-color: rgba(0, 0, 0, 100);
            border-radius: 20px;
            border: 2px solid #FFD700;
        """)
        
        # 结果标签
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.result_label.setStyleSheet("color: #FFD700;")
        
        # 分数标签
        self.score_label = QLabel()
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setFont(QFont("Arial", 48, QFont.Bold))
        self.score_label.setStyleSheet("color: #FFFFFF;")
        
        # 动画标签
        self.animation_label = QLabel()
        self.animation_label.setAlignment(Qt.AlignCenter)
        self.animation_label.setMinimumSize(300, 300)
        
        # 按钮容器
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(30)
        
        # 再玩一次按钮
        btn_restart = AnimatedButton("再玩一次")
        btn_restart.setIcon(QIcon(self.resource_path("assets/restart_icon.png")))
        btn_restart.setIconSize(QSize(24, 24))
        btn_restart.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        
        # 返回主菜单按钮
        btn_menu = AnimatedButton("返回主菜单")
        btn_menu.setIcon(QIcon(self.resource_path("assets/home_icon.png")))
        btn_menu.setIconSize(QSize(24, 24))
        btn_menu.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        
        button_layout.addWidget(btn_restart)
        button_layout.addWidget(btn_menu)
        
        # 添加内容
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.score_label)
        result_layout.addWidget(self.animation_label, 1)
        
        # 添加组件
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(result_container, 1)
        layout.addStretch(1)
        layout.addWidget(button_container, 0, Qt.AlignCenter)
        layout.addStretch(1)
        
        return widget
    
    def show_recognition(self):
        """显示球星识别页面"""
        self.recog_image_label.clear()
        self.recog_result_label.setText("上传图片后点击识别按钮")
        self.btn_recognize.setEnabled(False)
        self.stacked_widget.setCurrentIndex(1)
    
    def show_game(self):
        """显示小游戏页面"""
        self.stacked_widget.setCurrentIndex(2)
    
    def upload_image(self):
        """上传图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", 
            "图片文件 (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            # 显示图片
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # 缩放图片以适应标签
                scaled_pixmap = pixmap.scaled(
                    self.recog_image_label.width(), 
                    self.recog_image_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.recog_image_label.setPixmap(scaled_pixmap)
                self.current_image_path = file_path
                self.btn_recognize.setEnabled(True)
                self.recog_result_label.setText("图片已上传，点击'识别球星'按钮进行分析")
    
    def recognize_player(self):
        """识别球星"""
        if not hasattr(self, 'current_image_path') or not self.model:
            self.recog_result_label.setText("请先上传图片或等待模型加载完成")
            return
            
        try:
            # 添加加载提示
            self.recog_result_label.setText("正在识别中，请稍后...（第一次加载需要较长时间哦）")
            QApplication.processEvents()  # 强制刷新UI
            
            # 预处理图像
            processed_img, original_img = self.preprocess_image(self.current_image_path)
            
            # 进行预测
            predictions = self.model.predict(processed_img)[0]
            
            # 获取top-3预测结果
            top_indices = np.argsort(predictions)[::-1][:3]
            top_indices = [int(idx) for idx in top_indices]
            
            # 获取球员名称和概率
            top_players = [
                self.class_mapping['index_to_class'][str(idx)].replace('_', ' ') 
                for idx in top_indices
            ]
            top_probs = predictions[top_indices]
            
            # 构建结果字符串
            result_text = "🏀 识别结果:\n\n"
            for i, (player, prob) in enumerate(zip(top_players, top_probs)):
                result_text += f"{i+1}. {player}: {prob*100:.2f}%\n"
            
            self.recog_result_label.setText(result_text)
            
        except Exception as e:
            self.recog_result_label.setText(f"⚠️ 识别失败: {str(e)}")
    
    def preprocess_image(self, image_path, target_size=(300, 300)):
        """预处理图像用于模型预测"""
        img = Image.open(image_path)
        # 保留原始图像用于显示
        original_img = img.copy()
        # 调整大小为模型输入尺寸
        img = img.resize(target_size)
        img_array = np.array(img)
        
        # 处理图像通道
        if len(img_array.shape) == 2:  # 灰度图
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA转RGB
            img_array = img_array[..., :3]
        
        img_array = img_array.astype('float32') / 255.0
        return np.expand_dims(img_array, axis=0), original_img
    
    def start_game(self):
        """开始小游戏"""
        # 获取选择的难度
        difficulty = self.difficulty_group.checkedId()
        if difficulty == -1:
            QMessageBox.warning(self, "提示", "请先选择游戏难度!")
            return
            
        # 检查球员数据是否加载成功
        if not hasattr(self, 'player_images') or len(self.player_images) == 0:
            QMessageBox.warning(self, "错误", "球员数据加载失败，无法开始游戏!")
            return
            
        # 随机选择球员
        if not self.player_names or len(self.player_names) < difficulty:
            QMessageBox.warning(self, "错误", "没有足够的球员数据!")
            return
            
        # 随机选择指定数量的球员
        self.selected_players = random.sample(self.player_names, difficulty)
        self.current_question = 0
        self.score = 0
        
        # 显示第一个问题
        self.show_question()
        self.stacked_widget.setCurrentWidget(self.game_question_page)
    
    def show_question(self):
        """显示当前问题"""
        if self.current_question >= len(self.selected_players):
            # 游戏结束，显示结果
            self.show_game_result()
            return
            
        # 获取当前球员
        current_player = self.selected_players[self.current_question]
        
        # 更新标题
        self.game_title.setText(f"问题 {self.current_question + 1}/{len(self.selected_players)}")
        
        # 更新进度
        self.progress_label.setText(f"当前得分: {self.score}/{len(self.selected_players)}")
        
        # 显示球员图片
        if current_player in self.player_images and self.player_images[current_player]:
            # 随机选择一张图片
            image_path = random.choice(self.player_images[current_player])
            pixmap = QPixmap(image_path)
            
            if not pixmap.isNull():
                # 计算缩放后的尺寸（固定高度为窗口高度，保持原图比例）
                target_height = self.game_image_label.height()
                scaled_pixmap = pixmap.scaled(
                    int(pixmap.width() * (target_height / pixmap.height())),
                    target_height,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                # 设置图片并居中显示
                self.game_image_label.setPixmap(scaled_pixmap)
                self.game_image_label.setAlignment(Qt.AlignCenter)
        else:
            # 如果找不到该球员的图片，显示错误信息
            self.game_image_label.setText(f"无法加载 {current_player} 的图片")
        
        # 生成选项
        options = [current_player]  # 正确答案
        
        # 添加三个错误选项
        wrong_options = [
            p for p in self.player_names 
            if p != current_player and p in self.player_images
        ]
        
        if len(wrong_options) >= 3:
            options.extend(random.sample(wrong_options, 3))
        else:
            # 如果没有足够的错误选项，重复添加
            while len(options) < 4:
                options.append(random.choice(self.player_names))
        
        # 随机打乱选项
        random.shuffle(options)
        
        # 设置选项文本
        for i, btn in enumerate(self.option_buttons):
            btn.setText(options[i])
            btn.setChecked(False)
        
        # 保存正确答案
        self.correct_answer = current_player
    
    def check_answer(self):
        """检查答案"""
        # 获取选中的选项
        selected_id = self.option_group.checkedId()
        if selected_id == -1:
            QMessageBox.warning(self, "提示", "请选择一个答案!")
            return
            
        # 获取选中的文本
        selected_player = self.option_buttons[selected_id].text()
        
        # 检查是否正确
        if selected_player == self.correct_answer:
            self.score += 1
            feedback = "✅ 回答正确!"
        else:
            feedback = f"❌ 回答错误! 正确答案是: {self.correct_answer}"
        
        # 显示反馈
        QMessageBox.information(self, "结果", feedback)
        
        # 进入下一题
        self.current_question += 1
        self.show_question()
    
    def show_game_result(self):
        """显示游戏结果"""
        # 设置结果文本
        self.result_label.setText("游戏结束!")
        self.score_label.setText(f"得分: {self.score}/{len(self.selected_players)}")
        
        # 根据得分设置动画
        if self.score == len(self.selected_players):
            # 满分
            self.result_label.setText("🎉 太棒了! 满分!")
            # 加载庆祝动画
            movie = QMovie(self.resource_path("assets/celebration.gif"))
            movie.setScaledSize(QSize(300, int(300*640/692)))
            if movie.isValid():
                self.animation_label.setMovie(movie)
                movie.start()
            else:
                self.animation_label.setText("🎉🎉🎉")
        elif self.score >= len(self.selected_players) * 0.7:
            # 良好
            self.result_label.setText("👍 表现不错!")
            movie = QMovie(self.resource_path("assets/thumbs_up.gif"))
            movie.setScaledSize(QSize(300, int(300*640/658)))
            if movie.isValid():
                self.animation_label.setMovie(movie)
                movie.start()
            else:
                self.animation_label.setText("👍")
        else:
            # 需要改进
            self.result_label.setText("💪 继续努力!")
            movie = QMovie(self.resource_path("assets/encouragement.gif"))
            movie.setScaledSize(QSize(300, int(300*640/688)))
            if movie.isValid():
                self.animation_label.setMovie(movie)
                movie.start()
            else:
                self.animation_label.setText("💪")
        
        # 显示结果页面
        self.stacked_widget.setCurrentIndex(3)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    # 创建并显示主窗口
    window = NBAApp()
    window.show()
    
    # 进入主事件循环
    sys.exit(app.exec_())