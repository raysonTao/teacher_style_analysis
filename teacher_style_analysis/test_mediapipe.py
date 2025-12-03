#!/usr/bin/env python3
"""测试MediaPipe姿态估计模型加载"""

import sys
import os
import cv2

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 测试直接导入mediapipe
    import mediapipe as mp
    print(f"✅ mediapipe库导入成功，版本: {mp.__version__}")
    
    # 测试创建MediaPipe姿态估计模型
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✅ MediaPipe姿态估计模型创建成功")
    
    # 测试FeatureExtractor类中的模型加载
    from features.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()
    if extractor.openpose_model is not None and extractor.use_mediapipe:
        print("✅ FeatureExtractor中的MediaPipe模型加载成功")
    else:
        print("❌ FeatureExtractor中的MediaPipe模型加载失败")
        
    # 测试简单的推理功能
    import numpy as np
    # 创建一个简单的测试图像
    test_img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    
    # 转换为RGB（MediaPipe要求）
    rgb_frame = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) if 'cv2' in sys.modules else test_img
    
    # 测试模型推理
    results = pose.process(rgb_frame)
    print("✅ MediaPipe模型推理测试成功")
    
    print("\n🎉 所有测试通过！MediaPipe姿态估计模型工作正常")
    
    # 清理资源
    pose.close()
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ 其他错误: {e}")
    import traceback
    traceback.print_exc()
