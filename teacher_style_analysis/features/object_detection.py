"""对象检测模块，负责使用YOLO模型进行目标检测"""
import os
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import BASE_DIR, MODEL_CONFIG

class YOLOObjectDetector:
    """YOLO目标检测类，用于检测视频中的目标对象"""
    
    def __init__(self):
        """初始化YOLO模型"""
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            print("初始化YOLO模型...")
            
            # 使用绝对路径加载YOLO模型
            yolo_path = os.path.join(BASE_DIR, MODEL_CONFIG['yolo_model_path'])
            print(f"YOLO模型绝对路径: {yolo_path}")
            print(f"文件是否存在: {os.path.exists(yolo_path)}")
            
            if os.path.exists(yolo_path):
                print(f"文件大小: {os.path.getsize(yolo_path)} 字节")
            
            # 尝试加载模型
            self.model = YOLO(yolo_path)
            print(f"YOLO模型加载成功，路径: {yolo_path}")
            print(f"模型类型: {type(self.model)}")
            
            # 测试模型是否能正常工作
            test_img = np.ones((640, 640, 3), dtype=np.uint8) * 255
            try:
                results = self.model(test_img)
                print(f"YOLO模型推理测试成功，结果类型: {type(results)}")
                print(f"检测到 {len(results[0].boxes)} 个目标")
            except Exception as infer_e:
                print(f"YOLO模型推理测试失败: {infer_e}")
                import traceback
                traceback.print_exc()
                self.model = None
                
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.25, 
              class_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        对输入帧进行目标检测
        
        Args:
            frame: 输入的视频帧
            confidence_threshold: 置信度阈值
            class_filter: 过滤的类别列表，只返回列表中的类别
            
        Returns:
            检测结果列表，每个元素包含类别、置信度和边界框
        """
        if self.model is None:
            return []
        
        results = self.model(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取检测类别
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 检查置信度
                if conf < confidence_threshold:
                    continue
                
                # 获取类别名称
                cls_name = result.names[cls]
                
                # 检查是否需要过滤类别
                if class_filter and cls_name not in class_filter:
                    continue
                
                # 获取边界框
                x1, y1, x2, y2 = box.xyxy[0]
                
                # 计算中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detection = {
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(center_x), float(center_y)]
                }
                
                detections.append(detection)
        
        return detections
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None
