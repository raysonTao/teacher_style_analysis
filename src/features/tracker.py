"""目标跟踪模块，仅使用DeepSORT"""
import os
import sys
from typing import List, Dict, Optional

import numpy as np

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MODEL_CONFIG, VIDEO_CONFIG, logger


class TeacherTracker:
    """教师身份连续追踪封装器"""

    def __init__(self):
        self.tracker_type = VIDEO_CONFIG.get('tracker_type', 'deepsort')
        self.tracker = None
        self._init_tracker()

    def _init_tracker(self):
        if self.tracker_type != 'deepsort':
            logger.error(f"当前仅支持DeepSORT跟踪，配置为: {self.tracker_type}")
            self.tracker = None
            return

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            model_path = MODEL_CONFIG.get('deepsort_model_path')
            model_path = model_path if model_path and os.path.exists(model_path) else None

            self.tracker = DeepSort(
                max_age=VIDEO_CONFIG.get('tracker_max_age', 30),
                n_init=2,
                nn_budget=100,
                embedder='osnet_x0_25',
                embedder_model_filename=model_path,
                half=False,
                bgr=True
            )
            logger.info("DeepSORT跟踪器初始化成功")
        except Exception as e:
            logger.error(f"DeepSORT不可用: {e}")
            self.tracker = None

    def update(self, detections: List[Dict], frame: Optional[np.ndarray] = None) -> List[Dict]:
        """输入检测结果并更新跟踪状态"""
        if self.tracker is None:
            logger.error("DeepSORT跟踪器未初始化，无法更新跟踪")
            return []

        try:
            ds_dets = []
            for det in detections:
                bbox = det['bbox']
                confidence = det.get('confidence', 0.0)
                class_name = det.get('class_name', 'person')
                ds_dets.append((bbox, confidence, class_name))

            tracks = self.tracker.update_tracks(ds_dets, frame=frame)
            output_tracks = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                ltrb = track.to_ltrb()
                output_tracks.append({
                    'track_id': track.track_id,
                    'bbox': [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])],
                    'confidence': float(getattr(track, 'det_conf', 0.0) or 0.0),
                    'class_name': 'person',
                    'hits': track.hits,
                    'missed': track.time_since_update
                })
            return output_tracks
        except Exception as e:
            logger.warning(f"DeepSORT更新失败，回退空结果: {e}")
            return []
