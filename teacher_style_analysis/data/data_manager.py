"""数据管理模块，负责视频、音频和文本数据的上传、存储和索引"""
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
import mysql.connector
import redis
from typing import Dict, Optional, Union

from ..config.config import (
    VIDEO_DIR, AUDIO_DIR, TEXT_DIR, 
    DB_CONFIG, REDIS_CONFIG
)


class DataManager:
    """数据管理器类"""
    
    def __init__(self):
        self._init_db_connection()
        self._init_redis_connection()
    
    def _init_db_connection(self):
        """初始化数据库连接"""
        try:
            self.db_conn = mysql.connector.connect(**DB_CONFIG)
            self.db_cursor = self.db_conn.cursor(dictionary=True)
            self._create_tables()
        except Exception as e:
            print(f"数据库连接失败: {e}")
            self.db_conn = None
            self.db_cursor = None
    
    def _init_redis_connection(self):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.Redis(**REDIS_CONFIG)
            self.redis_client.ping()
        except Exception as e:
            print(f"Redis连接失败: {e}")
            self.redis_client = None
    
    def _create_tables(self):
        """创建必要的数据表"""
        if not self.db_conn:
            return
        
        # 创建视频数据表
        create_video_table = """
        CREATE TABLE IF NOT EXISTS videos (
            id VARCHAR(36) PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            filepath VARCHAR(512) NOT NULL,
            duration FLOAT,
            upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(50) DEFAULT 'uploaded',
            teacher_id VARCHAR(36),
            course_id VARCHAR(36)
        )
        """
        
        # 创建音频数据表
        create_audio_table = """
        CREATE TABLE IF NOT EXISTS audios (
            id VARCHAR(36) PRIMARY KEY,
            video_id VARCHAR(36),
            filepath VARCHAR(512) NOT NULL,
            duration FLOAT,
            created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos(id)
        )
        """
        
        # 创建文本数据表
        create_text_table = """
        CREATE TABLE IF NOT EXISTS transcripts (
            id VARCHAR(36) PRIMARY KEY,
            video_id VARCHAR(36),
            filepath VARCHAR(512) NOT NULL,
            created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos(id)
        )
        """
        
        try:
            self.db_cursor.execute(create_video_table)
            self.db_cursor.execute(create_audio_table)
            self.db_cursor.execute(create_text_table)
            self.db_conn.commit()
        except Exception as e:
            print(f"创建数据表失败: {e}")
    
    def upload_video(self, file_path: str, teacher_id: Optional[str] = None, 
                    course_id: Optional[str] = None) -> Dict:
        """
        上传视频文件
        
        Args:
            file_path: 视频文件路径
            teacher_id: 教师ID
            course_id: 课程ID
            
        Returns:
            包含视频信息的字典
        """
        try:
            # 生成唯一ID
            video_id = str(uuid.uuid4())
            filename = os.path.basename(file_path)
            
            # 复制文件到存储目录
            dest_path = VIDEO_DIR / f"{video_id}_{filename}"
            shutil.copy2(file_path, dest_path)
            
            # 保存到数据库
            if self.db_conn:
                query = """
                INSERT INTO videos (id, filename, filepath, teacher_id, course_id)
                VALUES (%s, %s, %s, %s, %s)
                """
                values = (video_id, filename, str(dest_path), teacher_id, course_id)
                self.db_cursor.execute(query, values)
                self.db_conn.commit()
            
            # 缓存到Redis
            if self.redis_client:
                self.redis_client.setex(
                    f"video:{video_id}",
                    3600,  # 1小时过期
                    str(dest_path)
                )
            
            return {
                'id': video_id,
                'filename': filename,
                'filepath': str(dest_path),
                'upload_time': datetime.now().isoformat(),
                'status': 'uploaded'
            }
            
        except Exception as e:
            print(f"视频上传失败: {e}")
            raise
    
    def save_audio(self, video_id: str, audio_path: str) -> Dict:
        """
        保存从视频中提取的音频文件
        
        Args:
            video_id: 关联的视频ID
            audio_path: 音频文件路径
            
        Returns:
            包含音频信息的字典
        """
        try:
            audio_id = str(uuid.uuid4())
            filename = f"{video_id}.wav"
            dest_path = AUDIO_DIR / filename
            
            shutil.copy2(audio_path, dest_path)
            
            if self.db_conn:
                query = """
                INSERT INTO audios (id, video_id, filepath)
                VALUES (%s, %s, %s)
                """
                values = (audio_id, video_id, str(dest_path))
                self.db_cursor.execute(query, values)
                self.db_conn.commit()
            
            return {
                'id': audio_id,
                'video_id': video_id,
                'filepath': str(dest_path)
            }
            
        except Exception as e:
            print(f"音频保存失败: {e}")
            raise
    
    def save_transcript(self, video_id: str, transcript_content: str) -> Dict:
        """
        保存转录文本
        
        Args:
            video_id: 关联的视频ID
            transcript_content: 转录文本内容
            
        Returns:
            包含文本信息的字典
        """
        try:
            transcript_id = str(uuid.uuid4())
            filepath = TEXT_DIR / f"{video_id}_transcript.txt"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(transcript_content)
            
            if self.db_conn:
                query = """
                INSERT INTO transcripts (id, video_id, filepath)
                VALUES (%s, %s, %s)
                """
                values = (transcript_id, video_id, str(filepath))
                self.db_cursor.execute(query, values)
                self.db_conn.commit()
            
            return {
                'id': transcript_id,
                'video_id': video_id,
                'filepath': str(filepath)
            }
            
        except Exception as e:
            print(f"转录文本保存失败: {e}")
            raise
    
    def get_video_info(self, video_id: str) -> Optional[Dict]:
        """
        获取视频信息
        
        Args:
            video_id: 视频ID
            
        Returns:
            视频信息字典，如果不存在则返回None
        """
        # 先从Redis缓存获取
        if self.redis_client:
            cached_path = self.redis_client.get(f"video:{video_id}")
            if cached_path:
                return {
                    'id': video_id,
                    'filepath': cached_path.decode('utf-8')
                }
        
        # 从数据库获取
        if self.db_conn:
            query = "SELECT * FROM videos WHERE id = %s"
            self.db_cursor.execute(query, (video_id,))
            result = self.db_cursor.fetchone()
            
            # 更新缓存
            if result and self.redis_client:
                self.redis_client.setex(
                    f"video:{video_id}",
                    3600,
                    result['filepath']
                )
            
            return result
        
        return None
    
    def update_video_status(self, video_id: str, status: str) -> bool:
        """
        更新视频处理状态
        
        Args:
            video_id: 视频ID
            status: 新状态
            
        Returns:
            是否更新成功
        """
        try:
            if self.db_conn:
                query = "UPDATE videos SET status = %s WHERE id = %s"
                self.db_cursor.execute(query, (status, video_id))
                self.db_conn.commit()
                return True
            return False
        except Exception as e:
            print(f"更新视频状态失败: {e}")
            return False
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'db_cursor') and self.db_cursor:
            self.db_cursor.close()
        if hasattr(self, 'db_conn') and self.db_conn:
            self.db_conn.close()


# 创建数据管理器实例
data_manager = DataManager()