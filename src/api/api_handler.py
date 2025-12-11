"""API模块，负责处理系统接口请求"""
import os
import uuid
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
try:
    import uvicorn
except ImportError:
    uvicorn = None

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    SYSTEM_CONFIG, API_CONFIG, DATA_DIR, TEMP_DIR,
    RESULTS_DIR, FEEDBACK_DIR
)
from data.data_manager import data_manager
from features.feature_extractor import feature_extractor
from models.core.style_classifier import style_classifier
from feedback.feedback_generator import feedback_generator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('teacher_style_analysis_api')

# 创建FastAPI应用
app = FastAPI(
    title="教师风格画像分析系统API",
    description="基于多模态数据的教师教学风格分析系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/upload_video", response_model=Dict[str, Any])
async def upload_video(
    video: UploadFile = File(...),
    teacher_id: str = Form(...),
    discipline: str = Form(...),
    grade: str = Form(...),
    class_id: Optional[str] = Form(None),
    lesson_title: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    上传教学视频
    
    Args:
        video: 视频文件
        teacher_id: 教师ID
        discipline: 学科
        grade: 年级
        class_id: 班级ID（可选）
        lesson_title: 课程标题（可选）
        
    Returns:
        上传结果，包含视频ID和状态
    """
    try:
        # 生成视频ID
        video_id = str(uuid.uuid4())[:10]
        
        # 验证文件类型
        if not video.filename.endswith(('.mp4', '.avi', '.mov', '.wmv', '.flv')):
            raise HTTPException(status_code=400, detail="不支持的视频格式")
        
        # 保存视频文件
        video_path = DATA_DIR / 'videos' / f"{video_id}.mp4"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # 保存视频信息
        video_info = {
            'video_id': video_id,
            'teacher_id': teacher_id,
            'discipline': discipline,
            'grade': grade,
            'class_id': class_id,
            'lesson_title': lesson_title,
            'filename': video.filename,
            'upload_time': datetime.now().isoformat(),
            'status': 'uploaded',
            'file_path': str(video_path)
        }
        
        # 存储到数据库
        data_manager.save_video_info(video_info)
        
        logger.info(f"视频上传成功: video_id={video_id}, teacher_id={teacher_id}")
        
        return {
            "success": True,
            "video_id": video_id,
            "message": "视频上传成功",
            "data": video_info
        }
        
    except HTTPException as e:
        logger.error(f"视频上传失败 - HTTP错误: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"视频上传失败 - 系统错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.post("/api/analyze_style/{video_id}", response_model=Dict[str, Any])
async def analyze_teaching_style(video_id: str) -> Dict[str, Any]:
    """
    分析教学风格
    
    Args:
        video_id: 视频ID
        
    Returns:
        风格分析结果
    """
    try:
        # 检查视频状态
        video_info = data_manager.get_video_info(video_id)
        if not video_info:
            raise HTTPException(status_code=404, detail="视频不存在")
        
        # 更新状态为分析中
        data_manager.update_video_status(video_id, "analyzing")
        
        # 1. 提取特征
        logger.info(f"开始提取视频特征: video_id={video_id}")
        features_path = feature_extractor.extract_and_save(video_info['file_path'], video_id)
        
        # 2. 风格分类
        logger.info(f"开始风格分类: video_id={video_id}")
        result_path = style_classifier.classify_and_save(video_id)
        
        # 3. 生成反馈
        logger.info(f"开始生成反馈: video_id={video_id}")
        feedback = feedback_generator.generate_feedback_report(
            video_id, video_info['discipline'], video_info['grade']
        )
        
        # 更新状态为完成
        data_manager.update_video_status(video_id, "completed")
        
        logger.info(f"教学风格分析完成: video_id={video_id}")
        
        # 读取分析结果
        with open(result_path, 'r', encoding='utf-8') as f:
            style_result = json.load(f)
        
        return {
            "success": True,
            "video_id": video_id,
            "message": "教学风格分析完成",
            "data": {
                "style_analysis": style_result,
                "feedback": feedback,
                "video_info": video_info
            }
        }
        
    except HTTPException as e:
        data_manager.update_video_status(video_id, "failed", error_info=str(e.detail))
        logger.error(f"教学风格分析失败 - HTTP错误: {e.detail}")
        raise
    except Exception as e:
        data_manager.update_video_status(video_id, "failed", error_info=str(e))
        logger.error(f"教学风格分析失败 - 系统错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.get("/api/videos", response_model=Dict[str, Any])
async def list_videos(
    teacher_id: Optional[str] = None,
    discipline: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    """
    查询视频列表
    
    Args:
        teacher_id: 教师ID（可选）
        discipline: 学科（可选）
        status: 状态（可选）
        page: 页码
        page_size: 每页数量
        
    Returns:
        视频列表
    """
    try:
        videos = data_manager.list_videos(
            teacher_id=teacher_id,
            discipline=discipline,
            status=status,
            page=page,
            page_size=page_size
        )
        
        total = data_manager.count_videos(
            teacher_id=teacher_id,
            discipline=discipline,
            status=status
        )
        
        return {
            "success": True,
            "data": {
                "videos": videos,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": (total + page_size - 1) // page_size
                }
            }
        }
        
    except Exception as e:
        logger.error(f"查询视频列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/api/videos/{video_id}", response_model=Dict[str, Any])
async def get_video_detail(video_id: str) -> Dict[str, Any]:
    """
    获取视频详情
    
    Args:
        video_id: 视频ID
        
    Returns:
        视频详情和分析结果
    """
    try:
        # 获取视频信息
        video_info = data_manager.get_video_info(video_id)
        if not video_info:
            raise HTTPException(status_code=404, detail="视频不存在")
        
        # 尝试获取分析结果
        result_file = RESULTS_DIR / f"{video_id}_style_result.json"
        feedback_file = FEEDBACK_DIR / f"{video_id}_feedback.json"
        
        style_result = None
        feedback = None
        
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                style_result = json.load(f)
        
        if feedback_file.exists():
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback = json.load(f)
        
        return {
            "success": True,
            "data": {
                "video_info": video_info,
                "style_analysis": style_result,
                "feedback": feedback
            }
        }
        
    except HTTPException as e:
        logger.error(f"获取视频详情失败 - HTTP错误: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"获取视频详情失败 - 系统错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/api/teachers/{teacher_id}/growth", response_model=Dict[str, Any])
async def get_teacher_growth(teacher_id: str) -> Dict[str, Any]:
    """
    获取教师成长轨迹
    
    Args:
        teacher_id: 教师ID
        
    Returns:
        成长分析结果
    """
    try:
        # 分析教学成长
        growth_analysis = feedback_generator.analyze_teaching_growth(teacher_id)
        
        # 获取教师的所有视频
        videos = data_manager.list_videos(teacher_id=teacher_id)
        
        # 整理历史分析结果
        history_results = []
        for video in videos:
            if video.get('status') == 'completed':
                video_id = video['video_id']
                feedback_file = FEEDBACK_DIR / f"{video_id}_feedback.json"
                
                if feedback_file.exists():
                    with open(feedback_file, 'r', encoding='utf-8') as f:
                        feedback = json.load(f)
                    
                    history_results.append({
                        'video_id': video_id,
                        'date': video.get('upload_time'),
                        'smi': feedback.get('smi', {}).get('score'),
                        'main_style': feedback.get('teaching_style', {}).get('main_styles', [[]])[0],
                        'discipline': video.get('discipline')
                    })
        
        # 按日期排序
        history_results.sort(key=lambda x: x['date'], reverse=True)
        
        return {
            "success": True,
            "data": {
                "teacher_id": teacher_id,
                "growth_analysis": growth_analysis,
                "history_results": history_results,
                "total_analyses": len(history_results)
            }
        }
        
    except Exception as e:
        logger.error(f"获取教师成长轨迹失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.delete("/api/videos/{video_id}", response_model=Dict[str, Any])
async def delete_video(video_id: str) -> Dict[str, Any]:
    """
    删除视频及相关数据
    
    Args:
        video_id: 视频ID
        
    Returns:
        删除结果
    """
    try:
        # 检查视频是否存在
        video_info = data_manager.get_video_info(video_id)
        if not video_info:
            raise HTTPException(status_code=404, detail="视频不存在")
        
        # 删除视频文件
        if os.path.exists(video_info.get('file_path')):
            os.remove(video_info.get('file_path'))
        
        # 删除相关文件
        files_to_delete = [
            RESULTS_DIR / f"{video_id}_style_result.json",
            FEEDBACK_DIR / f"{video_id}_feedback.json",
            DATA_DIR / 'audio' / f"{video_id}.wav",
            DATA_DIR / 'text' / f"{video_id}.txt",
            DATA_DIR / 'features' / f"{video_id}_features.json"
        ]
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # 从数据库删除记录
        data_manager.delete_video(video_id)
        
        logger.info(f"视频删除成功: video_id={video_id}")
        
        return {
            "success": True,
            "message": "视频及相关数据已成功删除"
        }
        
    except HTTPException as e:
        logger.error(f"删除视频失败 - HTTP错误: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"删除视频失败 - 系统错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@app.get("/api/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    健康检查接口
    
    Returns:
        系统状态
    """
    try:
        return {
            "success": True,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": API_CONFIG['version']
        }
    except Exception:
        raise HTTPException(status_code=503, detail="服务不可用")


@app.get("/api/config", response_model=Dict[str, Any])
async def get_system_config() -> Dict[str, Any]:
    """
    获取系统配置信息
    
    Returns:
        系统配置
    """
    return {
        "success": True,
        "data": {
            "supported_disciplines": list(data_manager.discipline_standards.keys()),
            "supported_grades": list(data_manager.grade_standards.keys()),
            "style_labels": list(data_manager.style_labels.keys()),
            "max_video_size": API_CONFIG['max_video_size']
        }
    }


def start_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    启动API服务器
    
    Args:
        host: 主机地址
        port: 端口号
    """
    if uvicorn is None:
        logger.error("uvicorn未安装，无法启动服务器")
        return
        
    logger.info(f"启动教师风格画像分析系统API服务 - 地址: {host}:{port}")
    uvicorn.run(
        "api.api_handler:app",
        host=host,
        port=port,
        reload=API_CONFIG.get('debug', False)
    )


if __name__ == "__main__":
    start_server()