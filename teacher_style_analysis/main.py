"""教师风格画像分析系统 - 主入口文件"""
import os
import sys
import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('teacher_style_analysis')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from teacher_style_analysis.config.config import PROJECT_ROOT, DATA_DIR, init_directories
from teacher_style_analysis.api.api_handler import start_server
from teacher_style_analysis.data.data_manager import data_manager
from teacher_style_analysis.features.feature_extractor import feature_extractor
from teacher_style_analysis.models.style_classifier import style_classifier
from teacher_style_analysis.feedback.feedback_generator import feedback_generator


def run_analysis_pipeline(video_path: str, teacher_id: str, 
                         discipline: str, grade: str) -> dict:
    """
    运行完整的风格分析流程
    
    Args:
        video_path: 视频文件路径
        teacher_id: 教师ID
        discipline: 学科类型
        grade: 年级水平
        
    Returns:
        分析结果
    """
    try:
        logger.info(f"开始分析流程: video={video_path}, teacher={teacher_id}")
        
        # 生成视频ID
        import uuid
        import os  # 确保os模块在函数内部可用
        video_id = str(uuid.uuid4())[:10]
        
        # 1. 保存视频信息
        logger.info("步骤1: 保存视频信息")
        video_info = {
            'video_id': video_id,
            'teacher_id': teacher_id,
            'discipline': discipline,
            'grade': grade,
            'filename': os.path.basename(video_path),
            'status': 'processing',
            'file_path': video_path
        }
        data_manager.save_video_info(video_info)
        
        # 2. 提取特征
        logger.info("步骤2: 提取多模态特征")
        features = feature_extractor.process_video(video_path)
        
        # 3. 风格分类
        logger.info("步骤3: 执行风格分类")
        # 添加调试信息
        logger.info(f"特征结构: {type(features)}, keys: {list(features.keys())}")
        logger.info(f"fused_features类型: {type(features.get('fused_features'))}")
        if features.get('fused_features'):
            logger.info(f"fused_features keys: {list(features['fused_features'].keys())}")
        
        # 使用完整的特征字典作为参数
        result = style_classifier.classify_style(None, features)
        
        # 保存特征和结果到文件系统
        logger.info("步骤3.5: 保存特征和结果到文件系统")
        import json
        import os
        import numpy as np
        
        # 创建目录（如果不存在）
        features_dir = os.path.join(PROJECT_ROOT, "data", "extracted_features")
        results_dir = os.path.join(PROJECT_ROOT, "data", "results")
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # 定义JSON序列化器，处理NumPy数组
        def convert_numpy_to_python(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(item) for item in obj]
            else:
                return obj
        
        # 转换特征数据
        serializable_features = convert_numpy_to_python(features)
        serializable_result = convert_numpy_to_python(result)
        
        # 保存特征
        features_file = os.path.join(features_dir, f"{video_id}_features.json")
        with open(features_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_features, f, ensure_ascii=False, indent=2)
        
        # 保存结果
        result_file = os.path.join(results_dir, f"{video_id}_style_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        # 4. 生成反馈报告
        logger.info("步骤4: 生成个性化反馈")
        feedback = feedback_generator.generate_feedback_report(
            video_id, discipline, grade
        )
        
        # 5. 更新状态
        data_manager.update_video_status(video_id, "completed")
        
        logger.info(f"分析流程完成: video_id={video_id}")
        
        # 返回结果
        return {
            'video_id': video_id,
            'status': 'completed',
            'feedback': feedback,
            'features': features,
            'result': result
        }
        
    except Exception as e:
        logger.error(f"分析流程失败: {str(e)}")
        raise


def batch_analysis(directory_path: str, teacher_id: str, 
                  discipline: str, grade: str) -> list:
    """
    批量分析指定目录下的所有视频文件
    
    Args:
        directory_path: 视频目录路径
        teacher_id: 教师ID
        discipline: 学科类型
        grade: 年级水平
        
    Returns:
        分析结果列表
    """
    results = []
    supported_formats = ['.mp4', '.avi', '.mov', '.wmv', '.flv']
    
    try:
        # 获取目录下所有视频文件
        video_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_formats):
                    video_files.append(os.path.join(root, file))
        
        logger.info(f"发现 {len(video_files)} 个视频文件待分析")
        
        # 逐个分析视频
        for i, video_path in enumerate(video_files, 1):
            try:
                logger.info(f"开始分析第 {i}/{len(video_files)} 个视频: {video_path}")
                result = run_analysis_pipeline(video_path, teacher_id, discipline, grade)
                results.append({
                    'video_path': video_path,
                    'success': True,
                    'data': result
                })
            except Exception as e:
                logger.error(f"分析视频 {video_path} 失败: {str(e)}")
                results.append({
                    'video_path': video_path,
                    'success': False,
                    'error': str(e)
                })
        
        # 统计结果
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"批量分析完成 - 成功: {success_count}, 失败: {len(results) - success_count}")
        
        return results
        
    except Exception as e:
        logger.error(f"批量分析过程失败: {str(e)}")
        raise


def export_results(video_id: str, output_format: str = 'json') -> str:
    """
    导出分析结果
    
    Args:
        video_id: 视频ID
        output_format: 输出格式 (json, csv, excel)
        
    Returns:
        导出文件路径
    """
    try:
        from teacher_style_analysis.config.config import RESULTS_DIR, FEEDBACK_DIR
        import json
        import pandas as pd
        from datetime import datetime
        
        # 获取分析结果
        result_file = FEEDBACK_DIR / f"{video_id}_feedback.json"
        if not result_file.exists():
            raise FileNotFoundError(f"分析结果不存在: {result_file}")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            feedback = json.load(f)
        
        # 根据格式导出
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = PROJECT_ROOT / 'exports'
        export_dir.mkdir(exist_ok=True)
        
        if output_format == 'json':
            # 导出完整的JSON
            output_path = export_dir / f"{video_id}_result_{timestamp}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(feedback, f, ensure_ascii=False, indent=2)
        
        elif output_format in ['csv', 'excel']:
            # 提取关键数据
            data = {
                'VideoID': [video_id],
                'TeacherID': [feedback.get('teacher_id', '')],
                'Discipline': [feedback.get('discipline', '')],
                'Grade': [feedback.get('grade', '')],
                'SMI': [feedback.get('smi', {}).get('score', 0)],
                'MainStyle': [feedback.get('teaching_style', {}).get('main_styles', [[None, None]])[0][0]],
                'MainStyleScore': [feedback.get('teaching_style', {}).get('main_styles', [[None, None]])[0][1]]
            }
            
            # 添加所有风格分数
            for style, score in feedback.get('teaching_style', {}).get('detailed_scores', {}).items():
                data[style] = [score]
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 导出
            if output_format == 'csv':
                output_path = export_dir / f"{video_id}_result_{timestamp}.csv"
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
            else:  # excel
                output_path = export_dir / f"{video_id}_result_{timestamp}.xlsx"
                df.to_excel(output_path, index=False)
        
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
        
        logger.info(f"结果已导出到: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"导出结果失败: {str(e)}")
        raise


def setup_database() -> None:
    """
    设置数据库（文件系统存储）
    """
    try:
        logger.info("设置数据存储系统")
        # 数据管理器已经自动初始化，无需额外操作
        logger.info("数据存储系统设置完成")
    except Exception as e:
        logger.error(f"数据存储系统设置失败: {str(e)}")
        raise


def check_system_status() -> dict:
    """
    检查系统状态
    
    Returns:
        系统状态信息
    """
    try:
        # 检查目录结构
        directories = [DATA_DIR, PROJECT_ROOT / 'exports']
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
        
        # 检查依赖
        from importlib import import_module
        required_packages = [
            'numpy', 'pandas', 'matplotlib', 'seaborn',
            'opencv-python', 'torch', 'transformers',
            'fastapi', 'uvicorn', 'scikit-learn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        # 返回状态
        return {
            'status': 'healthy' if not missing_packages else 'warning',
            'missing_packages': missing_packages,
            'directories': {
                'data': str(DATA_DIR),
                'exports': str(PROJECT_ROOT / 'exports')
            }
        }
        
    except Exception as e:
        logger.error(f"系统状态检查失败: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }


def main() -> None:
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='教师风格画像分析系统')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 启动服务器命令
    server_parser = subparsers.add_parser('server', help='启动API服务器')
    server_parser.add_argument('--host', type=str, default='0.0.0.0', help='主机地址')
    server_parser.add_argument('--port', type=int, default=8000, help='端口号')
    
    # 分析视频命令
    analyze_parser = subparsers.add_parser('analyze', help='分析单个视频')
    analyze_parser.add_argument('--video', type=str, required=True, help='视频文件路径')
    analyze_parser.add_argument('--teacher', type=str, required=True, help='教师ID')
    analyze_parser.add_argument('--discipline', type=str, required=True, help='学科类型')
    analyze_parser.add_argument('--grade', type=str, required=True, help='年级水平')
    
    # 批量分析命令
    batch_parser = subparsers.add_parser('batch', help='批量分析视频')
    batch_parser.add_argument('--dir', type=str, required=True, help='视频目录路径')
    batch_parser.add_argument('--teacher', type=str, required=True, help='教师ID')
    batch_parser.add_argument('--discipline', type=str, required=True, help='学科类型')
    batch_parser.add_argument('--grade', type=str, required=True, help='年级水平')
    
    # 导出结果命令
    export_parser = subparsers.add_parser('export', help='导出分析结果')
    export_parser.add_argument('--video_id', type=str, required=True, help='视频ID')
    export_parser.add_argument('--format', type=str, choices=['json', 'csv', 'excel'], default='json', help='输出格式')
    
    # 设置数据库命令
    setup_parser = subparsers.add_parser('setup', help='设置系统')
    setup_parser.add_argument('--init-db', action='store_true', help='初始化数据库')
    
    # 系统状态命令
    status_parser = subparsers.add_parser('status', help='检查系统状态')
    
    args = parser.parse_args()
    
    # 初始化目录
    init_directories()
    
    if args.command == 'server':
        # 启动API服务器
        print(f"启动教师风格画像分析系统API服务...")
        print(f"服务地址: http://{args.host}:{args.port}")
        print(f"API文档: http://{args.host}:{args.port}/docs")
        start_server(host=args.host, port=args.port)
    
    elif args.command == 'analyze':
        # 分析单个视频
        print(f"开始分析视频: {args.video}")
        print(f"教师ID: {args.teacher}, 学科: {args.discipline}, 年级: {args.grade}")
        
        result = run_analysis_pipeline(args.video, args.teacher, args.discipline, args.grade)
        
        print("\n分析完成!")
        print(f"视频ID: {result['video_id']}")
        print(f"SMI分数: {result['feedback']['smi']['score']}")
        print(f"主要风格: {result['feedback']['teaching_style']['main_styles'][0][0]}")
    
    elif args.command == 'batch':
        # 批量分析
        print(f"开始批量分析视频目录: {args.dir}")
        print(f"教师ID: {args.teacher}, 学科: {args.discipline}, 年级: {args.grade}")
        
        results = batch_analysis(args.dir, args.teacher, args.discipline, args.grade)
        
        print("\n批量分析完成!")
        print(f"总视频数: {len(results)}")
        print(f"成功: {sum(1 for r in results if r['success'])}")
        print(f"失败: {sum(1 for r in results if not r['success'])}")
    
    elif args.command == 'export':
        # 导出结果
        print(f"导出视频ID {args.video_id} 的分析结果...")
        
        output_path = export_results(args.video_id, args.format)
        
        print(f"结果已导出到: {output_path}")
    
    elif args.command == 'setup':
        # 设置系统
        print("开始设置系统...")
        
        # 初始化目录
        print("初始化目录结构...")
        init_directories()
        
        # 初始化数据库
        if args.init_db:
            setup_database()
        
        print("系统设置完成!")
    
    elif args.command == 'status':
        # 检查系统状态
        print("检查系统状态...")
        
        status = check_system_status()
        
        print(f"系统状态: {status['status']}")
        print(f"数据目录: {status['directories']['data']}")
        print(f"导出目录: {status['directories']['exports']}")
        
        if status['missing_packages']:
            print(f"\n缺少的依赖包: {', '.join(status['missing_packages'])}")
            print("建议运行: pip install " + ' '.join(status['missing_packages']))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()