#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
教师风格分析系统测试运行器
用于运行所有单元测试和集成测试
"""

import unittest
import sys
import os
import argparse
from datetime import datetime

# 导入logger
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.config import logger

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 测试文件列表
TEST_FILES = [
    'test_feature_extractor.py',
    'test_style_classifier.py',
    'test_feedback_generator.py',
    'test_api_handler.py',
    'test_main.py',
    'test_integration.py'
]


def run_all_tests(verbosity=2):
    """
    运行所有测试文件
    
    Args:
        verbosity: 测试输出详细程度 (1=简要, 2=详细)
    
    Returns:
        unittest.TextTestResult: 测试结果对象
    """
    logger.info("=" * 70)
    logger.info(f"开始运行教师风格分析系统所有测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # 创建测试加载器和测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 加载每个测试文件
    for test_file in TEST_FILES:
        try:
            # 从文件名导入模块
            module_name = f"tests.{test_file.replace('.py', '')}"
            module = __import__(module_name, fromlist=['*'])
            
            # 加载测试用例
            test_cases = loader.loadTestsFromModule(module)
            suite.addTests(test_cases)
            logger.info(f"✓ 成功加载测试文件: {test_file}")
        except Exception as e:
            logger.error(f"✗ 加载测试文件失败: {test_file}")
            logger.error(f"  错误信息: {str(e)}")
    
    logger.info("=" * 70)
    logger.info(f"总共加载了 {suite.countTestCases()} 个测试用例")
    logger.info("=" * 70)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # 打印测试摘要
    logger.info("\n" + "=" * 70)
    logger.info("测试运行摘要")
    logger.info("=" * 70)
    logger.info(f"运行测试总数: {result.testsRun}")
    logger.info(f"失败测试数: {len(result.failures)}")
    logger.info(f"错误测试数: {len(result.errors)}")
    logger.info(f"跳过测试数: {len(result.skipped)}")
    
    if result.wasSuccessful():
        logger.info("\n✅ 所有测试通过!")
    else:
        logger.error("\n❌ 测试运行失败!")
    
    logger.info("=" * 70)
    return result


def run_specific_test(test_file, verbosity=2):
    """
    运行特定的测试文件
    
    Args:
        test_file: 要运行的测试文件名
        verbosity: 测试输出详细程度
    
    Returns:
        unittest.TextTestResult: 测试结果对象
    """
    logger.info("=" * 70)
    logger.info(f"开始运行测试文件: {test_file} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    try:
        # 从文件名导入模块
        module_name = f"tests.{test_file.replace('.py', '')}"
        module = __import__(module_name, fromlist=['*'])
        
        # 加载并运行测试
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        logger.info(f"✓ 成功加载测试文件: {test_file}")
        logger.info(f"总共加载了 {suite.countTestCases()} 个测试用例")
        logger.info("=" * 70)
        
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        # 打印测试摘要
        logger.info("\n" + "=" * 70)
        logger.info(f"测试文件 {test_file} 运行摘要")
        logger.info("=" * 70)
        logger.info(f"运行测试总数: {result.testsRun}")
        logger.info(f"失败测试数: {len(result.failures)}")
        logger.info(f"错误测试数: {len(result.errors)}")
        logger.info(f"跳过测试数: {len(result.skipped)}")
        
        if result.wasSuccessful():
            logger.info("\n✅ 测试通过!")
        else:
            logger.error("\n❌ 测试失败!")
        
        logger.info("=" * 70)
        return result
    
    except Exception as e:
        logger.error(f"✗ 运行测试文件失败: {test_file}")
        logger.error(f"  错误信息: {str(e)}")
        # 返回失败的结果对象
        class FailedResult(unittest.TestResult):
            def __init__(self):
                super().__init__()
                self.testsRun = 0
                self.failures = []
                self.errors = [(None, str(e))]
                self.skipped = []
            
            def wasSuccessful(self):
                return False
        
        return FailedResult()


def generate_test_report(result, output_file=None):
    """
    生成测试报告
    
    Args:
        result: 测试结果对象
        output_file: 输出文件路径，如果为None则仅打印到控制台
    """
    report_lines = []
    report_lines.append("# 教师风格分析系统测试报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("## 测试概览")
    report_lines.append(f"- 运行测试总数: {result.testsRun}")
    report_lines.append(f"- 失败测试数: {len(result.failures)}")
    report_lines.append(f"- 错误测试数: {len(result.errors)}")
    report_lines.append(f"- 跳过测试数: {len(result.skipped)}")
    report_lines.append(f"- 测试状态: {'通过' if result.wasSuccessful() else '失败'}")
    report_lines.append("")
    
    # 添加失败的测试详情
    if result.failures:
        report_lines.append("## 失败测试详情")
        for i, (test, error) in enumerate(result.failures, 1):
            report_lines.append(f"### {i}. {test.id()}")
            report_lines.append("```")
            report_lines.append(error)
            report_lines.append("```")
            report_lines.append("")
    
    # 添加错误的测试详情
    if result.errors:
        report_lines.append("## 错误测试详情")
        for i, (test, error) in enumerate(result.errors, 1):
            report_lines.append(f"### {i}. {test.id() if test else '未知测试'}")
            report_lines.append("```")
            report_lines.append(error)
            report_lines.append("```")
            report_lines.append("")
    
    report = '\n'.join(report_lines)
    
    # 输出报告
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"测试报告已保存至: {output_file}")
    else:
        logger.info(report)


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行教师风格分析系统测试')
    parser.add_argument('-f', '--file', help='指定要运行的测试文件')
    parser.add_argument('-v', '--verbosity', type=int, default=2, choices=[1, 2],
                        help='测试输出详细程度 (1=简要, 2=详细)')
    parser.add_argument('-r', '--report', help='生成测试报告文件路径')
    args = parser.parse_args()
    
    # 运行测试
    if args.file:
        # 验证文件是否存在
        if args.file in TEST_FILES:
            result = run_specific_test(args.file, args.verbosity)
        else:
            logger.error(f"错误: 测试文件 '{args.file}' 不存在")
            logger.error(f"可用的测试文件: {', '.join(TEST_FILES)}")
            sys.exit(1)
    else:
        # 运行所有测试
        result = run_all_tests(args.verbosity)
    
    # 生成测试报告
    if args.report:
        generate_test_report(result, args.report)
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)