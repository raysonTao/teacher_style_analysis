"""个性化反馈模块，实现风格匹配度指数(SMI)计算和改进建议生成"""
import os
import json
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    SYSTEM_CONFIG, STYLE_LABELS, 
    RESULTS_DIR, FEEDBACK_DIR
)
from models.core.style_classifier import style_classifier


class FeedbackGenerator:
    """个性化反馈生成器"""
    
    def __init__(self):
        self._init_reference_standards()
        self._init_improvement_rules()
    
    def _init_reference_standards(self):
        """初始化参考标准"""
        # 学科类型对应的最佳风格权重配置
        self.discipline_standards = {
            '数学': {
                '理论讲授型': 0.3,
                '启发引导型': 0.25,
                '互动导向型': 0.15,
                '逻辑推导型': 0.4,
                '题目驱动型': 0.35,
                '情感表达型': 0.1,
                '耐心细致型': 0.2
            },
            '语文': {
                '理论讲授型': 0.35,
                '启发引导型': 0.3,
                '互动导向型': 0.25,
                '逻辑推导型': 0.2,
                '题目驱动型': 0.15,
                '情感表达型': 0.4,
                '耐心细致型': 0.25
            },
            '英语': {
                '理论讲授型': 0.2,
                '启发引导型': 0.25,
                '互动导向型': 0.4,
                '逻辑推导型': 0.2,
                '题目驱动型': 0.3,
                '情感表达型': 0.3,
                '耐心细致型': 0.2
            },
            '物理': {
                '理论讲授型': 0.35,
                '启发引导型': 0.25,
                '互动导向型': 0.2,
                '逻辑推导型': 0.4,
                '题目驱动型': 0.35,
                '情感表达型': 0.15,
                '耐心细致型': 0.2
            },
            '化学': {
                '理论讲授型': 0.3,
                '启发引导型': 0.25,
                '互动导向型': 0.25,
                '逻辑推导型': 0.35,
                '题目驱动型': 0.3,
                '情感表达型': 0.15,
                '耐心细致型': 0.2
            },
            '生物': {
                '理论讲授型': 0.3,
                '启发引导型': 0.3,
                '互动导向型': 0.25,
                '逻辑推导型': 0.3,
                '题目驱动型': 0.25,
                '情感表达型': 0.2,
                '耐心细致型': 0.25
            },
            '历史': {
                '理论讲授型': 0.4,
                '启发引导型': 0.25,
                '互动导向型': 0.2,
                '逻辑推导型': 0.3,
                '题目驱动型': 0.15,
                '情感表达型': 0.35,
                '耐心细致型': 0.2
            },
            '地理': {
                '理论讲授型': 0.35,
                '启发引导型': 0.3,
                '互动导向型': 0.25,
                '逻辑推导型': 0.3,
                '题目驱动型': 0.2,
                '情感表达型': 0.25,
                '耐心细致型': 0.2
            },
            '政治': {
                '理论讲授型': 0.35,
                '启发引导型': 0.3,
                '互动导向型': 0.25,
                '逻辑推导型': 0.35,
                '题目驱动型': 0.2,
                '情感表达型': 0.3,
                '耐心细致型': 0.2
            }
        }
        
        # 年级水平对应的最佳风格权重配置
        self.grade_standards = {
            '初中': {
                '理论讲授型': 0.25,
                '启发引导型': 0.35,
                '互动导向型': 0.3,
                '逻辑推导型': 0.25,
                '题目驱动型': 0.25,
                '情感表达型': 0.35,
                '耐心细致型': 0.3
            },
            '高中': {
                '理论讲授型': 0.35,
                '启发引导型': 0.25,
                '互动导向型': 0.2,
                '逻辑推导型': 0.4,
                '题目驱动型': 0.35,
                '情感表达型': 0.2,
                '耐心细致型': 0.2
            },
            '大学': {
                '理论讲授型': 0.4,
                '启发引导型': 0.2,
                '互动导向型': 0.2,
                '逻辑推导型': 0.45,
                '题目驱动型': 0.3,
                '情感表达型': 0.15,
                '耐心细致型': 0.15
            }
        }
    
    def _init_improvement_rules(self):
        """初始化改进建议规则"""
        self.improvement_rules = {
            '理论讲授型': {
                'low': [
                    "建议增加理论知识点的系统性讲解，可采用概念图或知识树的方式组织内容",
                    "在讲解重要概念时，可适当放慢语速，确保学生充分理解",
                    "增加理论与实践的联系，通过具体案例加深学生对理论知识的理解"
                ],
                'high': [
                    "理论讲解非常出色，可考虑进一步丰富内容深度",
                    "建议在保持高质量理论讲解的同时，适度增加与学生的互动环节"
                ]
            },
            '启发引导型': {
                'low': [
                    "建议增加提问频率，通过问题引导学生思考",
                    "可尝试使用苏格拉底式教学法，通过连续提问引导学生自主探索",
                    "在学生回答问题后，给予更有针对性的反馈和引导"
                ],
                'high': [
                    "启发引导能力强，可考虑尝试更具挑战性的开放性问题",
                    "建议结合小组讨论，扩大启发式教学的影响范围"
                ]
            },
            '互动导向型': {
                'low': [
                    "建议增加课堂互动环节，如小组讨论、问答环节等",
                    "可使用投票、快速测验等方式提高学生参与度",
                    "注意观察学生反应，根据学生反馈及时调整教学节奏"
                ],
                'high': [
                    "互动性表现优秀，可考虑引入更多元化的互动形式",
                    "建议在互动中注重引导，避免为互动而互动"
                ]
            },
            '逻辑推导型': {
                'low': [
                    "建议强化逻辑推理过程的展示，清晰呈现思路脉络",
                    "使用连接词和过渡句增强内容的逻辑性和连贯性",
                    "可将复杂问题分解为多个简单步骤，逐步推导"
                ],
                'high': [
                    "逻辑推导能力出色，可考虑适当引入更具挑战性的逻辑推理内容",
                    "建议在保持严密逻辑的同时，注重表达的生动性"
                ]
            },
            '题目驱动型': {
                'low': [
                    "建议增加典型例题讲解，通过题目帮助学生理解知识点",
                    "注重解题思路和方法的传授，而非单纯讲解答案",
                    "可设计梯度化的题目，满足不同层次学生的需求"
                ],
                'high': [
                    "例题讲解非常出色，可考虑引入更多变式训练",
                    "建议总结解题规律和技巧，帮助学生形成解题方法论"
                ]
            },
            '情感表达型': {
                'low': [
                    "建议增强情感投入，通过语调变化和表情丰富课堂氛围",
                    "尝试将学科知识与情感、价值观教育相结合",
                    "多给予学生积极的情感反馈和鼓励"
                ],
                'high': [
                    "情感表达感染力强，可考虑如何将这种优势更好地服务于教学目标",
                    "建议保持情感表达的适度性，避免过度情感化影响教学效果"
                ]
            },
            '耐心细致型': {
                'low': [
                    "建议放慢教学节奏，给予学生充分的思考和消化时间",
                    "注意观察学生反应，及时调整教学难度和进度",
                    "对学生的问题给予详细、耐心的解答"
                ],
                'high': [
                    "教学耐心细致，可考虑如何在保持细致的同时提高教学效率",
                    "建议设计分层教学内容，兼顾不同学习速度的学生"
                ]
            },
            'interaction_level': {
                'low': [
                    "建议增加师生互动频率，可设计小组活动或问答环节",
                    "使用眼神交流和肢体语言鼓励学生参与",
                    "尝试使用教学工具增加互动性"
                ]
            },
            'explanation_clarity': {
                'low': [
                    "建议使用更简洁明了的语言进行讲解",
                    "采用图表、示例等可视化手段辅助讲解",
                    "关键概念可重复强调，确保学生理解"
                ]
            },
            'speech_rate': {
                'too_fast': [
                    "语速偏快，建议适当放慢，特别是讲解复杂概念时",
                    "在重要知识点后适当停顿，给学生思考时间"
                ],
                'too_slow': [
                    "语速偏慢，可适当加快以提高教学效率",
                    "保持语速变化，在关键内容时放缓"
                ]
            }
        }
    
    def calculate_smi(self, style_scores: Dict, discipline: str, grade: str) -> Tuple[float, Dict]:
        """
        计算风格匹配度指数(SMI)
        
        Args:
            style_scores: 教师风格评分
            discipline: 学科类型
            grade: 年级水平
            
        Returns:
            SMI分数和各维度贡献
        """
        # 获取对应学科和年级的标准
        discipline_std = self.discipline_standards.get(discipline, self.discipline_standards['数学'])
        grade_std = self.grade_standards.get(grade, self.grade_standards['高中'])
        
        # 计算加权标准向量
        combined_std = {}
        for style in style_scores.keys():
            # 学科权重占60%，年级权重占40%
            combined_std[style] = 0.6 * discipline_std.get(style, 0) + 0.4 * grade_std.get(style, 0)
        
        # 归一化标准向量
        std_sum = sum(combined_std.values())
        if std_sum > 0:
            for style in combined_std:
                combined_std[style] /= std_sum
        
        # 计算SMI分数（余弦相似度）
        score_values = np.array(list(style_scores.values()))
        std_values = np.array(list(combined_std.values()))
        
        # 归一化教师风格向量
        score_sum = sum(score_values)
        if score_sum > 0:
            normalized_scores = score_values / score_sum
        else:
            normalized_scores = score_values
        
        # 计算余弦相似度
        dot_product = np.dot(normalized_scores, std_values)
        norm_scores = np.linalg.norm(normalized_scores)
        norm_std = np.linalg.norm(std_values)
        
        if norm_scores * norm_std > 0:
            similarity = dot_product / (norm_scores * norm_std)
        else:
            similarity = 0
        
        # 转换为0-100的SMI分数
        smi_score = (similarity + 1) / 2 * 100
        
        # 计算各维度的贡献度
        dimension_contributions = {}
        for style in style_scores.keys():
            # 计算风格匹配度贡献
            contribution = abs(style_scores[style] - combined_std[style])
            dimension_contributions[style] = {
                'teacher_score': style_scores[style],
                'ideal_score': combined_std[style],
                'difference': style_scores[style] - combined_std[style],
                'contribution': 1.0 - min(1.0, contribution * 2)  # 差异越小，贡献越大
            }
        
        return smi_score, dimension_contributions
    
    def generate_improvement_suggestions(self, style_scores: Dict, 
                                        dimension_contributions: Dict, 
                                        feature_contributions: Dict) -> List[str]:
        """
        生成个性化改进建议
        
        Args:
            style_scores: 教师风格评分
            dimension_contributions: 各维度贡献分析
            feature_contributions: 特征贡献度分析
            
        Returns:
            改进建议列表
        """
        suggestions = []
        
        # 1. 根据风格分数生成建议
        sorted_styles = sorted(style_scores.items(), key=lambda x: x[1])
        
        # 为得分最低的2-3个风格生成建议
        for style, score in sorted_styles[:min(3, len(sorted_styles))]:
            if score < 0.4:  # 分数较低
                if style in self.improvement_rules:
                    low_suggestions = self.improvement_rules[style].get('low', [])
                    suggestions.extend(low_suggestions[:1])  # 每个风格选1个建议
        
        # 2. 根据维度贡献生成建议
        for style, contribution_info in dimension_contributions.items():
            difference = contribution_info['difference']
            # 如果与理想值相差过大
            if abs(difference) > 0.3:
                if difference < 0 and style in self.improvement_rules:
                    # 低于理想值
                    low_suggestions = self.improvement_rules[style].get('low', [])
                    if low_suggestions and low_suggestions[0] not in suggestions:
                        suggestions.append(low_suggestions[0])
        
        # 3. 根据特征贡献生成建议
        # 找出主要特征问题
        main_features = []
        for style, contributions in feature_contributions.items():
            for contribution in contributions:
                if contribution['contribution_score'] < 0.3:
                    main_features.append(contribution['feature_name'])
        
        # 去重并排序
        main_features = list(set(main_features))
        
        # 生成特征相关建议
        for feature in main_features[:2]:  # 最多处理2个特征问题
            if feature == '互动水平' and 'interaction_level' in self.improvement_rules:
                suggestions.extend(self.improvement_rules['interaction_level'].get('low', []))
            elif feature == '讲解清晰度' and 'explanation_clarity' in self.improvement_rules:
                suggestions.extend(self.improvement_rules['explanation_clarity'].get('low', []))
            elif feature == '语速':
                # 假设语速过快（实际应用中应该从audio特征中获取具体值）
                suggestions.extend(self.improvement_rules['speech_rate'].get('too_fast', []))
        
        # 4. 基于教师优势的建议
        top_style = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        if top_style in self.improvement_rules:
            high_suggestions = self.improvement_rules[top_style].get('high', [])
            if high_suggestions:
                suggestions.append(high_suggestions[0])
        
        # 去重并限制数量
        suggestions = list(dict.fromkeys(suggestions))  # 保持顺序去重
        return suggestions[:5]  # 最多5条建议
    
    def analyze_teaching_growth(self, teacher_id: str) -> Dict:
        """
        分析教师教学成长轨迹
        
        Args:
            teacher_id: 教师ID
            
        Returns:
            成长分析结果
        """
        growth_analysis = {
            'trend': 'stable',  # stable, improving, declining
            'smi_trend': [],
            'style_evolution': {},
            'key_changes': []
        }
        
        # 模拟成长数据（实际应用中应该从数据库查询历史记录）
        # 这里生成模拟的历史数据
        months = ['2024-05', '2024-06', '2024-07', '2024-08', '2024-09', '2024-10']
        base_smi = 75
        
        smi_values = []
        for i, month in enumerate(months):
            # 添加一些随机波动和上升趋势
            noise = np.random.normal(0, 3)
            trend = i * 0.5  # 缓慢上升趋势
            smi_value = base_smi + trend + noise
            smi_value = min(100, max(0, smi_value))  # 确保在0-100范围内
            smi_values.append({
                'date': month,
                'smi': round(smi_value, 2)
            })
        
        growth_analysis['smi_trend'] = smi_values
        
        # 分析趋势
        if len(smi_values) > 1:
            recent_avg = sum(s['smi'] for s in smi_values[-3:]) / 3
            previous_avg = sum(s['smi'] for s in smi_values[:-3]) / len(smi_values[:-3]) if len(smi_values) > 3 else 0
            
            if recent_avg > previous_avg + 2:
                growth_analysis['trend'] = 'improving'
                growth_analysis['key_changes'].append(f"近期SMI指数提高了约{recent_avg - previous_avg:.1f}分")
            elif recent_avg < previous_avg - 2:
                growth_analysis['trend'] = 'declining'
                growth_analysis['key_changes'].append(f"近期SMI指数下降了约{previous_avg - recent_avg:.1f}分")
        
        # 模拟风格演变
        growth_analysis['style_evolution'] = {
            '互动导向型': {'change': 0.15, 'direction': 'up'},
            '逻辑推导型': {'change': 0.08, 'direction': 'up'},
            '情感表达型': {'change': 0.05, 'direction': 'down'}
        }
        
        return growth_analysis
    
    def generate_feedback_report(self, video_id: str, discipline: str, grade: str) -> Dict:
        """
        生成完整的反馈报告
        
        Args:
            video_id: 视频ID
            discipline: 学科类型
            grade: 年级水平
            
        Returns:
            反馈报告
        """
        print(f"为视频 {video_id} 生成反馈报告，学科: {discipline}, 年级: {grade}")
        
        # 读取风格分析结果
        result_file = RESULTS_DIR / f"{video_id}_style_result.json"
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                style_result = json.load(f)
        except Exception as e:
            print(f"读取风格分析结果失败: {e}")
            # 如果没有结果，先进行风格分类
            style_classifier.classify_and_save(video_id)
            with open(result_file, 'r', encoding='utf-8') as f:
                style_result = json.load(f)
        
        # 获取风格分数
        style_scores = style_result['style_scores']
        feature_contributions = style_result['feature_contributions']
        
        # 计算SMI
        smi_score, dimension_contributions = self.calculate_smi(style_scores, discipline, grade)
        
        # 生成改进建议
        improvement_suggestions = self.generate_improvement_suggestions(
            style_scores, dimension_contributions, feature_contributions
        )
        
        # 分析教学成长（假设教师ID为视频ID的前8位）
        teacher_id = video_id[:8] if len(video_id) > 8 else video_id
        growth_analysis = self.analyze_teaching_growth(teacher_id)
        
        # 生成综合评价
        comprehensive_evaluation = self._generate_comprehensive_evaluation(
            style_scores, smi_score, growth_analysis['trend']
        )
        
        # 构建反馈报告
        feedback_report = {
            'video_id': video_id,
            'teacher_id': teacher_id,
            'discipline': discipline,
            'grade': grade,
            'timestamp': datetime.datetime.now().isoformat(),
            'smi': {
                'score': round(smi_score, 2),
                'level': self._get_smi_level(smi_score),
                'dimension_contributions': dimension_contributions
            },
            'teaching_style': {
                'main_styles': style_result['top_styles'],
                'detailed_scores': style_scores
            },
            'improvement_suggestions': improvement_suggestions,
            'comprehensive_evaluation': comprehensive_evaluation,
            'growth_analysis': growth_analysis,
            'confidence': style_result.get('confidence', 0.0)
        }
        
        # 保存反馈报告
        feedback_file = FEEDBACK_DIR / f"{video_id}_feedback.json"
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_report, f, ensure_ascii=False, indent=2)
        
        print(f"反馈报告已保存到: {feedback_file}")
        return feedback_report
    
    def _get_smi_level(self, smi_score: float) -> str:
        """
        根据SMI分数获取级别
        
        Args:
            smi_score: SMI分数
            
        Returns:
            级别描述
        """
        if smi_score >= 90:
            return "优秀匹配"
        elif smi_score >= 80:
            return "良好匹配"
        elif smi_score >= 70:
            return "基本匹配"
        elif smi_score >= 60:
            return "一般匹配"
        else:
            return "需要改进"
    
    def _generate_comprehensive_evaluation(self, style_scores: Dict, 
                                         smi_score: float, 
                                         growth_trend: str) -> str:
        """
        生成综合评价
        
        Args:
            style_scores: 风格评分
            smi_score: SMI分数
            growth_trend: 成长趋势
            
        Returns:
            综合评价文本
        """
        top_style = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)[0]
        
        evaluation = f"""
        综合评价：
        1. 您的教学风格以「{top_style[0]}」为主（得分：{top_style[1]:.2f}），这是一种{self._get_style_description(top_style[0])}的教学方式。
        2. 您的风格匹配度指数(SMI)为{round(smi_score, 2)}，属于{self._get_smi_level(smi_score)}水平。
        """
        
        # 添加趋势评价
        if growth_trend == 'improving':
            evaluation += "3. 通过历史数据分析，您的教学风格正在持续优化，特别是在互动性和逻辑性方面有明显进步。"
        elif growth_trend == 'declining':
            evaluation += "3. 通过历史数据分析，您的教学效果近期略有波动，建议关注改进建议中的重点方向。"
        else:
            evaluation += "3. 您的教学风格表现稳定，建议在保持现有优势的同时，关注可提升空间。"
        
        return evaluation.strip()
    
    def _get_style_description(self, style: str) -> str:
        """
        获取风格描述
        
        Args:
            style: 风格名称
            
        Returns:
            风格描述
        """
        descriptions = {
            '理论讲授型': '注重系统知识传授和理论讲解',
            '启发引导型': '通过问题引导学生自主思考和探索',
            '互动导向型': '强调师生互动和课堂参与',
            '逻辑推导型': '注重逻辑推理过程和思维训练',
            '题目驱动型': '通过例题讲解帮助学生理解和应用',
            '情感表达型': '教学过程中情感丰富，富有感染力',
            '耐心细致型': '教学节奏适中，注重细节和学生接受度'
        }
        return descriptions.get(style, '有特色')


# 创建反馈生成器实例
feedback_generator = FeedbackGenerator()