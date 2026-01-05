# VLM 标注系统 - API 配置问题排查

## 问题现状

经过测试，公司内部 Claude API (`https://aidev.deyecloud.com/api`) 目前无法识别标准的 Claude 模型名称。

### 测试结果

尝试的所有模型都返回错误:
```
Error: No available Claude accounts support the requested model
```

测试的模型列表:
- ❌ claude-3-5-sonnet-20241022
- ❌ claude-3-5-sonnet-20240620
- ❌ claude-3-sonnet-20240229
- ❌ claude-3-opus-20240229
- ❌ claude-3-haiku-20240307
- ❌ claude-2.1
- ❌ claude-2.0
- ❌ claude-instant-1.2

## 可能的原因

### 1. 公司内部使用自定义模型名称

公司的 API 中继服务可能使用了不同的模型命名规则，例如:
- `sonnet-3.5`
- `claude-sonnet`
- `default`
- 自定义的模型 ID

### 2. API 配置问题

- API 密钥权限不足
- 需要额外的认证头
- Base URL 不正确

### 3. API 中继服务问题

错误信息显示 "Relay service error"，说明这是一个代理/中继服务，可能:
- 后端 Claude 账号未配置
- 账号配额已用完
- 服务暂时不可用

## 解决方案

### 方案 1: 联系 IT 部门获取正确配置

**需要询问的问题:**

1. **正确的模型名称是什么?**
   ```
   公司内部 Claude API 支持哪些模型名称？
   例如: claude-3-5-sonnet, sonnet-3.5, 或其他？
   ```

2. **是否需要额外的配置?**
   ```
   - 是否需要特殊的认证头？
   - 是否有特定的请求格式？
   - 是否有使用文档？
   ```

3. **当前 API 状态**
   ```
   - API 服务是否正常运行？
   - 是否有账号配额限制？
   - 如何查看使用情况？
   ```

### 方案 2: 测试不同的模型名称

创建一个脚本尝试各种可能的模型名称:

```python
# 测试更多可能的名称
MODELS_TO_TEST = [
    "claude",
    "sonnet",
    "sonnet-3.5",
    "claude-3.5-sonnet",
    "claude-3-sonnet",
    "default",
    "gpt-4",  # 某些中继也支持 OpenAI
]
```

### 方案 3: 使用官方 Anthropic API (备选)

如果公司内部 API 暂时不可用，可以临时使用官方 API:

```bash
export ANTHROPIC_API_KEY="your-official-api-key"
export ANTHROPIC_BASE_URL=""  # 留空使用官方 API
```

### 方案 4: 查看 API 文档

如果公司内部有 API 文档或 Swagger/OpenAPI 规范:

```bash
# 尝试访问 API 文档
curl https://aidev.deyecloud.com/api/docs
curl https://aidev.deyecloud.com/api/swagger
curl https://aidev.deyecloud.com/api/v1/models  # 查看可用模型列表
```

## 临时解决方案 - 使用模拟标注器

在等待 API 配置期间，可以使用基于规则的标注器:

### 1. 修改代码使用规则映射

```python
# 在 annotate_tbu.py 中添加 --method 参数
parser.add_argument('--method', choices=['vlm', 'rule'],
                   default='vlm', help='标注方法')

if args.method == 'rule':
    # 使用规则映射
    from src.annotation.rule_mapper import RuleBasedAnnotator
    annotator = RuleBasedAnnotator()
else:
    # 使用 VLM
    annotator = VLMStyleAnnotator(...)
```

### 2. 实现规则映射器

```python
# src/annotation/rule_mapper.py
class RuleBasedAnnotator:
    """基于规则的标注器（不需要 API）"""

    def annotate_single_sample(self, behavior_sequence, behavior_durations, **kwargs):
        # 基于行为序列推断风格
        # （使用 RECOMMENDED_DATASETS.md 中的映射规则）
        pass
```

## 下一步行动

### 优先级 P0 - 立即执行

1. ✅ **确认 API 问题** - 已完成测试，确认所有标准模型名称不可用

2. ⏳ **联系 IT 或 API 提供者**
   ```
   收件人: IT部门 / API管理员
   主题: Claude API 模型配置咨询

   您好，

   我在使用公司内部 Claude API 时遇到问题，所有标准模型名称
   都返回 "No available Claude accounts support the requested model"。

   请问:
   1. 正确的模型名称是什么？
   2. 是否有 API 使用文档？
   3. 当前 API 服务状态如何？

   API 地址: https://aidev.deyecloud.com/api

   谢谢！
   ```

3. ⏳ **查找内部文档**
   - 搜索公司 Wiki / Confluence
   - 查看 Slack / 飞书聊天记录
   - 询问同事是否有人使用过

### 优先级 P1 - 短期方案

如果 API 在 1-2 天内无法解决:

1. **实现规则映射器** - 不依赖 API 的标注方法
2. **使用官方 Anthropic API** - 临时付费使用官方服务
3. **手动标注小批量数据** - 先标注 100-500 个样本用于训练

### 优先级 P2 - 长期方案

API 解决后:

1. **大规模标注** - 使用 VLM 标注 TBU 全量数据
2. **训练模型** - 基于高质量标注训练
3. **持续优化** - 根据模型效果调整标注策略

## 代码已完成部分

即使 API 暂时不可用，以下代码已经完成并可以使用:

### ✅ 完整的代码框架

1. **VLM 标注器** (`src/annotation/vlm_annotator.py`)
   - 完整的 API 调用逻辑
   - 错误处理和重试机制
   - 批量标注功能
   - 只需要正确的模型名称即可使用

2. **数据转换** (`src/annotation/convert_tbu.py`)
   - TBU 数据格式转换
   - 可以独立使用

3. **批量标注脚本** (`annotate_tbu.py`)
   - 命令行工具
   - 断点续传功能
   - 进度跟踪

4. **一键流程** (`annotate_and_train_tbu.sh`)
   - 自动化完整流程
   - 只需要修改模型名称

### ✅ 文档完整

- `VLM_ANNOTATION_GUIDE.md` - 完整使用指南
- `test_available_models.py` - 模型测试工具
- 本文档 - 问题排查指南

## 测试数据

如果你有几张教学视频的截图，我可以帮你:

1. 手动测试标注逻辑（不通过 API）
2. 演示规则映射方法
3. 验证数据转换流程

## 需要帮助？

告诉我:
1. 是否能联系到 IT 部门？
2. 是否有公司内部 API 文档？
3. 是否考虑使用官方 Anthropic API？
4. 是否需要我实现规则映射器作为备选方案？

---

**更新时间**: 2026-01-05
**状态**: 等待 API 配置确认
