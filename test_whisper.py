"""
测试Whisper语音识别功能
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.features.audio_feature_extractor import AudioFeatureExtractor
from src.config.config import logger

print("="*60)
print("Whisper语音识别功能测试")
print("="*60)

# 初始化音频特征提取器
print("\n1. 初始化AudioFeatureExtractor...")
extractor = AudioFeatureExtractor()

# 检查Whisper模型状态
print(f"\n2. 检查Whisper模型状态...")
if extractor.whisper_model is not None:
    print(f"   ✓ Whisper模型已加载")
    print(f"   模型类型: {type(extractor.whisper_model)}")
    print(f"   模型设备: {extractor.whisper_model.device}")
else:
    print(f"   ✗ Whisper模型未加载")
    sys.exit(1)

# 测试音频文件
test_audio = "data/custom/videos/高中语文优质课《声声慢》（含课件教案）.mp4"

if not os.path.exists(test_audio):
    print(f"\n✗ 测试文件不存在: {test_audio}")
    print("请将视频文件放到data/custom/videos/目录")
    sys.exit(1)

print(f"\n3. 测试音频文件: {os.path.basename(test_audio)}")

# 首先从视频中提取音频
print(f"\n4. 从视频中提取音频...")
try:
    temp_audio = extractor.extract_audio_from_video(test_audio)
    if temp_audio and os.path.exists(temp_audio):
        print(f"   ✓ 音频提取成功: {temp_audio}")
        audio_size = os.path.getsize(temp_audio) / (1024*1024)
        print(f"   音频文件大小: {audio_size:.2f} MB")
    else:
        print(f"   ✗ 音频提取失败")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ 音频提取异常: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 提取音频特征（包括语音识别）
print(f"\n5. 提取音频特征（包括Whisper语音识别）...")
print("   这可能需要1-2分钟，请耐心等待...")

try:
    features = extractor.extract_features(temp_audio)

    print(f"\n✓ 音频特征提取完成！")
    print(f"\n特征统计:")
    print(f"  - 音频时长: {features.get('audio_duration', 0):.2f} 秒")
    print(f"  - 音量数据点: {len(features.get('volume', []))} 个")
    print(f"  - 音调数据点: {len(features.get('pitch', []))} 个")
    print(f"  - 语音活动检测: {len(features.get('voice_activity', []))} 个")

    transcription = features.get('transcription', '')
    if transcription:
        print(f"\n✓ 语音识别成功！")
        print(f"\n转录文本预览（前300字）:")
        print("="*60)
        print(transcription[:300])
        if len(transcription) > 300:
            print(f"... （共{len(transcription)}字）")
        print("="*60)
    else:
        print(f"\n✗ 语音识别失败或文本为空")

    # 清理临时文件
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
        print(f"\n临时音频文件已清理")

except Exception as e:
    print(f"\n✗ 音频特征提取失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("测试完成！")
print("="*60)
