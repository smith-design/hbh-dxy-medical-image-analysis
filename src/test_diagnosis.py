"""
测试诊断系统 - 使用真实标签
"""

import pandas as pd
from pathlib import Path
import sys

# 添加 API Key
API_KEY = "sk-80e2a97a-5492-4c23-bd7c-2bb45497001e"

# 疾病名称
DISEASE_NAMES = {
    'akiec': '光化性角化病和上皮内癌',
    'bcc': '基底细胞癌',
    'bkl': '良性角化病变',
    'df': '皮肤纤维瘤',
    'mel': '黑色素瘤',
    'nv': '黑色素痣',
    'vasc': '血管病变'
}

def test_diagnosis(image_path):
    """测试诊断功能"""
    
    print(f"\n{'='*60}")
    print("🔍 皮肤病变诊断系统测试")
    print(f"{'='*60}\n")
    
    # 加载元数据
    metadata_path = Path("datasets/archive (6)/HAM10000_metadata.csv")
    if not metadata_path.exists():
        print("❌ 未找到元数据文件")
        return
    
    metadata = pd.read_csv(metadata_path)
    
    # 从文件名提取 image_id
    image_id = Path(image_path).stem
    print(f"📸 图像 ID: {image_id}")
    
    # 查找真实标签
    row = metadata[metadata['image_id'] == image_id]
    
    if len(row) == 0:
        print(f"❌ 未找到图像 {image_id} 的元数据")
        return
    
    # 获取信息
    dx = row.iloc[0]['dx']
    age = row.iloc[0]['age']
    sex = row.iloc[0]['sex']
    localization = row.iloc[0]['localization']
    
    disease_name = DISEASE_NAMES[dx]
    
    print(f"\n✅ 分类结果:")
    print(f"   病变类型: {disease_name} ({dx})")
    print(f"   置信度: 100% (真实标签)")
    
    print(f"\n👤 患者信息:")
    print(f"   年龄: {age}岁")
    print(f"   性别: {sex}")
    print(f"   位置: {localization}")
    
    # 生成报告
    print(f"\n📝 生成诊断报告...")
    
    try:
        sys.path.append('src')
        from modelscope_api import ModelScopeAPI
        
        api = ModelScopeAPI(api_key=API_KEY)
        
        patient_info = {
            'age': age,
            'sex': sex,
        'localization': localization
        }
        
        report = api.generate_diagnosis_report(
            disease_type=dx,
            disease_name=disease_name,
            confidence=1.0,
            patient_info=patient_info
        )
        
        print(f"\n{'='*60}")
        print("📋 诊断报告")
        print(f"{'='*60}\n")
        print(report)
        
    except Exception as e:
        print(f"⚠️  报告生成失败: {e}")
        print("使用备用报告...")
        
        from modelscope_api import ModelScopeAPI
        api = ModelScopeAPI()
        report = api._generate_fallback_report(dx, disease_name, 1.0, patient_info)
        
        print(f"\n{'='*60}")
        print("📋 诊断报告")
        print(f"{'='*60}\n")
        print(report)
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 src/test_diagnosis.py <图像路径>")
        sys.exit(1)
    
    test_diagnosis(sys.argv[1])
