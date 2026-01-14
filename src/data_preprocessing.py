"""
HAM10000 数据预处理脚本
将 HAM10000 数据集转换为 LLaMA-Factory 支持的多模态训练格式
"""

import os
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

# 疾病类型映射
DISEASE_MAPPING = {
    'akiec': '光化性角化病和上皮内癌 (Actinic Keratoses and Intraepithelial Carcinoma)',
    'bcc': '基底细胞癌 (Basal Cell Carcinoma)',
    'bkl': '良性角化病变 (Benign Keratosis)',
    'df': '皮肤纤维瘤 (Dermatofibroma)',
    'mel': '黑色素瘤 (Melanoma)',
    'nv': '黑色素痣 (Melanocytic Nevi)',
    'vasc': '血管病变 (Vascular Lesions)'
}

# 疾病描述模板
DISEASE_DESCRIPTIONS = {
    'akiec': '这是一种光化性角化病或上皮内癌，通常由长期日晒引起，表现为粗糙、鳞状的皮肤斑块。',
    'bcc': '这是基底细胞癌，最常见的皮肤癌类型，通常生长缓慢，很少转移，但需要及时治疗。',
    'bkl': '这是良性角化病变，包括脂溢性角化病等，通常无害但可能影响美观。',
    'df': '这是皮肤纤维瘤，一种良性的纤维组织增生，通常表现为坚硬的小结节。',
    'mel': '这是黑色素瘤，最危险的皮肤癌类型，可能快速生长和转移，需要紧急医疗关注。',
    'nv': '这是黑色素痣，俗称痣或痦子，通常是良性的，但需要监测变化。',
    'vasc': '这是血管病变，包括血管瘤等，由血管异常增生引起。'
}

def prepare_data():
    """准备训练数据"""

    # 路径配置
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "datasets" / "archive (6)"
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建图像目录
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print("📊 读取元数据...")
    metadata = pd.read_csv(dataset_dir / "HAM10000_metadata.csv")

    print(f"✅ 总样本数: {len(metadata)}")
    print(f"✅ 疾病类型分布:")
    print(metadata['dx'].value_counts())

    # 准备训练数据
    data_samples = []

    print("\n🔄 处理图像和生成训练样本...")
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image_id = row['image_id']
        lesion_id = row['lesion_id']
        dx = row['dx']
        dx_type = row['dx_type']
        age = row['age']
        sex = row['sex']
        localization = row['localization']

        # 查找图像文件
        image_path_1 = dataset_dir / "HAM10000_images_part_1" / f"{image_id}.jpg"
        image_path_2 = dataset_dir / "HAM10000_images_part_2" / f"{image_id}.jpg"

        if image_path_1.exists():
            source_image = image_path_1
        elif image_path_2.exists():
            source_image = image_path_2
        else:
            print(f"⚠️  图像未找到: {image_id}")
            continue

        # 复制图像到处理目录
        target_image = images_dir / f"{image_id}.jpg"
        if not target_image.exists():
            shutil.copy2(source_image, target_image)

        # 生成多样化的训练样本
        disease_name = DISEASE_MAPPING[dx]
        disease_desc = DISEASE_DESCRIPTIONS[dx]

        # 样本1: 疾病分类
        sample1 = {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>请分析这张皮肤病变图像，判断病变类型。"
                },
                {
                    "role": "assistant",
                    "content": f"根据图像分析，这是{disease_name}。{disease_desc}"
                }
            ],
            "images": [f"images/{image_id}.jpg"]
        }

        # 样本2: 详细诊断
        patient_info = f"患者信息：{sex}，{age}岁，病变位置：{localization}"
        sample2 = {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>请提供详细的诊断分析。{patient_info}。"
                },
                {
                    "role": "assistant",
                    "content": f"诊断结果：{disease_name}\n\n{disease_desc}\n\n{patient_info}\n诊断方法：{dx_type}\n\n建议：请咨询皮肤科医生进行进一步评估和治疗方案制定。"
                }
            ],
            "images": [f"images/{image_id}.jpg"]
        }

        # 样本3: 简短问答
        sample3 = {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>这是什么类型的皮肤病变？"
                },
                {
                    "role": "assistant",
                    "content": f"{disease_name}"
                }
            ],
            "images": [f"images/{image_id}.jpg"]
        }

        data_samples.extend([sample1, sample2, sample3])

    print(f"\n✅ 生成训练样本总数: {len(data_samples)}")

    # 划分训练集和验证集
    train_samples, val_samples = train_test_split(
        data_samples,
        test_size=0.1,
        random_state=42
    )

    print(f"📊 训练集样本数: {len(train_samples)}")
    print(f"📊 验证集样本数: {len(val_samples)}")

    # 保存为 JSON 格式
    train_file = output_dir / "train.json"
    val_file = output_dir / "val.json"

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)

    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 训练数据已保存到: {train_file}")
    print(f"✅ 验证数据已保存到: {val_file}")

    # 生成数据集配置文件
    dataset_info = {
        "ham10000_skin_lesion": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
    }

    dataset_info_file = output_dir / "dataset_info.json"
    with open(dataset_info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"✅ 数据集配置已保存到: {dataset_info_file}")

    # 生成统计报告
    print("\n" + "="*50)
    print("📈 数据集统计报告")
    print("="*50)
    print(f"原始图像数量: {len(metadata)}")
    print(f"生成训练样本: {len(data_samples)}")
    print(f"训练集: {len(train_samples)}")
    print(f"验证集: {len(val_samples)}")
    print(f"图像存储路径: {images_dir}")
    print("="*50)

if __name__ == "__main__":
    prepare_data()
