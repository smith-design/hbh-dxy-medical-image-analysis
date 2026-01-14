"""
模型推理脚本
用于测试微调后的 Qwen2-VL 模型
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from pathlib import Path
import sys

def load_model(model_path=None, base_model="Qwen/Qwen2-VL-7B-Instruct"):
    """加载模型"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 使用设备: {device}")

    if model_path and Path(model_path).exists():
        print(f"📦 加载微调模型: {model_path}")
        from peft import PeftModel

        # 加载基座模型
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto"
        )

        # 加载 LoRA 权重
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        print(f"📦 加载基座模型: {base_model}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto"
        )

    processor = AutoProcessor.from_pretrained(base_model)
    model.eval()

    print("✅ 模型加载完成！")
    return model, processor, device

def diagnose_image(image_path, model, processor, device, question=None):
    """诊断图像"""

    if question is None:
        question = "请分析这张皮肤病变图像，判断病变类型并提供详细的诊断建议。"

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    print(f"📸 图像大小: {image.size}")

    # 准备输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    # 处理输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 生成诊断
    print("🔍 正在分析...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.7,
            top_p=0.9
        )

    # 解码输出
    generated_text = processor.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 提取助手回复
    diagnosis = generated_text.split("assistant\n")[-1].strip()

    return diagnosis

def main():
    """主函数"""

    if len(sys.argv) < 2:
        print("用法: python src/inference.py <图像路径> [问题]")
        print("示例: python src/inference.py test_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(image_path).exists():
        print(f"❌ 图像文件不存在: {image_path}")
        sys.exit(1)

    # 模型路径
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "models" / "qwen2vl_ham10000_lora"

    # 加载模型
    model, processor, device = load_model(
        model_path=str(model_path) if model_path.exists() else None
    )

    # 诊断
    diagnosis = diagnose_image(image_path, model, processor, device, question)

    # 输出结果
    print("\n" + "="*60)
    print("📋 诊断结果")
    print("="*60)
    print(diagnosis)
    print("="*60)

if __name__ == "__main__":
    main()
