
import os
from openai import OpenAI
import base64

# 配置 API Key 和 Base URL
API_KEY = "ms-80e2a97a-5492-4c23-bd7c-2bb45497001e"
BASE_URL = "https://api-inference.modelscope.cn/v1"

print(f"🔄 正在连接魔塔 API: {BASE_URL}")

try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # 测试用的简单对话
    print("📤 发送测试请求...")
    response = client.chat.completions.create(
        model="Qwen/QVQ-72B-Preview",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "你好，请简单介绍一下自己。"
            }
        ],
        stream=False
    )
    
    print("\n✅ API 调用成功！")
    print("-" * 30)
    print(response.choices[0].message.content)
    print("-" * 30)

except Exception as e:
    print(f"\n❌ API 调用失败: {e}")
