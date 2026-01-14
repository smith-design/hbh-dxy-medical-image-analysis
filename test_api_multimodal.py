
import os
import base64
from openai import OpenAI

# 配置 API Key 和 Base URL
API_KEY = "ms-80e2a97a-5492-4c23-bd7c-2bb45497001e"
BASE_URL = "https://api-inference.modelscope.cn/v1"

print(f"🔄 正在连接魔塔 API (多模态测试): {BASE_URL}")

try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # 使用网络图片测试，避免依赖本地文件
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/QVQ/demo.png"

    print("📤 发送包含图片的测试请求...")
    response = client.chat.completions.create(
        model="Qwen/QVQ-72B-Preview",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": "这张图片里有什么？"
                    }
                ]
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
