import json

# 读取数据集文件的内容
with open('ZenHeart401713.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 转换格式
new_data = []
for conversation in data:
    new_conversation = {
        "conversation": [
            {
                "system": conversation["system"],
                "input": conversation["input"],
                "output": "\n".join(conversation["output"].split("\n"))
            }
        ]
    }
    new_data.append(new_conversation)

# 保存到新的文件ZenHeartXTuner.json
with open('ZenHeartXTunerData.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)