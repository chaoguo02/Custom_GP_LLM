import os
import json

def ensure_directory_exists(file_path):
    """ 确保文件的目录存在 """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

def write_json(file_path, data):
    """ 将数据写入 JSON 文件 """
    ensure_directory_exists(file_path)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def write_jsonl(file_path, data_list):
    """ 将数据写入 JSONL 文件（逐行 JSON 记录） """
    ensure_directory_exists(file_path)
    with open(file_path, 'w') as f:
        for data in data_list:
            f.write(json.dumps(data) + "\n")

def write_jsonl2(file_path, data):
    """
    以 JSONL 格式写入数据，每行是一个 JSON 对象。
    :param file_path: 输出文件路径
    :param data: 需要写入的列表，列表中的每个元素应为字典格式
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")  # 确保每条数据占据一行


def read_json(file_path):
    """ 从 JSON 文件中读取数据 """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def read_jsonl(file_path):
    """ 从 JSONL 文件中读取所有数据行 """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    return []
