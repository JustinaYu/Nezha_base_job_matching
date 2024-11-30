"""
本文件用于处理最原始的fine_tuning/data/train.json
expected_keys = {'positionName', 'resumeData', 'positionID', 'resumeRecordId'}
将 'resumeRecordId': 'resumeData'和'resumeRecordId': 'positionID'分别存入两个json文件在data目录下，作为FineTuningProcessor的构建参数document_file和position_file
"""
import json


# spilt train.json
def spilt_source_data(input_file, resume_data_json_path, position_id_json_path):
    with open(input_file, 'r') as file:
        data = json.load(file)  # 读取并解析 JSON 文件

    expected_keys = {'positionName', 'resumeData', 'positionID', 'resumeRecordId'}
    resume_data_dict = {}
    position_id_dict = {}
    # 检查是否为列表，并打印每个字典中的键和值
    if isinstance(data, list):
        for idx, obj in enumerate(data):
            if isinstance(obj, dict):
                obj_keys = set(obj.keys())
                if obj_keys != expected_keys:
                    print(f"Keys in object {idx}: {list(obj.keys())}")
                else:
                    positionName = obj['positionName']
                    resumeData = obj['resumeData']
                    positionID = obj['positionID']
                    resumeRecordId = obj['resumeRecordId']

                    resume_data_dict[resumeRecordId] = resumeData
                    position_id_dict[resumeRecordId] = positionID
    # 将两个字典保存到 JSON 文件
    with open(resume_data_json_path, 'w') as resume_file:
        json.dump(resume_data_dict, resume_file, indent=4)

    with open(position_id_json_path, 'w') as position_file:
        json.dump(position_id_dict, position_file, indent=4)


# spilt test.json
def spilt_test_data(input_file, test_resume_data_json_path):
    with open(input_file, 'r') as file:
        data = json.load(file)  # 读取并解析 JSON 文件

    expected_keys = {'resumeData', 'resumeRecordId'}
    test_resume_data_dict = {}
    # 检查是否为列表，并打印每个字典中的键和值
    if isinstance(data, list):
        for idx, obj in enumerate(data):
            if isinstance(obj, dict):
                obj_keys = set(obj.keys())
                if obj_keys != expected_keys:
                    print(f"Keys in object {idx}: {list(obj.keys())}")
                else:
                    resumeData = obj['resumeData']
                    resumeRecordId = obj['resumeRecordId']

                    test_resume_data_dict[resumeRecordId] = resumeData
    # 将两个字典保存到 JSON 文件
    with open(test_resume_data_json_path, 'w') as resume_file:
        json.dump(test_resume_data_dict, resume_file, indent=4)
