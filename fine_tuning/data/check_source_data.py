
import json
# JSON 文件名是 'train.json'
with open('train.json', 'r') as file:
    data = json.load(file)  # 读取并解析 JSON 文件

expected_keys = {'positionName', 'resumeData', 'positionID', 'resumeRecordId'}
pid = ['020', '049', '028', '027', '033', '032', '009', '041', '050', '031', '012', '018', '001', '006', '035', '016', '022', '036', '043', '046', '039', '015', '047', '004', '007', '044', '017', '002', '026', '010', '003', '029', '013', '045', '037', '024', '019', '030', '011', '021', '042', '023', '005', '025', '051', '040', '008', '048', '038', '034', '014']
counts = {}
for p in pid:
    counts[p] = 0

# 检查是否为列表，并打印每个字典中的键和值
if isinstance(data, list):
    print("data:")
    print(len(data))
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
                counts[positionID] += 1
                if positionName and resumeData and positionID and resumeRecordId:
                    continue
                # ngram的数据需要
                # 把resumeid和对应的resume存储为json在get_document时使用
                # 把resumeid对应的positionid存储为json,读取成字典，key是resumeid，在create example时处理完document之后改一下
                # wwm的数据需要
                # data_collator改了，LineByLineTextDataset改了，然后测试dataloader的运行逻辑
                else:
                    print(f"Values in object {idx}: {positionName} and {resumeData} and {positionID} and {resumeRecordId}")

print("counts:")
print(counts)
# 经过验证，不存在缺键
# positionName和positionID匹配
# 经过验证，不存在空值
# train.json存在不同id的简历值相同，但label不同，怪。
counts_sum = 0
for k in counts.keys():
    counts_sum += int(counts[k])
print("counts_sum:")
print(counts_sum)
for k in counts.keys():
    counts[k] = counts[k] / counts_sum
print("counts:")
print(counts)

alpha = [0] * 51
for k in counts.keys():
    alpha[int(k)-1] = counts[k]
print("alpha:")
print(alpha)

for i in range(len(alpha)):
    alpha[i] = float(f"{alpha[i] * 10:.4f}")


print("alpha:")
print(alpha)
print(len(alpha))




