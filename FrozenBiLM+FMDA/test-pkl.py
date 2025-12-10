import pickle

# 读取.pkl文件
with open('/home/zhouhao/lmy/result_frozenBilm/MSRVTT-QA/subtitles_bak.pkl', 'rb') as f:
   data = pickle.load(f)

# 使用读取的数据
print(data)