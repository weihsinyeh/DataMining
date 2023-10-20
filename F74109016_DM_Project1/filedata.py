import pandas as pd

# 讀取 .data 檔案
data = pd.read_csv(".\inputs\data.data")  # 這裡的 delimiter 根據你的檔案分隔符號設定

# 假設你的檔案已經成功讀取並儲存在 'data' 變數中

# 做一些資料處理（如果需要的話）

# 將資料寫入 CSV 檔案
data.to_csv('data.csv', index=False)  # index=False 表示不寫入索引欄位