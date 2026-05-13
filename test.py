import os
import numpy as np

# 确保这里的路径和你的数据集路径一致
DATASET_PATH = "./speech_commands" 
EXCLUDE = ["_background_noise_"]

# 按照文件夹顺序重新生成映射表
TARGET_LABELS = sorted([d for d in os.listdir(DATASET_PATH) 
                       if os.path.isdir(os.path.join(DATASET_PATH, d)) and d not in EXCLUDE])

np.save("label_map.npy", np.array(TARGET_LABELS))
print(f"✅ 成功补全 label_map.npy，包含 {len(TARGET_LABELS)} 个类别。")
# import numpy as np
# # 强制手动创建一个只有 4 类的字典
# TARGET_LABELS = ["yes", "no", "stop", "go"]
# np.save("label_map.npy", np.array(TARGET_LABELS))
# print("✅ label_map 已与你的采集数据对齐：[yes, no, stop, go]")

# import numpy as np
# X = np.load("X_features.npy")
# print(X.shape)