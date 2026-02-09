import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('data.csv', dtype=float)

# 删除包含 NaN 的行
df_cleaned = df.dropna()

# 随机抽取 1000 行的索引
np.random.seed(10)  # 设置随机种子以确保可重复性（可选）
n_samples = 3000
total_rows_cleaned = df_cleaned.shape[0]  # 删除 NaN 后的数据集总行数

random_indices = np.random.choice(total_rows_cleaned, size=n_samples, replace=False)

# 提取 'data_arc' 列的值
label = df_cleaned.iloc[random_indices]['data_arc']

# 检查 label 中是否存在 NaN
has_nan = label.isna().any()

print("label 中是否包含 NaN：", has_nan)
np.save('Original Dataset_labels.npy', label, allow_pickle=True)

# 如果没有 NaN，对除 'data_arc' 之外的列进行标准化
if not has_nan:
    columns_to_scale = [col for col in df_cleaned.columns if col != 'data_arc']
    scaler = StandardScaler()
    df_cleaned[columns_to_scale] = scaler.fit_transform(df_cleaned[columns_to_scale])

    print("\n标准化后的数据：")
    print(df_cleaned.head())

print("variables 中是否包含 NaN：", has_nan)
variables = df_cleaned.iloc[random_indices,0:24].T
print(variables.shape)
np.save('Original Dataset_variables.npy', variables, allow_pickle=True)
row_variances = np.var(variables, axis=1, ddof=1)
row_std = np.sqrt(row_variances)
std = 0.5 * row_std.mean()
noise_dimensions = 476  # 要添加的噪声维度数量
low_dim = 100

def add_noise_and_shuffle(vectors, noise_dims, noise_scale=0.1):
    """
    为矩阵的每个列向量添加噪声维度并随机打乱所有维度

    Parameters:
    vectors: 输入矩阵 (n_dims, n_samples)
    noise_dims: 要添加的噪声维度数量
    noise_scale: 噪声的标准差

    Returns:
    shuffled_vectors: 处理后的矩阵
    shuffle_idx: 打乱后的维度索引（用于追踪原始维度位置）
    """
    n_dims, n_samples = vectors.shape

    # 生成高斯噪声
    noise = np.random.normal(0, noise_scale, (noise_dims, n_samples))

    # 将原始信号与噪声拼接
    extended_vectors = np.vstack([vectors, noise])

    # 生成随机打乱的索引
    shuffle_idx = np.random.permutation(n_dims + noise_dims)

    # 按索引打乱维度
    shuffled_vectors = extended_vectors[shuffle_idx, :]

    return shuffled_vectors, shuffle_idx

# 调用函数
extended_variables, dimension_indices = add_noise_and_shuffle(variables, noise_dimensions, std)
high_dim = extended_variables.shape[0]
A = np.load('measurement matrix.pkl', allow_pickle=True)

A_fake = np.random.randn(low_dim, high_dim)
with open("Another measurement matrix.pkl", "wb") as f:
    pickle.dump(A_fake, f)

transformed_variables = A @ extended_variables
print(transformed_variables.shape)

# # 打印结果信息
# print(f"原始矩阵形状: {variables.shape}")
# print(f"处理后矩阵形状: {processed_vectors.shape}")
# print(f"维度打乱索引: {dimension_indices}")

# 可选：保存处理后的结果
np.save("Processed_vectors.npy", extended_variables, allow_pickle=True)
np.save("Dimension_indices.npy", dimension_indices, allow_pickle=True)
np.save('transformed_variables.npy', transformed_variables, allow_pickle=True)