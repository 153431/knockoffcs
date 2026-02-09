import numpy as np
import pickle

# 使用 numpy.genfromtxt() 读取 CSV 文件，并确保数据类型为 float
import numpy as np

# 读取 CSV 文件
dataset = np.genfromtxt("magic04.csv", delimiter=",", dtype=None, encoding=None)

# 检查 dataset 的结构
print(dataset.dtype)  # 查看字段名和类型
print(dataset.shape)  # 查看形状

# 随机抽取 1000 行的索引
np.random.seed(42)  # 设置随机种子以确保可重复性（可选）
n_samples = 1000
total_rows = dataset.shape[0]  # 数据集总行数（例如 19020）
random_indices = np.random.choice(total_rows, size=n_samples, replace=False)

# 提取变量（随机 1000 行的前 10 列，数值特征）
variables = np.array([list(row)[:10] for row in dataset[random_indices]], dtype=float).T
np.save('Original Dataset_variables.npy', variables.T, allow_pickle=True)

# 提取标签（随机 1000 行的第 11 列，字符串）
label = dataset[random_indices]['f10']  # 'f10' 是第 11 列的字段名（字符串列）

# 将 'g' 映射为 1，'h' 映射为 0
encoded_label = np.where(label == 'g', 1, 0)

np.save('Original Dataset_labels.npy', encoded_label, allow_pickle=True)
row_variances = np.var(variables, axis=1, ddof=1)
row_std = np.sqrt(row_variances)
std = 0.5 * row_std.mean()
noise_dimensions = 990  # 要添加的噪声维度数量
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
A = np.random.randn(low_dim, high_dim)

# 保存到文件
with open("real measurement matrix.pkl", "wb") as f:
    pickle.dump(A, f)

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