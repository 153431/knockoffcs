import numpy as np
import cvxpy as cp
from tqdm import tqdm
import pickle

knockoff_selection_ratio = 0.01
high_dim = 100

variables_with_error= np.load("transformed_variables.npy", allow_pickle=True)

# 从文件中读取测量矩阵
with open("Another measurement matrix.pkl", "rb") as f:
    A = pickle.load(f)


def clime_compressive_sensing(A, y, lambda_):
    """
    使用 CLIME 风格优化进行压缩感知支持集恢复
    参数：
        X: 测量矩阵 (n x p)
        y: 观测向量 (n,)
        lambda_: 正则化参数
    返回：
        beta_hat: 估计的稀疏信号
        support: 估计的支持集
    """
    p = A.shape[1]

    # 定义变量
    beta = cp.Variable(p)

    # 目标：min ||beta||_1
    objective = cp.Minimize(cp.norm(beta, 1))

    # 约束：||X beta - y||_inf <= lambda
    constraints = [cp.max(cp.abs(A @ beta - y)) <= lambda_]

    # 求解
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # 估计 beta
    beta_hat = beta.value

    # 支持集恢复：选择 |beta_i| > tau 的索引
    tau = 1e-3  # 阈值，需根据问题调整
    support = np.where(np.abs(beta_hat) > tau)[0]

    return beta_hat, support



CLIME01_signal_list = []

for i in tqdm(range(variables_with_error.shape[1])):
    # 运行 CLIME 风格优化
    beta_hat, support_hat = clime_compressive_sensing(A,variables_with_error[:, i], 0.1)
    CLIME01_signal_list.append(beta_hat)

CLIME01_signal = np.array(CLIME01_signal_list)


np.save('CLIME01_signal.npy', CLIME01_signal, allow_pickle=True)
