import numpy as np
import cvxpy as cp
from tqdm import tqdm
import pickle
import mosek

knockoff_selection_ratio = 0.01
high_dim = 2000

variables_with_error= np.load("transformed_variables.npy", allow_pickle=True)

# 从文件中读取测量矩阵
with open("Another measurement matrix.pkl", "rb") as f:
    A = pickle.load(f)

def dantzig_compressive_sensing(A, y, lambda_):
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
    ATA = A.T @ A
    # 定义变量
    beta = cp.Variable(p)

    # 目标：min ||beta||_1
    objective = cp.Minimize(cp.norm(beta, 1))

    # 约束：||X^T (y - X beta)||_inf <= lambda
    # 约束：||X beta - y||_inf <= lambda
    constraints = [cp.max(cp.abs(ATA @ beta - A.T @ y)) <= lambda_]

    # 求解
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    # 估计 beta
    beta_hat = beta.value

    # 支持集恢复：选择 |beta_i| > tau 的索引
    tau = 1e-3  # 阈值，需根据问题调整
    support = np.where(np.abs(beta_hat) > tau)[0]

    return beta_hat, support


def dantzig_selector_admm(X, y, lambda_, rho=1.0, max_iter=1000, tol=1e-4):
    """
    使用 ADMM 实现 Dantzig Selector
    参数：
        X: 设计矩阵 (n x p)
        y: 响应向量 (n,)
        lambda_: 正则化参数
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛容差
    返回：
        beta: 估计的回归系数
    """
    n, p = X.shape
    Xt = X.T
    XtX = Xt @ X
    Xty = Xt @ y

    # 初始化
    beta = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)

    # ADMM 迭代
    for _ in range(max_iter):
        # 更新 beta：(XtX + rho I) beta = Xty - z + u
        beta_old = beta.copy()
        A = XtX + rho * np.eye(p)
        b = Xty - z + u
        beta = np.linalg.solve(A, b)

        # 更新 z：投影到 ||z||_inf <= lambda
        z = np.clip(Xt @ (y - X @ beta) + u, -lambda_, lambda_)

        # 更新 u
        u = u + Xt @ (y - X @ beta) - z

        # 检查收敛
        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta


dantzig01_signal_list = []

for i in tqdm(range(variables_with_error.shape[1])):
    # 运行 dantzig 风格优化
    beta_hat = dantzig_selector_admm(A,variables_with_error[:, i], 0.1)
    dantzig01_signal_list.append(beta_hat)

dantzig01_signal = np.array(dantzig01_signal_list)

np.save('dantzig01_signal.npy', dantzig01_signal, allow_pickle=True)