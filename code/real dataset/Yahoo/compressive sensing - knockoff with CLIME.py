import numpy as np
from tqdm import tqdm
import pickle
from unittest import mock
from scipy.linalg import _decomp as linalg_decomp
from knockpy.knockoff_filter import KnockoffFilter
from knockpy.knockoffs import GaussianSampler
from sklearn.covariance import LedoitWolf
from knockpy.knockoff_stats import FeatureStatistic, combine_Z_stats
from sklearn.linear_model import LinearRegression
import cvxpy as cp

knockoff_selection_ratio = 0.01
high_dim = 1000

variables_with_error= np.load("transformed_variables.npy", allow_pickle=True)

# 从文件中读取测量矩阵
with open("measurement matrix.pkl", "rb") as f:
    A = pickle.load(f)

class ClimeStatistic(FeatureStatistic):
    """ CLIME-style compressive sensing statistic wrapper class """

    def __init__(self):
        super().__init__()

    def fit(
        self,
        X,
        Xk,
        y,
        groups=None,
        antisym="cd",
        group_agg="avg",
        cv_score=False,
        lambda_=0.1,  # 默认正则化参数
        tau=1e-3,     # 默认支持集阈值
        **kwargs,
    ):
        """
        Wraps the FeatureStatistic class but uses CLIME-style compressive sensing
        coefficients as variable importances.

        Parameters
        ----------
        X : np.ndarray
            the ``(n, p)``-shaped design matrix
        Xk : np.ndarray
            the ``(n, p)``-shaped matrix of knockoffs
        y : np.ndarray
            ``(n,)``-shaped response vector
        groups : np.ndarray
            For group knockoffs, a p-length array of integers from 1 to
            num_groups such that ``groups[j] == i`` indicates that variable `j`
            is a member of group `i`. Defaults to None (regular knockoffs).
        antisym : str
            The antisymmetric function used to create (ungrouped) feature
            statistics. Three options:
            - "CD" (Difference of absolute vals of coefficients),
            - "SM" (Signed maximum).
            - "SCD" (Simple difference of coefficients - NOT recommended)
        group_agg : str
            For group knockoffs, specifies how to turn individual feature
            statistics into grouped feature statistics. Two options:
            "sum" and "avg".
        cv_score : bool
            If true, score the feature statistic's predictive accuracy
            using cross validation. (Not implemented for CLIME)
        lambda_ : float
            Regularization parameter for CLIME
        tau : float
            Threshold for support recovery in CLIME
        kwargs : dict
            Extra kwargs (ignored in this implementation)

        Returns
        -------
        W : np.ndarray
            an array of feature statistics. This is ``(p,)``-dimensional
            for regular knockoffs and ``(num_groups,)``-dimensional for
            group knockoffs.
        """
        # 设置默认分组
        p = X.shape[1]
        if groups is None:
            groups = np.arange(1, p + 1, 1)

        # 合并 X 和 Xk 进行联合计算
        X_full = np.hstack([X, Xk])

        # 标准化 y（可选，确保数值稳定性）
        y = (y - np.mean(y)) / np.std(y)

        # Step 1: 使用 CLIME 风格压缩感知计算 Z 统计量
        beta_hat, _ = clime_compressive_sensing(
            A=X_full,
            y=y,
            lambda_=lambda_,
            tau=tau
        )

        # Step 2: 使用 beta_hat 作为 Z（前 p 个为原始特征，后 p 个为敲除特征）
        Z = beta_hat  # beta_hat 已经是 (2p,) 形状

        # Step 3: 组合 Z 统计量
        W_group = combine_Z_stats(Z, groups, antisym=antisym, group_agg=group_agg)

        # 保存值以供后续使用
        self.Z = Z
        self.groups = groups
        self.W = W_group

        # 交叉验证分数（未实现）
        if cv_score:
            self.score = None
            self.score_type = "not_implemented"
            print("Warning: cv_score is not implemented for CLIME.")

        return W_group

def clime_compressive_sensing(A, y, lambda_, tau=1e-3):
    """
    使用 CLIME 风格优化进行压缩感知支持集恢复
    参数：
        A: 测量矩阵 (n x p)
        y: 观测向量 (n,)
        lambda_: 正则化参数
        tau: 支持集阈值
    返回：
        beta_hat: 估计的稀疏信号
        support: 估计的支持集
    """
    p = A.shape[1]

    # 定义变量
    beta = cp.Variable(p)

    # 目标：min ||beta||_1
    objective = cp.Minimize(cp.norm(beta, 1))

    # 约束：||A beta - y||_inf <= lambda
    constraints = [cp.max(cp.abs(A @ beta - y)) <= lambda_]

    # 求解
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    # 估计 beta
    beta_hat = beta.value

    # 支持集恢复：选择 |beta_i| > tau 的索引
    support = np.where(np.abs(beta_hat) > tau)[0]

    return beta_hat, support


# 生成 knockoff 矩阵（只需生成一次）
kfilter = KnockoffFilter(
    fstat=ClimeStatistic(),
    ksampler='gaussian',
    knockoff_kwargs={"method": "mvr"}
)

def generate_knockoffs_for_A(A, method="mvr"):
    """
    给定一个 (m, n) 的测量矩阵 A，生成其 knockoff 版本 A_k。
    自动处理协方差矩阵估计或采样中的异常。
    """
    X = A.copy()

    # Step 1: 安全估计协方差矩阵 Sigma
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lw = LedoitWolf().fit(X)
            Sigma = lw.covariance_
    except Exception as e:
        print(f"LedoitWolf failed with {type(e).__name__}: {e}")

        # 替换 scipy 的 eigh 实现为 eig
        def patched_eig(a, lower=None, check_finite=False):
            w, vr = np.linalg.eig(a)
            return w, vr

        with mock.patch.object(linalg_decomp, 'eigh', patched_eig):
            print("Using patched eig() instead of eigh() for covariance estimation.")
            lw = LedoitWolf().fit(X)
            Sigma = lw.covariance_

    mu = X.mean(axis=0)

    # Step 2: Knockoff sampling，防止内部线性代数出错
    try:
        sampler = GaussianSampler(X=X, mu=mu, Sigma=Sigma, method=method)
        Xk = sampler.sample_knockoffs()
    except np.linalg.LinAlgError as e:
        raise RuntimeError("Knockoff sampling failed due to linear algebra issue.") from e
    return Xk

# 从文件中读取测量矩阵
with open("knockoff of measurement matrix.pkl", "rb") as f:
    Ak = pickle.load(f)


def run_experiment(y):
    results = {}
    # Compute knockoffs and feature statistics
    # Apply knockoff filter with FDR control
    fdr_selected, W = kfilter.forward(X=A, y=y, Xk=Ak)
    #print(f"W:{W}")
    # 计算变量总数和需要选择的变量个数（10%）
    num_vars = len(W)
    num_to_select = int(np.ceil(knockoff_selection_ratio * num_vars))

    # 获取W分数最高的变量索引，直接用selected表示
    selected = np.argsort(W)[-num_to_select:]
    #print(selected)

    # Estimate non-zero coefficients
    x_hat_knockoff = np.zeros(high_dim)
    if len(selected) > 0:
        lr = LinearRegression()
        x_hat_knockoff[selected] = lr.fit(A[:, selected], y).coef_

    return x_hat_knockoff

knockoffCS_signal_list_CLIME = []

for i in tqdm(range(variables_with_error.shape[1])):
    results = run_experiment(variables_with_error[:, i])
    knockoffCS_signal_list_CLIME.append(results)

knockoffCS_signal = np.array(knockoffCS_signal_list_CLIME)

np.save('knockoffCS_signal_CLIME.npy', knockoffCS_signal, allow_pickle=True)