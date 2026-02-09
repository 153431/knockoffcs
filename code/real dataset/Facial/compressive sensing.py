import numpy as np
from unittest import mock
from scipy.linalg import _decomp as linalg_decomp
from knockpy.knockoff_filter import KnockoffFilter
from knockpy.knockoffs import GaussianSampler
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.linear_model import orthogonal_mp
from tqdm import tqdm
import pickle

knockoff_selection_ratio = 0.01
high_dim = 500

variables_with_error= np.load("transformed_variables.npy", allow_pickle=True)

# 从文件中读取测量矩阵
with open("Another measurement matrix.pkl", "rb") as f:
    A = pickle.load(f)


# 生成 knockoff 矩阵（只需生成一次）
kfilter = KnockoffFilter(
    fstat='lcd',
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

Ak = generate_knockoffs_for_A(A)
# 从文件中读取测量矩阵
with open("knockoff of measurement matrix.pkl", "wb") as f:
    pickle.dump(Ak, f)

def run_experiment(y, methods=['lasso', 'omp', 'knockoff']):
    results = {}
    # LASSO implementation
    if 'lasso' in methods:
        lasso10 = Lasso(alpha=10)
        lasso10.fit(A, y)
        results['LASSO10'] = lasso10.coef_

        lasso1 = Lasso(alpha=1)
        lasso1.fit(A, y)
        results['LASSO1'] = lasso1.coef_

        lasso01 = Lasso(alpha=0.1)
        lasso01.fit(A, y)
        results['LASSO01'] = lasso01.coef_

        lasso001 = Lasso(alpha=0.01)
        lasso001.fit(A, y)
        results['LASSO001'] = lasso001.coef_

        lasso0001 = Lasso(alpha=0.001)
        lasso0001.fit(A, y)
        results['LASSO0001'] = lasso0001.coef_

    # Orthogonal Matching Pursuit
    if 'omp' in methods:
        omp_coef = orthogonal_mp(A, y)
        results['OMP'] = omp_coef

    # Generate MVR knockoffs
    if 'knockoff' in methods:
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
        #print(f"x_hat_knockoff:{x_hat_knockoff}")
        results['Knockoff'] = x_hat_knockoff

    return results

lasso10_signal_list = []
lasso1_signal_list = []
lasso01_signal_list = []
lasso001_signal_list = []
lasso0001_signal_list = []
OMP_signal_list = []
knockoffCS_signal_list = []

for i in tqdm(range(variables_with_error.shape[1])):
    results = run_experiment(variables_with_error[:, i])
    lasso10_signal_list.append(results['LASSO10'])
    lasso1_signal_list.append(results['LASSO1'])
    lasso01_signal_list.append(results['LASSO01'])
    lasso001_signal_list.append(results['LASSO001'])
    lasso0001_signal_list.append(results['LASSO0001'])
    OMP_signal_list.append(results['OMP'])
    knockoffCS_signal_list.append(results['Knockoff'])

lasso10_signal = np.array(lasso10_signal_list)
lasso1_signal = np.array(lasso1_signal_list)
lasso01_signal = np.array(lasso01_signal_list)
lasso001_signal = np.array(lasso001_signal_list)
lasso0001_signal = np.array(lasso0001_signal_list)
OMP_signal = np.array(OMP_signal_list)
knockoffCS_signal = np.array(knockoffCS_signal_list)

np.save('lasso10_signal.npy', lasso10_signal, allow_pickle=True)
np.save('lasso1_signal.npy', lasso1_signal, allow_pickle=True)
np.save('lasso01_signal.npy', lasso01_signal, allow_pickle=True)
np.save('lasso001_signal.npy', lasso001_signal, allow_pickle=True)
np.save('lasso0001_signal.npy', lasso0001_signal, allow_pickle=True)
np.save('OMP_signal.npy', OMP_signal, allow_pickle=True)
np.save('knockoffCS_signal.npy', knockoffCS_signal, allow_pickle=True)