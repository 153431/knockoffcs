import pandas as pd

# 读取 CSV 文件
file_path = "result.csv"
df = pd.read_csv(file_path)

# 选择需要的列
fdr_power_columns = ["m", "n", "s", "snr",  "LASSO_FDR", "OMP_FDR", "Knockoff_FDR","LASSO_Power", "OMP_Power", "Knockoff_Power"]
fdr_power_df = df[fdr_power_columns].copy()
fdr_power_df["snr"] = fdr_power_df["snr"].round(0).astype(int)  # SNR 保留整数
#fdr_power_df["knockoff_selection_ratio"] = fdr_power_df["knockoff_selection_ratio"].map(lambda x: f"{x:.2f}")  # 保留 1 位小数
# 对 FDR 数据进行四舍五入处理
fdr_power_df[["LASSO_FDR", "OMP_FDR", "Knockoff_FDR","LASSO_Power", "OMP_Power", "Knockoff_Power"]] = fdr_power_df[["LASSO_FDR", "OMP_FDR", "Knockoff_FDR","LASSO_Power", "OMP_Power", "Knockoff_Power"]].round(3)
fdr_power_df["Lasso_F1"] = 2 * (1 - fdr_power_df["LASSO_FDR"]) * fdr_power_df["LASSO_Power"] / (1 - fdr_power_df["LASSO_FDR"] + fdr_power_df["LASSO_Power"])
fdr_power_df["OMP_F1"] = 2 * (1 - fdr_power_df["OMP_FDR"]) * fdr_power_df["OMP_Power"] / (1 - fdr_power_df["OMP_FDR"] + fdr_power_df["OMP_Power"])
fdr_power_df["Knockoff_F1"] = 2 * (1 - fdr_power_df["Knockoff_FDR"]) * fdr_power_df["Knockoff_Power"] / (1 - fdr_power_df["Knockoff_FDR"] + fdr_power_df["Knockoff_Power"])
#fdr_power_df.drop(inplace = True,columns=["LASSO_FDR", "OMP_FDR", "Knockoff_FDR","LASSO_Power", "OMP_Power", "Knockoff_Power"])

# 转换为 LaTeX 表格
fdr_power_latex_table = fdr_power_df.to_latex(index=False, float_format="%.3f")
# 输出 LaTeX 代码
print(fdr_power_latex_table)

error_columns = ["m", "n", "s", "snr",  "LASSO_RelError", "OMP_RelError", "Knockoff_RelError","LASSO_ReconstructionError", "OMP_ReconstructionError", "Knockoff_ReconstructionError"]
error_df = df[error_columns].copy()
error_df["snr"] = error_df["snr"].round(0).astype(int)  # SNR 保留整数
#error_df["knockoff_selection_ratio"] = error_df["knockoff_selection_ratio"].map(lambda x: f"{x:.2f}")  # 保留 1 位小数
# 对 FDR 数据进行四舍五入处理
#error_df[["LASSO_RelError", "OMP_RelError", "Knockoff_RelError","LASSO_ReconstructionError", "OMP_ReconstructionError", "Knockoff_ReconstructionError"]] = error_df[["LASSO_RelError", "OMP_RelError", "Knockoff_RelError","LASSO_ReconstructionError", "OMP_ReconstructionError", "Knockoff_ReconstructionError"]].round(3)
# 转换为 LaTeX 表格
#error_latex_table = error_df.to_latex(index=False, float_format="%.3f")
# 转换为 LaTeX 表格
error_latex_table = error_df.to_latex(index=False)
# 输出 LaTeX 代码
print(error_latex_table)