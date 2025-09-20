import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = "./inspire_hand_data_process/pure_joint_data.xlsx"  # 请根据自己的路径进行修改
df = pd.read_excel(file_path)

# 将每一列转换为 numpy ndarray
thumb1A_degree = np.deg2rad(df["thumb1-A"].to_numpy())
thumb2B_degree = np.deg2rad(df["thumb2-B"].to_numpy())
thumb3C_degree = np.deg2rad(df["thumb3-C"].to_numpy())
thumbRotD_degree = np.deg2rad(df["thumb_rot-D"].to_numpy())
finger1E_degree = np.deg2rad(df["finger1-E"].to_numpy())
finger2F_degree = np.deg2rad(df["finger2-F"].to_numpy())

data_dict = {
    "thumb1-A": thumb1A_degree,
    "thumb2-B": thumb2B_degree,
    "thumb3-C": thumb3C_degree,
    "thumb_rot-D": thumbRotD_degree,
    "finger1-E": finger1E_degree,
    "finger2-F": finger2F_degree
}

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

slope_list = []
intercept_list = []

for i, (joint_name, joint_data) in enumerate(data_dict.items()):
    x = np.arange(len(joint_data))
    slope, intercept = np.polyfit(x, joint_data, 1)
    slope_list.append(slope)
    intercept_list.append(intercept)

    y_fit = slope * x + intercept
    ax = axes[i]
    ax.scatter(x, joint_data, label="Data", alpha=0.7)
    ax.plot(x, y_fit, color="red", label=f"Fit (slope={slope:.2f})")
    ax.set_title(joint_name)
    ax.legend()
    ax.set_xlabel("Index")
    ax.set_ylabel("Radian")

plt.tight_layout()
plt.savefig("./inspire_hand_data_process/inspire_hand_joint_angle.png")
plt.show()
plt.close()

thumb1A_slope = slope_list[0]
thumb2B_slope = slope_list[1]
thumb3C_slope = slope_list[2]
thumbRotD_slope = slope_list[3]
finger1E_slope = slope_list[4]
finger2F_slope = slope_list[5]

thumb_mimic_A2B = thumb2B_slope / thumb1A_slope
thumb_mimic_A2C = thumb3C_slope / thumb1A_slope
finger_mimin_E2F = finger2F_slope / finger1E_slope

print(f"thumb_mimic_A2B is {thumb_mimic_A2B}")
print(f"thumb_mimic_A2C is {thumb_mimic_A2C}")
print(f"finger_mimin_E2F is {finger_mimin_E2F}")

# -------- 在这里对指定关节对进行「X-Y」线性拟合并绘图 -------- #
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

# 1) 以 thumb1-A 为横轴， thumb2-B 为纵轴
x_A = thumb1A_degree
y_B = thumb2B_degree
slope_AB, intercept_AB = np.polyfit(x_A, y_B, 1)
y_fit_AB = slope_AB * x_A + intercept_AB

axes2[0].scatter(x_A, y_B, label="Data", alpha=0.7)
axes2[0].plot(x_A, y_fit_AB, color="red",
             label=f"Fit: slope={slope_AB:.2f}, intercept={intercept_AB:.2f}")
axes2[0].set_title("thumb1-A vs thumb2-B")
axes2[0].set_xlabel("thumb1-A(rad)")
axes2[0].set_ylabel("thumb2-B(rad)")
axes2[0].legend()

# 2) 以 thumb1-A 为横轴， thumb3-C 为纵轴
x_A = thumb1A_degree
y_C = thumb3C_degree
slope_AC, intercept_AC = np.polyfit(x_A, y_C, 1)
y_fit_AC = slope_AC * x_A + intercept_AC

axes2[1].scatter(x_A, y_C, label="Data", alpha=0.7)
axes2[1].plot(x_A, y_fit_AC, color="red",
             label=f"Fit: slope={slope_AC:.2f}, intercept={intercept_AC:.2f}")
axes2[1].set_title("thumb1-A vs thumb3-C")
axes2[1].set_xlabel("thumb1-A(rad)")
axes2[1].set_ylabel("thumb3-C(rad)")
axes2[1].legend()

# 3) 以 finger1-E 为横轴， finger2-F 为纵轴
x_E = finger1E_degree
y_F = finger2F_degree
slope_EF, intercept_EF = np.polyfit(x_E, y_F, 1)
y_fit_EF = slope_EF * x_E + intercept_EF

axes2[2].scatter(x_E, y_F, label="Data", alpha=0.7)
axes2[2].plot(x_E, y_fit_EF, color="red",
             label=f"Fit: slope={slope_EF:.2f}, intercept={intercept_EF:.2f}")
axes2[2].set_title("finger1-E vs finger2-F")
axes2[2].set_xlabel("finger1-E(rad)")
axes2[2].set_ylabel("finger2-F(rad)")
axes2[2].legend()

plt.tight_layout()
plt.savefig("./inspire_hand_data_process/inspire_hand_mimic_fit.png")
plt.show()
plt.close()


print("==== 线性拟合结果（X→Y） ====")
print(f"thumb1-A → thumb2-B: slope={slope_AB:.3f}, intercept={intercept_AB:.3f}")
print(f"thumb1-A → thumb3-C: slope={slope_AC:.3f}, intercept={intercept_AC:.3f}")
print(f"finger1-E → finger2-F: slope={slope_EF:.3f}, intercept={intercept_EF:.3f}")
