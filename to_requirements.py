import os

# 检查 conda_list.txt 文件是否存在
if not os.path.exists("conda_list.txt"):
    print("Error: conda_list.txt not found in the current directory.")
    exit(1)

# 读取 conda_list.txt 文件，指定编码为 utf-16
with open("conda_list.txt", "r", encoding="utf-16") as file:
    lines = file.readlines()

# 打印读取到的行数（调试用）
print(f"Total lines read: {len(lines)}")

# 初始化两个列表，分别存储 pip 和 conda-forge 包
pip_packages = []
conda_forge_packages = []

# 处理每一行
for line in lines[2:]:  # 跳过前两行（表头）
    parts = line.split()
    if len(parts) >= 4:  # 确保有包名、版本号和 Channel 信息
        package_name = parts[0]
        package_version = parts[1]
        channel = parts[3]  # 第四列是 Channel

        # 根据 Channel 分类
        if channel == "pypi":
            pip_packages.append(f"{package_name}=={package_version}")
        elif channel == "conda-forge":
            conda_forge_packages.append(f"{package_name}=={package_version}")
    else:
        print(f"Skipping line (invalid format): {line.strip()}")  # 打印被跳过的行（调试用）

# 将 pip 包写入 requirements_pip.txt
with open("requirements_pip.txt", "w", encoding="utf-16") as file:
    file.write("\n".join(pip_packages))

# 将 conda-forge 包写入 requirements_conda.txt
with open("requirements_conda.txt", "w", encoding="utf-16") as file:
    file.write("\n".join(conda_forge_packages))

# 提示完成
print("requirements_pip.txt and requirements_conda.txt have been generated.")