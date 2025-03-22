import subprocess
import re
from statistics import mean
from itertools import product

# 定义参数组合
configs = ["modelnet40", "scanobjectnn", "objaverse"]
seeds = [1, 2, 3]
shots = [1, 2, 4, 8, 16]

# 结果存储结构
results = {cfg: {shot: [] for shot in shots} for cfg in configs}

# 正则表达式提取准确率（假设输出中有"TIMO: 0.95"格式）
pattern = re.compile(r"TIMO: (\d+\.\d+)")

for config, shot, seed in product(configs, shots, seeds):
    cmd = f"python main_few_shots.py --config configs/{config}.yaml --seed {seed} --shot {shot}"
    print(f"Running: {cmd}")

    # 执行命令并捕获输出
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # 解析输出中的准确率
    match = pattern.search(result.stdout)
    if match:
        acc = float(match.group(1))
        results[config][shot].append(acc)
        print(f"Config: {config}, Shot: {shot}, Seed: {seed}, Accuracy: {acc}")
    else:
        print(f"Error parsing output for {config} {shot} {seed}")

# 计算平均值并输出结果
for config in configs:
    print(f"\nResults for {config}:")
    for shot in shots:
        avg = mean(results[config][shot]) if results[config][shot] else 0
        print(f"Shot {shot}: Average Accuracy = {avg:.4f}")
