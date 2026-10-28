import json
import numpy as np
import matplotlib.pyplot as plt
with open("carm_data.json", "r") as f:
    data = json.load(f)
p_max = data["p_max"]
b_l1 = data["b_l1"]
b_l2 = data["b_l2"]
b_l3 = data["b_l3"]
b_dram = data["b_dram"]
ai = np.logspace(-2, 3, 600)
carm_l1 = np.minimum(p_max, ai * b_l1)
carm_l2 = np.minimum(p_max, ai * b_l2)
carm_l3 = np.minimum(p_max, ai * b_l3)
carm_dram = np.minimum(p_max, ai * b_dram)
plt.figure(figsize=(11, 6.5))
plt.loglog(ai, carm_l1, color='#9C27B0', linestyle='-', linewidth=2.0, label=f'L1 Cache ({b_l1:.1f} GB/s)')
plt.loglog(ai, carm_l2, color='#2196F3', linestyle='-', linewidth=2.0, label=f'L2 Cache ({b_l2:.1f} GB/s)')
plt.loglog(ai, carm_l3, color='#4CAF50', linestyle='-', linewidth=2.0, label=f'L3 Cache ({b_l3:.1f} GB/s)')
plt.loglog(ai, carm_dram, color='#F44336', linestyle='-', linewidth=2.5, label=f'DRAM ({b_dram:.1f} GB/s)')
plt.axhline(p_max, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
plt.text(0.015, p_max * 1.08, f'Peak Compute (P_max): {p_max:.1f} GFLOPS', fontsize=11, weight='bold')
ridge_dram = p_max / b_dram
plt.axvline(ridge_dram, color='#F44336', linestyle=':', alpha=0.6)
plt.text(ridge_dram * 1.1, 1.5, f'DRAM Ridge:\n{ridge_dram:.1f} FLOPs/B', color='#D32F2F', fontsize=9)
benchmarks = {
    "STREAM Triad": (0.083, 3.7, '#9C27B0'),
    "Naive GEMM": (0.25, 12.4, '#FF9800'),
    "Tiled GEMM": (4.0, 110.0, '#3F51B5'),
    "AVX2 GEMM (Old)": (16.0, 245.0, '#E91E63'),
    "BLIS GEMM 6x16": (336.0, 510.8, '#00C853') 
}
for name, (k_ai, k_perf, col) in benchmarks.items():
    plt.scatter(k_ai, k_perf, color=col, s=80, zorder=6)
    plt.annotate(f"{name}\n({k_perf:.1f} GFLOPS)", 
                 (k_ai, k_perf), 
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, weight='bold')
plt.title('Cache-Aware Roofline Model (CARM)', fontsize=14, weight='bold')
plt.xlabel('Arithmetic Intensity [FLOPs / Byte]', fontsize=11)
plt.ylabel('Attainable Performance [GFLOPS]', fontsize=11)
plt.grid(True, which="both", ls="-", alpha=0.25)
plt.legend(loc='lower right', frameon=True, fontsize=10)
plt.xlim(0.01, 500)
plt.ylim(1.0, p_max * 2.2)
plt.tight_layout()
plt.savefig("carm_roofline.png", dpi=300)
print("Saved CARM chart to carm_roofline.png")
plt.show()