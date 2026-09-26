import json
import matplotlib.pyplot as plt
import numpy as np

with open("results.json") as f:
    all_results = json.load(f)

models = list(all_results.keys())
categories = sorted({cat for m in all_results.values() for cat in m["category_totals"]})

x = np.arange(len(categories))
width = 0.8 / len(models)

fig, ax = plt.subplots(figsize=(8, 5))
for i, model in enumerate(models):
    rates = []
    for cat in categories:
        counts = all_results[model]["category_totals"].get(cat, {"matching": 0, "total": 1})
        rates.append(counts["matching"] / counts["total"] * 100)
    ax.bar(x + i * width, rates, width, label=model)

ax.set_xlabel("Behavior category")
ax.set_ylabel("Concerning-behavior rate (%)")
ax.set_title("Alignment behavior eval: concerning-behavior rate by model")
ax.set_xticks(x + width * (len(models) - 1) / 2)
ax.set_xticklabels(categories)
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("results_chart.png", dpi=150)
print("Saved chart to results_chart.png")