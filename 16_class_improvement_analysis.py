# Cell 16: Analyze Which Classes Benefit Most from Fusion

print("\n" + "="*70)
print("ANALYZING WHICH CLASSES BENEFIT MOST FROM FUSION")
print("="*70 + "\n")

# Extract F1-scores per class
raw_f1_per_class = []
spec_f1_per_class = []
fusion_f1_per_class = []

for class_name in class_names:
    raw_f1_per_class.append(raw_report[class_name]['f1-score'])
    spec_f1_per_class.append(spec_report[class_name]['f1-score'])
    fusion_f1_per_class.append(fusion_report[class_name]['f1-score'])

# Calculate improvement
improvement_over_raw = np.array(fusion_f1_per_class) - np.array(raw_f1_per_class)
improvement_over_spec = np.array(fusion_f1_per_class) - np.array(spec_f1_per_class)
best_single_stream = np.maximum(raw_f1_per_class, spec_f1_per_class)
improvement_over_best = np.array(fusion_f1_per_class) - best_single_stream

# Create comparison table
print(f"{'Class':<15} {'Raw F1':<10} {'Spec F1':<10} {'Fusion F1':<12} {'Improvement':<12}")
print("="*70)

for i, class_name in enumerate(class_names):
    print(f"{class_name:<15} {raw_f1_per_class[i]:<10.4f} {spec_f1_per_class[i]:<10.4f} "
          f"{fusion_f1_per_class[i]:<12.4f} {improvement_over_best[i]:+.4f}")

print("\n")

# Visualize improvement
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(class_names))
width = 0.25

bars1 = ax.bar(x - width, raw_f1_per_class, width, label='RawOnly', alpha=0.8)
bars2 = ax.bar(x, spec_f1_per_class, width, label='SpecOnly', alpha=0.8)
bars3 = ax.bar(x + width, fusion_f1_per_class, width, label='Fusion', alpha=0.8)

ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.show()

# Identify classes that benefit most
most_improved_idx = np.argmax(improvement_over_best)
least_improved_idx = np.argmin(improvement_over_best)

print(f"\nClasses that benefit MOST from fusion:")
print(f"  {class_names[most_improved_idx]}: {improvement_over_best[most_improved_idx]:+.4f}")

print(f"\nClasses that benefit LEAST from fusion:")
print(f"  {class_names[least_improved_idx]}: {improvement_over_best[least_improved_idx]:+.4f}")
