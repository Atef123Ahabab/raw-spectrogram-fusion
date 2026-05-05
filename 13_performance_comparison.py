# Cell 13: Performance Comparison

# Display comparison table and plots
compare_model_performance(results)

# Calculate improvement
raw_to_fusion = fusion_acc - raw_acc
spec_to_fusion = fusion_acc - spec_acc

print("\nImprovement Analysis:")
print(f"Fusion vs. Raw-Only: {raw_to_fusion:+.2f}%")
print(f"Fusion vs. Spec-Only: {spec_to_fusion:+.2f}%")

if fusion_acc > max(raw_acc, spec_acc):
    print("\n✓ Feature fusion improves performance!")
    print(f"  Best single-stream: {max(raw_acc, spec_acc):.2f}%")
    print(f"  Fusion model: {fusion_acc:.2f}%")
    print(f"  Improvement: {fusion_acc - max(raw_acc, spec_acc):.2f}%")
else:
    print("\n✗ Feature fusion does not improve performance in this case.")
