# Cell 17: Generate Discussion Summary

print("\n" + "="*70)
print("PROJECT SUMMARY AND DISCUSSION POINTS")
print("="*70 + "\n")

print("1. PERFORMANCE COMPARISON")
print("-" * 70)
print(f"   RawOnlyModel:           Accuracy = {raw_acc:.2f}%, F1-Score = {raw_f1:.4f}")
print(f"   SpecOnlyModel:          Accuracy = {spec_acc:.2f}%, F1-Score = {spec_f1:.4f}")
print(f"   TwoStreamFusionModel:   Accuracy = {fusion_acc:.2f}%, F1-Score = {fusion_f1:.4f}")

print("\n2. DOES FUSION IMPROVE PERFORMANCE?")
print("-" * 70)
if fusion_acc > max(raw_acc, spec_acc):
    improvement = fusion_acc - max(raw_acc, spec_acc)
    print(f"   YES - Fusion improves accuracy by {improvement:.2f}%")
    print(f"   The two-stream fusion model outperforms both single-stream models.")
else:
    print(f"   NO - Fusion does not improve performance in this experiment.")
    print(f"   This may indicate that one representation already captures sufficient information,")
    print(f"   or that the fusion strategy needs refinement.")

print("\n3. WHY DOES FUSION HELP (OR NOT)?")
print("-" * 70)
print("   Possible explanations:")
print("   • Raw signals capture temporal patterns and waveform shapes")
print("   • Spectrograms capture frequency content and harmonic structure")
print("   • When fused, the model can learn from both representations")
print("   • Complementary features lead to more discriminative representations")
print("   • Late fusion allows each branch to specialize independently")

print("\n4. WHICH CLASSES BENEFIT MOST?")
print("-" * 70)
for i, class_name in enumerate(class_names):
    if improvement_over_best[i] > 0.01:
        print(f"   • {class_name}: +{improvement_over_best[i]:.4f} (benefits from fusion)")
    elif improvement_over_best[i] < -0.01:
        print(f"   • {class_name}: {improvement_over_best[i]:.4f} (performs worse with fusion)")
    else:
        print(f"   • {class_name}: {improvement_over_best[i]:.4f} (no significant change)")

print("\n5. KEY OBSERVATIONS")
print("-" * 70)
avg_improvement = np.mean(improvement_over_best)
print(f"   • Average per-class improvement: {avg_improvement:+.4f}")
print(f"   • Number of classes improved: {np.sum(improvement_over_best > 0)}/{len(class_names)}")
print(f"   • Maximum single improvement: {np.max(improvement_over_best):+.4f}")
print(f"   • Minimum single improvement: {np.min(improvement_over_best):+.4f}")

# Determine which single stream is stronger
if raw_acc > spec_acc:
    print(f"   • Raw signal branch is stronger (Acc: {raw_acc:.2f}% vs {spec_acc:.2f}%)")
else:
    print(f"   • Spectrogram branch is stronger (Acc: {spec_acc:.2f}% vs {raw_acc:.2f}%)")

print("\n6. RECOMMENDATIONS FOR REPORT")
print("-" * 70)
print("   Include in your report:")
print("   ✓ The performance comparison table above")
print("   ✓ All three confusion matrices")
print("   ✓ Training curves showing convergence")
print("   ✓ Per-class F1-score comparison")
print("   ✓ Discussion of why fusion helps (or doesn't)")
print("   ✓ Analysis of which classes benefit most and why")
print("   ✓ Potential improvements or future work")

print("\n" + "="*70)
print("END OF ANALYSIS")
print("="*70 + "\n")
