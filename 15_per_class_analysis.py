# Cell 15: Per-Class Performance Analysis

# Analyze per-class performance for all models
print("\n" + "="*70)
print("PER-CLASS PERFORMANCE ANALYSIS")
print("="*70 + "\n")

print("RawOnlyModel:")
raw_report = analyze_per_class_performance(raw_labels, raw_preds, class_names, 'RawOnlyModel')

print("\nSpecOnlyModel:")
spec_report = analyze_per_class_performance(spec_labels, spec_preds, class_names, 'SpecOnlyModel')

print("\nTwoStreamFusionModel:")
fusion_report = analyze_per_class_performance(fusion_labels, fusion_preds, class_names, 'TwoStreamFusionModel')

# Print detailed classification reports
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORTS")
print("="*70 + "\n")

print("RawOnlyModel:")
print(classification_report(raw_labels, raw_preds, target_names=class_names, zero_division=0))

print("\nSpecOnlyModel:")
print(classification_report(spec_labels, spec_preds, target_names=class_names, zero_division=0))

print("\nTwoStreamFusionModel:")
print(classification_report(fusion_labels, fusion_preds, target_names=class_names, zero_division=0))
