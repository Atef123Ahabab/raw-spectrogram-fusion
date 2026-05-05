# Cell 12: Evaluate All Models on Test Set

print("\n" + "="*70)
print("EVALUATING ALL MODELS ON TEST SET")
print("="*70 + "\n")

# Evaluate Raw-Only Model
print("Evaluating RawOnlyModel...")
_, raw_acc, raw_preds, raw_labels = evaluate(
    raw_only_model, test_loader, criterion, device, model_type='raw'
)
raw_f1 = f1_score(raw_labels, raw_preds, average='weighted')
print(f"RawOnlyModel - Test Accuracy: {raw_acc:.2f}%, F1-Score: {raw_f1:.4f}\n")

# Evaluate Spec-Only Model
print("Evaluating SpecOnlyModel...")
_, spec_acc, spec_preds, spec_labels = evaluate(
    spec_only_model, test_loader, criterion, device, model_type='spec'
)
spec_f1 = f1_score(spec_labels, spec_preds, average='weighted')
print(f"SpecOnlyModel - Test Accuracy: {spec_acc:.2f}%, F1-Score: {spec_f1:.4f}\n")

# Evaluate Fusion Model
print("Evaluating TwoStreamFusionModel...")
_, fusion_acc, fusion_preds, fusion_labels = evaluate(
    fusion_model, test_loader, criterion, device, model_type='fusion'
)
fusion_f1 = f1_score(fusion_labels, fusion_preds, average='weighted')
print(f"TwoStreamFusionModel - Test Accuracy: {fusion_acc:.2f}%, F1-Score: {fusion_f1:.4f}\n")

# Store results
results = {
    'RawOnlyModel': (raw_acc, raw_f1),
    'SpecOnlyModel': (spec_acc, spec_f1),
    'TwoStreamFusionModel': (fusion_acc, fusion_f1)
}
