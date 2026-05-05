# Cell 14: Confusion Matrices

# Plot confusion matrix for Raw-Only Model
print("Confusion Matrix: RawOnlyModel")
cm_raw = plot_confusion_matrix(raw_labels, raw_preds, class_names, 
                                title='Confusion Matrix - RawOnlyModel')

# Plot confusion matrix for Spec-Only Model
print("\nConfusion Matrix: SpecOnlyModel")
cm_spec = plot_confusion_matrix(spec_labels, spec_preds, class_names, 
                                 title='Confusion Matrix - SpecOnlyModel')

# Plot confusion matrix for Fusion Model
print("\nConfusion Matrix: TwoStreamFusionModel")
cm_fusion = plot_confusion_matrix(fusion_labels, fusion_preds, class_names, 
                                   title='Confusion Matrix - TwoStreamFusionModel')
