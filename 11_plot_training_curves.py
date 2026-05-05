# Cell 11: Plot Training Curves

# Compare training histories
histories = [raw_history, spec_history, fusion_history]
model_names = ['RawOnly', 'SpecOnly', 'Fusion']

plot_training_history(histories, model_names)
