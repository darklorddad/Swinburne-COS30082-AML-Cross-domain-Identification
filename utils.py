import os
import json
import pandas as pd
import matplotlib.pyplot as plt


# #############################################################################
# CORE LOGIC FROM UTILITY SCRIPTS
# #############################################################################

# --- From plot_metrics.py ---
def util_plot_training_metrics(json_path):
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    df = pd.DataFrame(data.get('log_history', []))
    if df.empty: raise ValueError("No 'log_history' found.")
    train_df = df[df['loss'].notna()].copy()
    eval_df = df[df['eval_loss'].notna()].copy()
    figures = {}
    # Plot Loss
    fig_loss, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Training vs. Evaluation Loss')
    if 'loss' in train_df: ax.plot(train_df['step'], train_df['loss'], label='Training loss', marker='o')
    if 'eval_loss' in eval_df: ax.plot(eval_df['step'], eval_df['eval_loss'], label='Evaluation loss', marker='x')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True); figures['Loss'] = fig_loss
    # Plot Accuracy
    fig_acc, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation Accuracy')
    if 'eval_accuracy' in eval_df: ax.plot(eval_df['step'], eval_df['eval_accuracy'], label='Evaluation accuracy', marker='o', color='g')
    ax.set_xlabel('Step'); ax.set_ylabel('Accuracy')
    ax.legend(); ax.grid(True); figures['Accuracy'] = fig_acc
    # Plot Learning Rate
    fig_lr, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Learning Rate Schedule')
    if 'learning_rate' in train_df: ax.plot(train_df['step'], train_df['learning_rate'], label='Learning rate', marker='o', color='r')
    ax.set_xlabel('Step'); ax.set_ylabel('Learning rate')
    ax.legend(); ax.grid(True); figures['Learning Rate'] = fig_lr
    # Plot Grad Norm
    fig_gn, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Gradient Norm')
    if 'grad_norm' in train_df: ax.plot(train_df['step'], train_df['grad_norm'], label='Grad norm', marker='o', color='purple')
    ax.set_xlabel('Step'); ax.set_ylabel('Gradient norm')
    ax.legend(); ax.grid(True); figures['Gradient Norm'] = fig_gn
    # Plot F1
    fig_f1, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation F1 Scores')
    if 'eval_f1_macro' in eval_df: ax.plot(eval_df['step'], eval_df['eval_f1_macro'], label='F1 macro', marker='o')
    if 'eval_f1_micro' in eval_df: ax.plot(eval_df['step'], eval_df['eval_f1_micro'], label='F1 micro', marker='x')
    if 'eval_f1_weighted' in eval_df: ax.plot(eval_df['step'], eval_df['eval_f1_weighted'], label='F1 weighted', marker='s')
    ax.set_xlabel('Step'); ax.set_ylabel('F1 score')
    ax.legend(); ax.grid(True); figures['F1 Scores'] = fig_f1
    # Plot Precision
    fig_prec, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation Precision Scores')
    if 'eval_precision_macro' in eval_df: ax.plot(eval_df['step'], eval_df['eval_precision_macro'], label='Precision macro', marker='o')
    if 'eval_precision_micro' in eval_df: ax.plot(eval_df['step'], eval_df['eval_precision_micro'], label='Precision micro', marker='x')
    if 'eval_precision_weighted' in eval_df: ax.plot(eval_df['step'], eval_df['eval_precision_weighted'], label='Precision weighted', marker='s')
    ax.set_xlabel('Step'); ax.set_ylabel('Precision')
    ax.legend(); ax.grid(True); figures['Precision'] = fig_prec
    # Plot Recall
    fig_recall, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation Recall Scores')
    if 'eval_recall_macro' in eval_df: ax.plot(eval_df['step'], eval_df['eval_recall_macro'], label='Recall macro', marker='o')
    if 'eval_recall_micro' in eval_df: ax.plot(eval_df['step'], eval_df['eval_recall_micro'], label='Recall micro', marker='x')
    if 'eval_recall_weighted' in eval_df: ax.plot(eval_df['step'], eval_df['eval_recall_weighted'], label='Recall weighted', marker='s')
    ax.set_xlabel('Step'); ax.set_ylabel('Recall')
    ax.legend(); ax.grid(True); figures['Recall'] = fig_recall
    # Plot Epoch
    fig_epoch, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Epoch Progression')
    if 'epoch' in df:
        epoch_df = df[['step', 'epoch']].dropna().drop_duplicates('step').sort_values('step')
        ax.plot(epoch_df['step'], epoch_df['epoch'], label='Epoch', marker='.')
    ax.set_xlabel('Step'); ax.set_ylabel('Epoch')
    ax.legend(); ax.grid(True); figures['Epoch'] = fig_epoch
    # Plot Eval Runtime
    fig_runtime, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation Runtime')
    if 'eval_runtime' in eval_df: ax.plot(eval_df['step'], eval_df['eval_runtime'], label='Eval runtime', marker='o')
    ax.set_xlabel('Step'); ax.set_ylabel('Runtime (s)')
    ax.legend(); ax.grid(True); figures['Eval Runtime'] = fig_runtime
    # Plot Eval Samples Per Second
    fig_sps, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation Samples Per Second')
    if 'eval_samples_per_second' in eval_df: ax.plot(eval_df['step'], eval_df['eval_samples_per_second'], label='Eval samples/sec', marker='o')
    ax.set_xlabel('Step'); ax.set_ylabel('Samples / second')
    ax.legend(); ax.grid(True); figures['Eval Samples/sec'] = fig_sps
    # Plot Eval Steps Per Second
    fig_steps_ps, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation Steps Per Second')
    if 'eval_steps_per_second' in eval_df: ax.plot(eval_df['step'], eval_df['eval_steps_per_second'], label='Eval steps/sec', marker='o')
    ax.set_xlabel('Step'); ax.set_ylabel('Steps / second')
    ax.legend(); ax.grid(True); figures['Eval Steps/sec'] = fig_steps_ps
    return figures

def util_save_training_metrics(json_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Save CSV
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    df = pd.DataFrame(data.get('log_history', []))
    if df.empty: raise ValueError("No 'log_history' found.")
    
    csv_path = os.path.join(save_dir, 'Training-metrics.csv')
    df.to_csv(csv_path, index=False)
    
    # 2. Generate and Save Plots
    figures = util_plot_training_metrics(json_path)
    
    saved_count = 1 # CSV
    for name, fig in figures.items():
        if fig:
            # Name format: Sentence case with dash, no space.
            # e.g. "Eval Steps/sec" -> "Eval-steps-sec.png"
            clean_name = name.replace('/', '-').replace(' ', '-')
            parts = clean_name.split('-')
            # Capitalize first part, lowercase rest
            formatted_parts = [parts[0].capitalize()] + [p.lower() for p in parts[1:]]
            filename = "-".join(formatted_parts) + ".png"
            
            path = os.path.join(save_dir, filename)
            fig.savefig(path)
            saved_count += 1
            plt.close(fig) # Close figure to free memory
            
    return f"Successfully saved metrics to {save_dir}\nSaved {saved_count} files (CSV + Plots)."
