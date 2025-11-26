import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import os

def compare_results():
    """
    Scans for training history CSVs and plots a comparison bar chart.
    """
    # Find all history files
    history_files = glob.glob("history_*.csv")
    
    if not history_files:
        print("No history CSV files found. Run training first!")
        return

    results = []

    for file in history_files:
        try:
            # Filename format expected: history_{MODEL}_{MODE}.csv
            # Example: history_resnet50_all.csv
            parts = os.path.splitext(file)[0].split('_')
            if len(parts) >= 3:
                model_name = parts[1]
                data_mode = parts[2]
                
                # Read the final validation accuracy
                df = pd.read_csv(file)
                if not df.empty and 'val_acc' in df.columns:
                    final_acc = df['val_acc'].iloc[-1]
                    results.append({
                        'Model': model_name,
                        'Data Mode': data_mode,
                        'Accuracy (%)': final_acc
                    })
        except Exception as e:
            print(f"Skipping file {file} due to error: {e}")

    if not results:
        print("No valid results found in CSV files.")
        return

    # Create DataFrame for plotting
    df_results = pd.DataFrame(results)
    
    # Drop duplicates to ensure we only have one entry per Model+Mode combination
    # This handles cases where you might have multiple files or backup copies
    df_results = df_results.drop_duplicates(subset=['Model', 'Data Mode'], keep='last')
    
    # Print Table
    print("\n--- Final Results Summary ---")
    print(df_results.to_string(index=False)) # Cleaner print without index
    
    # Plot Grouped Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_results, x='Model', y='Accuracy (%)', hue='Data Mode', palette='viridis')
    
    plt.title('Model Performance Comparison across Domains')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Dataset')
    
    # Add value labels on top of bars
    for p in plt.gca().patches:
        if p.get_height() > 0:
            plt.gca().annotate(f'{p.get_height():.1f}%', 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha = 'center', va = 'center', 
                               xytext = (0, 9), 
                               textcoords = 'offset points')

    save_path = 'comparison_results.png'
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nComparison chart saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    compare_results()

