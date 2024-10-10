# Cumulative Gaines Chart: https://chatgpt.com/c/b6213638-94d0-4e45-81d2-48a17d106bf1
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def cum_gaines(mdl, x_tst, y_tst, plot_name, save_plot=False):
    """calculate values for and display cumulative gaines chart for mdl"""
    # y_probs is an array of probability that the observation is positive
    y_probs = mdl.predict_proba(x_tst)[:, 1]  # Probabilities for the positive class

    # Create a dataframe with true values and probabilities
    results = pd.DataFrame({'true': y_tst, 'proba': y_probs})

    # Sort by predicted probabilities
    results = results.sort_values(by='proba', ascending=False)

    # Calculate the cumulative sum of true positives
    results['cumulative_true_positives'] = np.cumsum(results['true'])

    # Calculate the cumulative sum of all positives
    total_positives = results['true'].sum()

    # Calculate the cumulative gains
    results['cumulative_gains'] = results['cumulative_true_positives'] / total_positives

    # Calculate the baseline cumulative gains (for a random model)
    results['baseline'] = np.arange(1, len(results) + 1) / len(results)

    # Normalize the index for the percentage of the sample
    results['sample_percentage'] = np.arange(1, len(results) + 1) / len(results)

    # Plot the cumulative gains chart
    plt.figure(figsize=(10, 6))
    plt.plot(results['sample_percentage'], results['cumulative_gains'], label='Cumulative Gains', color='blue')
    plt.plot(results['sample_percentage'], results['baseline'], label='Baseline', color='red', linestyle='--')
    plt.xlabel('Percentage of Sample')
    plt.ylabel('Percentage of Target')
    plt.title('Baseline Cumulative Gains Chart')
    plt.title(plot_name)
    plt.legend()
    plt.grid(True)

    if save_plot:
        file_name = plot_name.replace(' ', '_')
        file_path = os.path.join('plots', file_name + '.png')
        plt.savefig(file_path, format='png')
        plt.close()
        print(f"Plot saved!")
    else:
        print("Don't save plot")
        plt.show()













