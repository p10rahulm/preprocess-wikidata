import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt


def read_file(filename):
    list_of_elements = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines[1:]:
            line_elements = []
            temp_elements = [elem.strip() for elem in line.split("|||")]
            line_elements += [int(elem)
                              for elem in temp_elements[0].split(",")]
            forw_elems = temp_elements[1][1:-1]
            line_elements += [float(elem.strip()[1:-1])
                              for elem in forw_elems.split(",")]
            back_elems = temp_elements[2][1:-1]
            line_elements += [float(elem.strip()[1:-1])
                              for elem in back_elems.split(",")]
            print("temp_elements", temp_elements)
            print("line_elements", line_elements)
            list_of_elements.append(line_elements)

    out_df = pd.DataFrame(list_of_elements, columns=[
                          'sentence_num', 'k_tokens',
                          'forw_kprime=10', 'forw_kprime=25', 'forw_kprime=50', 'forw_kprime=100', 'forw_kprime=255',
                          'back_kprime=10', 'back_kprime=25', 'back_kprime=50', 'back_kprime=100', 'back_kprime=255'])
    return out_df


def reformat_df(df):
    max_k = max(df['k_tokens'])
    max_sentences = max(df['sentence_num'])
    print("max_k", max_k, "max_sentences", max_sentences)
    forward_entropy = np.zeros((max_sentences+1, max_k//5, 5))
    backward_entropy = np.zeros((max_sentences+1, max_k//5, 5))
    for i in range(len(df)):
        row = df.iloc[i, 0]
        col = df.iloc[i, 1]//5-1
        print("row", row, "col", col)
        forward_entropy[row, col, :] = df.iloc[i, 2:7]
        backward_entropy[row, col, :] = df.iloc[i, 7:]
    return forward_entropy, backward_entropy


def plot_entropy_with_k(forward_entropy, backward_entropy, filepath="outputs/boxplots_for_entropy.png", title_suffix=""):
    # Set the font to Times
    plt.rcParams['font.family'] = 'serif'
    # Labels for the x-axis
    k_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # Create a figure with two subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.boxplot(forward_entropy, labels=k_values, notch=True, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='ghostwhite'), medianprops=dict(color='mediumorchid'))
    ax2.boxplot(backward_entropy, labels=k_values, notch=True, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='mintcream'), medianprops=dict(color='mediumorchid'))

    # Create titles
    ax1.set_title('Forward Entropy ' + title_suffix,
                  fontsize=14, fontweight='bold')
    ax2.set_title('Backward Entropy ' + title_suffix,
                  fontsize=14, fontweight='bold')
    # Beautify the plots
    for ax in [ax1, ax2]:
        ax.set_xlabel('Number of Tokens in the Input Prompt', fontsize=12)
        ax.set_ylabel('Avg. Entropy per Token', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

    # Adjust the spacing between subplots
    plt.tight_layout()
    # Save the plot to a file with a specified filepath
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_entropy_with_kprime(forward_entropy, backward_entropy, filepath="outputs/boxplots_for_entropy_fixed_kprime.png", title_suffix=""):
    # Set the font to Times
    plt.rcParams['font.family'] = 'serif'
    # Labels for the x-axis
    k_values = [10,25,50,100,255]

    # Create a figure with two subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.boxplot(forward_entropy, labels=k_values, notch=True, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='mintcream'), medianprops=dict(color='steelblue'))
    ax2.boxplot(backward_entropy, labels=k_values, notch=True, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='azure'), medianprops=dict(color='steelblue'))

    # Create titles
    ax1.set_title('Forward Entropy ' + title_suffix,
                  fontsize=14, fontweight='bold')
    ax2.set_title('Backward Entropy ' + title_suffix,
                  fontsize=14, fontweight='bold')
    # Beautify the plots
    for ax in [ax1, ax2]:
        ax.set_xlabel('Number of Tokens in the Completion', fontsize=12)
        ax.set_ylabel('Avg. Entropy per Token', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

    # Adjust the spacing between subplots
    plt.tight_layout()
    # Save the plot to a file with a specified filepath
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    random.seed(8)
    start_time = time.time()

    filename = "inputs/entropy_vs_kkprime.txt"

    df = read_file(filename)
    print("df=", df)
    forward_entropy, backward_entropy = reformat_df(df)
    forward_entropy = forward_entropy[:-1]
    backward_entropy = backward_entropy[:-1]
    print("forward_entropy=", forward_entropy)
    print("backward_entropy=", backward_entropy)

    # Plot for kprime = 10
    filepath = "outputs/boxplots_for_entropy_for_kprime10.png"
    title_suffix = "with k for completion len = 10"
    plot_entropy_with_k(forward_entropy[:, :, 0], backward_entropy[:,
             :, 0], filepath=filepath, title_suffix=title_suffix)

    # Plot for kprime = 25
    filepath = "outputs/boxplots_for_entropy_for_kprime25.png"
    title_suffix = "with k for completion len = 25"
    plot_entropy_with_k(forward_entropy[:, :, 1], backward_entropy[:,
             :, 1], filepath=filepath, title_suffix=title_suffix)

    # Plot for kprime = 50
    filepath = "outputs/boxplots_for_entropy_for_kprime50.png"
    title_suffix = "with k for completion len = 50"
    plot_entropy_with_k(forward_entropy[:, :, 2], backward_entropy[:,
             :, 2], filepath=filepath, title_suffix=title_suffix)

    # Plot for kprime = 100
    filepath = "outputs/boxplots_for_entropy_for_kprime100.png"
    title_suffix = "with k for completion len = 100"
    plot_entropy_with_k(forward_entropy[:, :, 3], backward_entropy[:,
             :, 3], filepath=filepath, title_suffix=title_suffix)

    # Plot for kprime = 255
    filepath = "outputs/boxplots_for_entropy_for_kprime255.png"
    title_suffix = "with k for completion len = 255"
    plot_entropy_with_k(forward_entropy[:, :, 4], backward_entropy[:,
             :, 4], filepath=filepath, title_suffix=title_suffix)

    # We will now check for entropy with fixed input length
    # Plot for input length = 5
    filepath = "outputs/boxplots_for_entropy_for_k5.png"
    title_suffix = "with k for prompt length = 5"
    plot_entropy_with_kprime(forward_entropy[:, 0, :], backward_entropy[:,
             0, :], filepath=filepath, title_suffix=title_suffix)

    # Plot for input length = 10
    filepath = "outputs/boxplots_for_entropy_for_k10.png"
    title_suffix = "with k for prompt length = 10"
    plot_entropy_with_kprime(forward_entropy[:, 1, :], backward_entropy[:,
             1, :], filepath=filepath, title_suffix=title_suffix)

    # Plot for input length = 15
    filepath = "outputs/boxplots_for_entropy_for_k15.png"
    title_suffix = "with k for prompt length = 15"
    plot_entropy_with_kprime(forward_entropy[:, 2, :], backward_entropy[:,
             2, :], filepath=filepath, title_suffix=title_suffix)

    # Plot for input length = 20
    filepath = "outputs/boxplots_for_entropy_for_k20.png"
    title_suffix = "with k for prompt length = 20"
    plot_entropy_with_kprime(forward_entropy[:, 3, :], backward_entropy[:,
             3, :], filepath=filepath, title_suffix=title_suffix)

    # Plot for input length = 25
    filepath = "outputs/boxplots_for_entropy_for_k25.png"
    title_suffix = "with k for prompt length = 25"
    plot_entropy_with_kprime(forward_entropy[:, 4, :], backward_entropy[:,
             4, :], filepath=filepath, title_suffix=title_suffix)
    
    # Plot for input length = 30
    filepath = "outputs/boxplots_for_entropy_for_k30.png"
    title_suffix = "with k for prompt length = 30"
    plot_entropy_with_kprime(forward_entropy[:, 5, :], backward_entropy[:,
             5, :], filepath=filepath, title_suffix=title_suffix)

    # Plot for input length = 35
    filepath = "outputs/boxplots_for_entropy_for_k35.png"
    title_suffix = "with k for prompt length = 35"
    plot_entropy_with_kprime(forward_entropy[:, 6, :], backward_entropy[:,
             6, :], filepath=filepath, title_suffix=title_suffix)

    # Plot for input length = 40
    filepath = "outputs/boxplots_for_entropy_for_k40.png"
    title_suffix = "with k for prompt length = 40"
    plot_entropy_with_kprime(forward_entropy[:, 7, :], backward_entropy[:,
             7, :], filepath=filepath, title_suffix=title_suffix)

    # Plot for input length = 45
    filepath = "outputs/boxplots_for_entropy_for_k45.png"
    title_suffix = "with k for prompt length = 45"
    plot_entropy_with_kprime(forward_entropy[:, 8, :], backward_entropy[:,
             8, :], filepath=filepath, title_suffix=title_suffix)

    # Plot for input length = 50
    filepath = "outputs/boxplots_for_entropy_for_k50.png"
    title_suffix = "with k for prompt length = 50"
    plot_entropy_with_kprime(forward_entropy[:, 9, :], backward_entropy[:,
             9, :], filepath=filepath, title_suffix=title_suffix)

    # print("time taken = ", time.time() - start_time)
