import matplotlib.pyplot as plt
import numpy as np


# hangi action kac defa secilmis
def plot_action_histogram(action_indices, action_space):
    plt.figure(figsize=(10, 5))
    plt.get_current_fig_manager().set_window_title('Action Selection Frequency')
    counts, bins, patches = plt.hist(action_indices, bins=np.arange(len(action_space.actions) + 1) - 0.5,
                                     edgecolor='black')
    plt.xlabel('Action Index')
    plt.ylabel('Frequency')
    plt.title('Action Selection Frequency')
    plt.xticks(np.arange(len(action_space.actions)), ['Roll Low', 'Roll High', 'Pitch Low', 'Pitch High',
                                                      'Thrust Low', 'Thrust High', 'Yaw Low', 'Yaw High', 'No Action'])

    for count, patch in zip(counts, patches):
        height = patch.get_height()
        plt.text(patch.get_x() + patch.get_width() / 2, height + 0.1, str(int(count)), ha='center', va='bottom')

    plt.show()
