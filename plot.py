class Plot:
    @staticmethod
    def plot_codes(ax, codes, labels):
        ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
        ax.set_aspect('equal')
        ax.set_xlim(codes.min() - .1, codes.max() + .1)
        ax.set_ylim(codes.min() - .1, codes.max() + .1)
        ax.tick_params(
            axis='both', which='both', left='off', bottom='off',
            labelleft='off', labelbottom='off')

    @staticmethod
    def plot_samples(ax, samples):
        for index, sample in enumerate(samples):
            ax[index].imshow(sample, cmap='gray')
            ax[index].axis('off')
