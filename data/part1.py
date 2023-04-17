import datasets

# start with a totally balanced split in 2 dimensions
N = 100
M = 100
dims = 2

# Part 1 A
# TODO: run the following lines.
#       Comment on the resulting plots
# dataset = datasets.generate_nd_dataset(N, M, datasets.kGaussian, dims)
# dataset.save_dataset_plot()

# # Part 1 B
# # TODO: Increase the dimensionality to 3.
dims = 3
gaus_dataset_points = datasets.generate_nd_dataset(N, M, datasets.kGaussian, dims).get_dataset()

# Draw and save 3D plot
def save_dataset_plot_3d(points):

    fig = datasets.plt.figure()
    ax = fig.add_subplot(projection='3d')

    zero_class = points[points[:,3] == 0]
    one_class = points[points[:,3] == 1]
    ax.scatter(zero_class[:,0], zero_class[:,1], zero_class[:,2], marker='x')
    ax.scatter(one_class[:,0], one_class[:,1], one_class[:,2], marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    datasets.plt.savefig('figs/{}.png'.format("Gaus_dim=3"))

save_dataset_plot_3d(gaus_dataset_points)

# Part 1 D
# TODO: go to random_control.py and set a random seed at your choosing. 
# Done!