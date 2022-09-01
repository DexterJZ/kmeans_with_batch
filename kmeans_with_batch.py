import torch
import matplotlib.pyplot as plt

def kmeans_with_batch(points: torch.Tensor, k: int, n_iter: int, target: str):
    # cluster centers initialization
    n_points_per_cluster = points.shape[1] // k
    centers = points[:, ::n_points_per_cluster, :].clone()

    # cluster initialization
    clusters = torch.tensor([_ for _ in range(k)], dtype=int)
    clusters = clusters.repeat((points.shape[0], n_points_per_cluster))

    for _ in range(n_iter):
        # initialize min_dists for holding distances between points and
        # its cluster center
        min_dists = torch.full(points.shape[: -1], float('inf'), dtype=float)

        # update cluster members
        for c in range(k):
            # calculate distances between points and its cluster center
            diffs = points - centers[:, c : c+1, :]
            dists = torch.linalg.norm(diffs, dim=2, dtype=float)

            # find out points which need to be reassigned
            has_new = dists < min_dists

            # update min_dists
            min_dists = torch.where(has_new, dists, min_dists)

            # update cluster assignment
            clusters[has_new] = c

        nans = torch.full(points.shape, float('nan'), dtype=float)

        # update cluster centers
        for c in range(k):
            # get cluster memebers
            pos = clusters == c
            pos_1 = pos.unsqueeze(-1)
            pos_2 = pos_1.repeat((1, 1, 2))
            members = torch.where(pos_2, points, nans)

            # calculate centers for each dimension
            # if you have more than two dimensions, you can use a loop here
            dimension_centers = []

            for i in range(points.shape[2]):
                d = members[:, :, i : i + 1]
                dimension_centers.append(d.nansum(dim=1) / pos_1.sum(dim=1))

            centers[:, c] = torch.cat(dimension_centers, dim=1)

            # x = members[:, :, 0 : 1]
            # x_center = x.nansum(dim=1) / pos_1.sum(dim=1)
            # y = members[:, :, 1 : 2]
            # y_center = y.nansum(dim=1) / pos_1.sum(dim=1)
            # centers[:, c] = torch.cat([x_center, y_center], dim=1)

    # if centers are needed, here can we return them
    if target == 'centers':
        return centers, clusters
    # if we want to get the centroids (the points closest to the centers),
    # we need the following steps
    elif target == 'centroids':
        centroids_mask = torch.full(points.shape[: -1], False, dtype=bool)

        for c in range(k):
            # calculate distances
            diffs = points - centers[:, c : c+1, :]
            dists = torch.linalg.norm(diffs, dim=2, dtype=float)

            # get the member positions
            pos = clusters == c

            # set the distance of non-members equal to positive infinity
            dists = torch.where(pos, dists, float('inf'))

            centroids_idx = dists.argmin(dim=1)
            centroids_mask[torch.arange(centroids_mask.shape[0]), centroids_idx] = True
        
        centroids_mask = centroids_mask.unsqueeze(-1).repeat((1, 1, 2))
        centroids = torch.masked_select(points, centroids_mask).view((points.shape[0], k, points.shape[2]))

        return centroids, clusters
    else:
        raise(SyntaxError('Target is either "centers" or "centroids".'))


if __name__ == '__main__':
    torch.manual_seed(13)
    
    # points has three dimensions:
    # batch size
    # the number of points in a batch
    # point dimensions
    points = torch.rand((12, 36, 2), dtype=float)

    # call kmeans_in_batch to get centroids
    k = 6
    n_iter = 10
    centroids, clusters = kmeans_with_batch(points, k, n_iter, 'centroids')

    # we visulize the clustering result of the first data entry in the batch
    colors = ['b', 'c', 'y', 'g', 'm', 'r', 'k']

    for c in range(k):
        pos = clusters == c
        members = points[0, :, :][pos[0, :]]

        plt.scatter(members[:, 0], members[:, 1], c=colors[c])
    
    plt.scatter(centroids[0, :, 0], centroids[0, :, 1], c=colors[6], marker='x')
    plt.show()
