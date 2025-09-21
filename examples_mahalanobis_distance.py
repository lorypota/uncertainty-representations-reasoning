import numpy as np
import matplotlib.pyplot as plt

def example_1():
    """
    R² PD example: run PD clustering first,
    then plot final PD decision regions and distances from the test point.
    """
    np.random.seed(0)

    # --- Data: two balanced clusters ----------------------------------------
    C1_points = 0.1 * np.random.randn(500, 2)          # dense cluster near (0,0)
    C2_points = 8 + 1.5 * np.random.randn(500, 2)      # wide cluster near (8,8)
    points  = np.vstack([C1_points, C2_points])
    init_centers = np.array([[0.0, 0.0], [8.0, 8.0]])  # starting guess
    test_point   = np.array([[4.0, 4.0]])

    # --- PD formula ----------------------------------------------------------
    def pd_prob(x, centers):
        d = np.linalg.norm(x - centers, axis=1)
        invd = 1 / np.maximum(d, 1e-10)
        return invd / invd.sum()

    # --- PD clustering loop --------------------------------------------------
    def run_pd(points, centers, tol=1e-4, max_iter=100):
        history = [centers.copy()]
        for it in range(max_iter):
            new_centers = np.zeros_like(centers)
            for k in range(len(centers)):
                weights = []
                weighted_points = []
                for x in points:
                    d = np.linalg.norm(x - centers, axis=1)
                    invd = 1 / np.maximum(d, 1e-10)
                    p = invd / invd.sum()
                    w = (p[k] ** 2) / max(d[k], 1e-10)
                    weights.append(w)
                    weighted_points.append(w * x)
                weights = np.array(weights)
                weighted_points = np.array(weighted_points)
                new_centers[k] = weighted_points.sum(axis=0) / weights.sum()
            move = np.sum(np.linalg.norm(new_centers - centers, axis=1))
            history.append(new_centers.copy())
            centers = new_centers
            print(f"Iteration {it+1}: centres = {centers}")
            if move < tol:
                break
        return centers, history

    # --- Run PD clustering before plotting -----------------------------------
    final_centers, history = run_pd(points, init_centers)

    print("\n--- Final PD centres ---")
    for i, c in enumerate(final_centers):
        print(f"C{i+1}: {c}")

    # PD membership of test point with final centres
    p_pd = pd_prob(test_point, final_centers)
    print(f"\nTest point (4,4) PD membership: {p_pd[0]:.1%} C1 vs {p_pd[1]:.1%} C2")

    # --- Compute PD decision map ---------------------------------------------
    xs = np.linspace(-1, 14, 200)
    ys = np.linspace(-1, 14, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            d = [np.hypot(X[i,j]-c[0], Y[i,j]-c[1]) for c in final_centers]
            invd = [1 / max(val, 1e-10) for val in d]
            Z[i,j] = invd[0] / (invd[0] + invd[1])

    # --- Plot ----------------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, Z, levels=np.linspace(0, 1, 21), cmap='coolwarm', alpha=0.5)
    plt.scatter(points[:, 0], points[:, 1], c='gray', s=15, alpha=0.4, label='Data')

    # final centres with labels
    plt.scatter(final_centers[:, 0], final_centers[:, 1],
                c=['red', 'blue'], s=50, marker='s', edgecolor='black', zorder=4)
    plt.annotate('C1', final_centers[0] + [0.3, 0.3], color='red',
                 weight='bold', fontsize=12)
    plt.annotate('C2', final_centers[1] + [0.3, 0.3], color='blue',
                 weight='bold', fontsize=12)

    # test point
    plt.scatter(test_point[:, 0], test_point[:, 1],
                c='gold', s=180, marker='*', edgecolor='black',
                zorder=5, label='Test point (4,4)')

    # dashed lines from test point to final centres to show distances
    for c, col in zip(final_centers, ['red', 'blue']):
        plt.plot([test_point[0, 0], c[0]],
                 [test_point[0, 1], c[1]],
                 linestyle='--', color=col, alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    

def example_2():
    """
    PD example using cluster–wise Mahalanobis distance.
    Shows how PD already accounts for different spread.
    """
    np.random.seed(0)

    # --- Data --------------------------------------------------------------
    C1_points = 0.1 * np.random.randn(500, 2)
    C2_points = 8 + 1.5 * np.random.randn(500, 2)
    points  = np.vstack([C1_points, C2_points])
    init_centers = np.array([[0., 0.], [8., 8.]])
    test_point   = np.array([[4., 4.]])

    # --- PD formula with Mahalanobis distance -------------------------------
    def pd_prob(x, centers, covs):
        d = []
        for k in range(len(centers)):
            diff = x - centers[k]
            inv_cov = np.linalg.inv(covs[k])
            d.append(np.sqrt(diff @ inv_cov @ diff))
        invd = 1 / np.maximum(d, 1e-10)
        return invd / invd.sum()

    def run_pd(points, centers, tol=1e-4, max_iter=100):
        # initialise covariances as isotropic
        covs = [np.eye(2)] * len(centers)
        for it in range(max_iter):
            new_centers = np.zeros_like(centers)
            new_covs    = []
            for k in range(len(centers)):
                weights = []
                weighted_points = []
                for x in points:
                    p = pd_prob(x, centers, covs)
                    diff = x - centers[k]
                    d = np.sqrt(diff @ np.linalg.inv(covs[k]) @ diff)
                    w = (p[k] ** 2) / max(d, 1e-10)
                    weights.append(w)
                    weighted_points.append(w * x)
                weights = np.array(weights)
                weighted_points = np.array(weighted_points)
                new_centers[k] = weighted_points.sum(axis=0) / weights.sum()
                # weighted covariance for Mahalanobis metric
                diffs = points - new_centers[k]
                W = weights[:, None]
                new_cov = (W * diffs).T @ diffs / np.maximum(weights.sum(), 1e-10)
                new_covs.append(new_cov + 1e-6 * np.eye(2))  # stabilise
            move = np.sum(np.linalg.norm(new_centers - centers, axis=1))
            centers, covs = new_centers, new_covs
            if move < tol:
                break
        return centers, covs

    # --- Run and report -----------------------------------------------------
    final_centers, final_covs = run_pd(points, init_centers)
    print("\n--- Final PD (Mahalanobis) centres ---")
    for i, c in enumerate(final_centers):
        print(f"C{i+1}: {c}")

    p_pd = pd_prob(test_point[0], final_centers, final_covs)
    print(f"\nTest point (4,4) PD membership: {p_pd[0]:.1%} C1 vs {p_pd[1]:.1%} C2")

    # --- Plot decision map --------------------------------------------------
    xs = np.linspace(-1, 14, 200)
    ys = np.linspace(-1, 14, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = pd_prob(np.array([X[i,j], Y[i,j]]), final_centers, final_covs)[0]

    plt.figure(figsize=(8,8))
    plt.contourf(X, Y, Z, levels=np.linspace(0,1,21), cmap='coolwarm', alpha=0.5)
    plt.scatter(points[:,0], points[:,1], c='gray', s=15, alpha=0.4)
    plt.scatter(final_centers[:,0], final_centers[:,1],
                c=['red','blue'], s=50, marker='s', edgecolor='black')
    plt.scatter(test_point[:,0], test_point[:,1],
                c='gold', s=180, marker='*', edgecolor='black')
    plt.title("PD with Mahalanobis distance")
    plt.tight_layout()
    plt.show()


def main():
    # without Mahalanobis distance
    example_1()
    
    # Mahalanobis distance
    example_2()
    

if __name__ == "__main__":
    main()
