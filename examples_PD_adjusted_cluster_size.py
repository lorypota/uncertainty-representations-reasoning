import numpy as np
import matplotlib.pyplot as plt

# ---------- colors ----------
COL_C1 = "#E15759"      # orange/red  (small)
COL_C2 = "#4E79A7"      # blue        (large)
COL_TEST_PD  = "#FFC300"  # gold
COL_TEST_PDQ = "#2ECC71"  # green

# ---------- data ----------
def make_data(n1=100, n2=2000, mu1=(0,0), mu2=(2,2), sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    C1 = rng.normal(mu1, sigma, size=(n1, 2))
    C2 = rng.normal(mu2, sigma, size=(n2, 2))
    centers = np.array([mu1, mu2], dtype=float)
    return C1, C2, centers

# ---------- memberships ----------
def pd_prob(x, centers):
    d = np.linalg.norm(x - centers, axis=1)
    invd = 1.0 / np.maximum(d, 1e-12)
    return invd / invd.sum()

def pdq_prob(x, centers, q):
    d = np.linalg.norm(x - centers, axis=1)
    s = np.sqrt(q / np.maximum(d, 1e-12))  # p_k ∝ sqrt(q_k / d_k)
    return s / s.sum()

# ---------- figure A: joint scatter + marginals (counts) ----------
def plot_joint(C1, C2, test, test_color):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(4, 4, wspace=0.05, hspace=0.05)
    ax_sc   = fig.add_subplot(gs[1:, :3])
    ax_top  = fig.add_subplot(gs[0,  :3], sharex=ax_sc)
    ax_right= fig.add_subplot(gs[1:, 3],  sharey=ax_sc)

    # scatter
    ax_sc.scatter(C1[:,0], C1[:,1], s=12, alpha=0.6, color=COL_C1, label="C1 (small)")
    ax_sc.scatter(C2[:,0], C2[:,1], s=6,  alpha=0.25, color=COL_C2, label="C2 (large)")
    ax_sc.scatter(test[0], test[1], s=140, marker='*', edgecolor='k',
                  facecolor=test_color, zorder=5, label="Test point")
    ax_sc.legend(loc="upper left", frameon=False)
    ax_sc.grid(True, alpha=0.2)

    # top histogram (x) — raw counts
    bins_x = 40
    ax_top.hist(C1[:,0], bins=bins_x, color=COL_C1, alpha=0.5, density=False)
    ax_top.hist(C2[:,0], bins=bins_x, color=COL_C2, alpha=0.5, density=False)
    ax_top.axvline(test[0], color=test_color, linestyle="--", alpha=0.9)
    ax_top.axis('off')

    # right histogram (y) — raw counts
    bins_y = 40
    ax_right.hist(C1[:,1], bins=bins_y, color=COL_C1, alpha=0.5,
                  density=False, orientation='horizontal')
    ax_right.hist(C2[:,1], bins=bins_y, color=COL_C2, alpha=0.5,
                  density=False, orientation='horizontal')
    ax_right.axhline(test[1], color=test_color, linestyle="--", alpha=0.9)
    ax_right.axis('off')

    plt.show()

# ---------- figure B: decision regions (background) + points ----------
def plot_regions(C1, C2, centers, scorer, test, test_color):
    xlim, ylim, res = (-4, 6), (-4, 6), 250
    xs = np.linspace(*xlim, res); ys = np.linspace(*ylim, res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)

    for i in range(res):
        for j in range(res):
            Z[i, j] = scorer(np.array([X[i, j], Y[i, j]]))[0]  # prob of C1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 21), cmap='coolwarm', alpha=0.45)
    ax.contour(X, Y, Z, levels=[0.5], colors='k', linewidths=1.2)

    # points for context
    ax.scatter(C1[:,0], C1[:,1], s=12, alpha=0.55, color=COL_C1, label="C1 points")
    ax.scatter(C2[:,0], C2[:,1], s=6,  alpha=0.25, color=COL_C2, label="C2 points")

    # centres + test
    ax.scatter(centers[0,0], centers[0,1], s=90, marker='s', color=COL_C1, edgecolor='k', label='C1 centre')
    ax.scatter(centers[1,0], centers[1,1], s=90, marker='s', color=COL_C2, edgecolor='k', label='C2 centre')
    ax.scatter(test[0], test[1], s=170, marker='*', edgecolor='k', facecolor=test_color, label='Test point')

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', frameon=False)
    plt.show()

# ---------- run sets ----------
def run_pd_plots():
    C1, C2, centers = make_data(n1=100, n2=2000, mu1=(0,0), mu2=(2,2), sigma=1.0, seed=0)
    test = np.array([1.0, 1.0])
    plot_joint(C1, C2, test, COL_TEST_PD)
    plot_regions(C1, C2, centers, lambda x: pd_prob(x, centers), test, COL_TEST_PD)
    
    # Compute PD membership for the test point (ignores cluster size)
    p_pd = pd_prob(test, centers)
    print(f"PD (size ignored) p(C1), p(C2): {p_pd[0]:.3f}, {p_pd[1]:.3f}")

def run_pdq_plots():
    C1, C2, centers = make_data(n1=100, n2=2000, mu1=(0,0), mu2=(2,2), sigma=1.0, seed=0)
    q = np.array([len(C1), len(C2)], dtype=float)  # known sizes
    test = np.array([1.0, 1.0])
    plot_joint(C1, C2, test, COL_TEST_PDQ)
    plot_regions(C1, C2, centers, lambda x: pdq_prob(x, centers, q), test, COL_TEST_PDQ)
    
    # Compute PDQ membership for the test point (uses true cluster sizes)
    p_pdq = pdq_prob(test, centers, q)
    print(f"PDQ (size handled) p(C1), p(C2): {p_pdq[0]:.3f}, {p_pdq[1]:.3f}")

if __name__ == "__main__":
    run_pd_plots()
    run_pdq_plots()