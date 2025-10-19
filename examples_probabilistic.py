import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define 8 data points - using clean coordinates
points = np.array([
    # Points near (1, 1)
    [1, 0.5], [0.5, 1], 
    # Points near (4, 3)
    [3.5, 3], [4, 2.5], [4.5, 3],
    # Points near (1.5, 3.5)
    [2, 3.5], [1, 3.5],
    # One "between" point for contrast
    [2.5, 2]
])

def initialize_centers_positioned(random_state=None):
    """Initialize centers with specific positioning: C1 top-left, C2 right, C3 bottom-left"""
    if random_state is not None:
        np.random.seed(random_state)
    
    # Set specific regions for each cluster
    centers = np.array([
        [0.8 + np.random.uniform(-0.2, 0.2), 3.2 + np.random.uniform(-0.2, 0.2)],  # C1: top-left
        [4.2 + np.random.uniform(-0.2, 0.2), 2.8 + np.random.uniform(-0.2, 0.2)],  # C2: right
        [1.0 + np.random.uniform(-0.2, 0.2), 0.8 + np.random.uniform(-0.2, 0.2)]   # C3: bottom-left
    ])
    
    return centers


def plot_initial_state(points, centers):
    """Plot initial centers with data points"""
    plt.figure(figsize=(8, 6))
    plt.scatter(centers[:, 0], centers[:, 1], c=['red', 'blue', 'green'], 
               s=200, marker='s', label='Initial Centers', edgecolor='black', alpha=0.7)
    plt.scatter(points[:, 0], points[:, 1], c='black', s=60, 
               label='Data Points', alpha=0.7)

    for i, (x, y) in enumerate(centers):
        plt.annotate(f'C{i+1}', (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=14, fontweight='bold')

    for i, (x, y) in enumerate(points):
        plt.annotate(f'P{i+1}', (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.xlim(-0.2, 5.2)
    plt.ylim(-0.2, 4.2)
    plt.xticks(np.arange(0, 5.5, 0.5))
    plt.yticks(np.arange(0, 4.5, 0.5))
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_algorithm(points, initial_centers, max_iterations=100, tolerance=1e-4):
    """Run the probabilistic clustering algorithm"""
    current_centers = initial_centers.copy().astype(float)
    centers_history = [current_centers.copy()]
    
    for iteration in range(max_iterations):
        new_centers = np.zeros_like(current_centers)
        
        for k in range(len(current_centers)):
            weights = []
            weighted_points = []
            
            for point in points:
                distances = [np.linalg.norm(point - center) for center in current_centers]
                inv_distances = [1/max(d, 1e-10) for d in distances]
                sum_inv = sum(inv_distances)
                probabilities = [inv_d / sum_inv for inv_d in inv_distances]
                
                prob_k = probabilities[k]
                dist_k = max(distances[k], 1e-10)
                weight_k = (prob_k ** 2) / dist_k
                
                weights.append(weight_k)
                weighted_points.append(weight_k * point)
            
            sum_weights = sum(weights)
            new_centers[k] = np.sum(weighted_points, axis=0) / sum_weights
        
        center_movements = [np.linalg.norm(new_centers[k] - current_centers[k]) 
                          for k in range(len(current_centers))]
        total_movement = sum(center_movements)
        
        centers_history.append(new_centers.copy())
        current_centers = new_centers.copy()
        
        if total_movement < tolerance:
            break
    
    return current_centers, centers_history, iteration + 1


def calculate_final_probabilities(points, centers):
    """Calculate final membership probabilities"""
    probabilities_matrix = []
    
    for point in points:
        distances = [np.linalg.norm(point - center) for center in centers]
        inv_distances = [1/max(d, 1e-10) for d in distances]
        sum_inv = sum(inv_distances)
        probabilities = [inv_d / sum_inv for inv_d in inv_distances]
        probabilities_matrix.append(probabilities)
    
    return np.array(probabilities_matrix)


def find_collapsed_centers(points, centers, threshold=0.05):
    """Find which centers have collapsed onto points"""
    collapsed = {}  # {center_idx: point_idx}
    
    for k, center in enumerate(centers):
        for i, point in enumerate(points):
            if np.linalg.norm(center - point) < threshold:
                collapsed[k] = i
                break
    
    return collapsed


def latex_final_probabilities_table(points, centers):
    """Generate LaTeX table for final probabilities"""
    probs = calculate_final_probabilities(points, centers)
    
    print(r"\begin{table}[h!]")
    print(r"\centering")
    print(r"\begin{tabular}{c|ccc}")
    print(r"\textbf{Point} & $p_1$ & $p_2$ & $p_3$ \\")
    print(r"\hline")
    
    for i, prob_row in enumerate(probs):
        print(f"$\\mathbf{{x}}_{{{i+1}}}$ & {prob_row[0]:.3f} & {prob_row[1]:.3f} & {prob_row[2]:.3f} \\\\")
    
    print(r"\end{tabular}")
    print(r"\caption{Final membership probabilities after convergence.}")
    print(r"\label{tab:final_probabilities}")
    print(r"\end{table}")
    print()


def plot_jdf_contours_final(centers, points, probabilities):
    """Plot JDF contours with final centers and probabilities"""
    x = np.linspace(-0.5, 5.5, 100)
    y = np.linspace(-0.5, 4.5, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            distances = [np.linalg.norm(point - center) for center in centers]
            d1, d2, d3 = [max(d, 1e-10) for d in distances]
            
            numerator = d1 * d2 * d3
            denominator = d1*d2 + d1*d3 + d2*d3
            Z[i, j] = numerator / denominator
    
    plt.figure(figsize=(10, 8))
    
    contour_levels = np.linspace(Z.min(), Z.max(), 20)
    contours = plt.contour(X, Y, Z, levels=contour_levels, colors='black', linewidths=0.8)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.3f')
    
    # Find collapsed centers
    collapsed = find_collapsed_centers(points, centers)
    
    # Get set of point indices that have centers collapsed on them
    collapsed_point_indices = set(collapsed.values())
    
    plt.scatter(centers[:, 0], centers[:, 1], c=['red', 'blue', 'green'], 
               s=200, marker='s', edgecolor='black', zorder=5)
    
    # Label centers (with collapse info if applicable)
    for k, (x, y) in enumerate(centers):
        if k in collapsed:
            label = f'C{k+1} = P{collapsed[k]+1}'
            # Use same positioning as points for collapsed centers
            offset = (8, 8)
        else:
            label = f'C{k+1}'
            # Use default positioning for non-collapsed centers
            offset = (-15, -15)
        
        plt.annotate(label, (x, y), xytext=offset, 
                    textcoords='offset points', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.scatter(points[:, 0], points[:, 1], c='black', s=60, 
               alpha=0.8, zorder=4)
    
    # Label points only if they don't have a center collapsed on them
    for i, point in enumerate(points):
        if i not in collapsed_point_indices:
            plt.annotate(f'P{i+1}', point, xytext=(8, 8), 
                        textcoords='offset points', fontsize=15, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.xlim(-0.5, 5.5)
    plt.ylim(-0.5, 4.5)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(-0.5, 6, 0.5))
    plt.yticks(np.arange(-0.5, 5, 0.5))
    plt.tight_layout()
    plt.show()


def plot_algorithm_convergence(points, centers_history):
    """Plot the convergence of the algorithm showing center movement"""
    plt.figure(figsize=(10, 8))
    
    plt.scatter(points[:, 0], points[:, 1], c='black', s=60, 
               alpha=0.7, zorder=3)
    
    for i, (x, y) in enumerate(points):
        plt.annotate(f'P{i+1}', (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12)
    
    colors = ['red', 'blue', 'green']
    
    # Find collapsed centers for final state
    final_centers = centers_history[-1]
    collapsed = find_collapsed_centers(points, final_centers)
    
    for k in range(3):
        path_x = [centers_history[i][k][0] for i in range(len(centers_history))]
        path_y = [centers_history[i][k][1] for i in range(len(centers_history))]
        
        plt.scatter(path_x[0], path_y[0], c=colors[k], s=200, marker='s', 
                   edgecolor='black', linewidth=2, alpha=0.5, zorder=4)
        
        plt.scatter(path_x[-1], path_y[-1], c=colors[k], s=200, marker='o', 
                   edgecolor='black', linewidth=2, zorder=5)
        
        if len(path_x) > 1:
            plt.plot(path_x, path_y, color=colors[k], linestyle='--', 
                    linewidth=2, alpha=0.8, zorder=2)
            
            for i in range(0, len(path_x) - 1, max(1, len(path_x)//5)):
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    plt.arrow(path_x[i], path_y[i], dx*0.8, dy*0.8, 
                             head_width=0.08, head_length=0.08, 
                             fc=colors[k], ec=colors[k], alpha=0.6, zorder=2)
        
        # Label with collapse info if applicable
        if k in collapsed:
            label = f'C{k+1} = P{collapsed[k]+1}'
        else:
            label = f'C{k+1}'
            
        plt.annotate(label, (path_x[-1], path_y[-1]), xytext=(8, 8), 
                    textcoords='offset points', fontsize=14, fontweight='bold',
                    color=colors[k], bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.2, 5.2)
    plt.ylim(-0.2, 4.2)
    plt.xticks(np.arange(0, 5.5, 0.5))
    plt.yticks(np.arange(0, 4.5, 0.5))
    plt.tight_layout()
    plt.show()


def main():
    # Example 1
    initial_centers = initialize_centers_positioned(random_state=42)
    plot_initial_state(points, initial_centers)
    
    # Example 2
    final_centers, centers_history, num_iterations = run_algorithm(points, initial_centers)
    plot_algorithm_convergence(points, centers_history)
    
    # Example 3
    final_probabilities = calculate_final_probabilities(points, final_centers)
    latex_final_probabilities_table(points, final_centers)
    
    # Example 4
    plot_jdf_contours_final(final_centers, points, final_probabilities)


main()