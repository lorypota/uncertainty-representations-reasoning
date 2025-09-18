import matplotlib.pyplot as plt
import numpy as np

# Define 3 cluster centers
centers = np.array([[1, 1], [4, 3], [1.5, 3.5]])
center_labels = ['C1', 'C2', 'C3']

# Define 8 data points - using clean coordinates
points = np.array([
    # Points near C1 (1, 1)
    [1, 0.5], [0.5, 1], 
    # Points near C2 (4, 3)
    [3.5, 3], [4, 2.5], [4.5, 3],
    # Points near C3 (2, 4)
    [2, 3.5], [1, 3.5],
    # One "between" point for contrast
    [2.5, 2]
])

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(centers[:, 0], centers[:, 1], c=['red', 'blue', 'green'], 
           s=200, marker='s', label='Cluster Centers', edgecolor='black')
plt.scatter(points[:, 0], points[:, 1], c='black', s=60, 
           label='Data Points', alpha=0.7)

# Label centers and points
for i, (x, y) in enumerate(centers):
    plt.annotate(f'C{[i+1]}', (x, y), xytext=(5, 5), 
                textcoords='offset points', fontsize=12, fontweight='bold')

for i, (x, y) in enumerate(points):
    plt.annotate(f'P{i+1}', (x, y), xytext=(5, 5), 
                textcoords='offset points', fontsize=10)

plt.grid(True, alpha=0.3)

# Add padding around the plot
plt.xlim(-0.2, 5.2)
plt.ylim(-0.2, 4.2)

# Force ticks every 0.5 on both axes
plt.xticks(np.arange(0, 5.5, 0.5))
plt.yticks(np.arange(0, 4.5, 0.5))

plt.tight_layout()
plt.show()

def latex_distances_table(points, centers):
    print(r"\begin{table}[h!]")
    print(r"\centering")
    print(r"\begin{tabular}{c|ccc}")
    print(r"\textbf{Point} & $d_1$ & $d_2$ & $d_3$ \\")
    print(r"\hline")

    for i, point in enumerate(points):
        distances = [np.linalg.norm(point - center) for center in centers]
        print(f"$\\mathbf{{x}}_{{{i+1}}}$ & {distances[0]:.2f} & {distances[1]:.2f} & {distances[2]:.2f} \\\\")

    print(r"\end{tabular}")
    print(r"\caption{Distances from centroids to data points.}")
    print(r"\label{tab:distances}")
    print(r"\end{table}")
    print()


def latex_probabilities_table(points, centers):
    print(r"\begin{table}[h!]")
    print(r"\centering")
    print(r"\begin{tabular}{c|ccc}")
    print(r"\textbf{Point} & $p_1$ & $p_2$ & $p_3$ \\")
    print(r"\hline")

    for i, point in enumerate(points):
        distances = [np.linalg.norm(point - center) for center in centers]
        inv_distances = [1/d for d in distances]
        sum_inv = sum(inv_distances)
        probabilities = [inv_d / sum_inv for inv_d in inv_distances]
        print(f"$\\mathbf{{x}}_{{{i+1}}}$ & {probabilities[0]:.2f} & {probabilities[1]:.2f} & {probabilities[2]:.2f} \\\\")

    print(r"\end{tabular}")
    print(r"\caption{Membership probabilities $(p_1, p_2, p_3)$ computed according to the inverse-distance rule.}")
    print(r"\label{tab:probabilities}")
    print(r"\end{table}")
    print()


def latex_jdf_table(points, centers):
    print(r"\begin{table}[h!]")
    print(r"\centering")
    print(r"\begin{tabular}{c|cccc|c}")
    print(r"\textbf{Point} & $d_1 d_2 + d_1 d_3 + d_2 d_3$ & $D(\mathbf{x})$ \\")
    print(r"\hline")
    
    total_jdf = 0
    
    for i, point in enumerate(points):
        distances = [np.linalg.norm(point - center) for center in centers]
        d1, d2, d3 = distances
        
        # Calculate numerator (product of all distances)
        numerator = d1 * d2 * d3
        
        # Calculate denominator (sum of leave-one-out products)
        denominator = d1*d2 + d1*d3 + d2*d3
        
        # Calculate D(x) for this point
        d_x = numerator / denominator
        total_jdf += d_x
        
        print(f"$\\mathbf{{x}}_{{{i+1}}}$ & {denominator:.2f} & {d_x:.3f} \\\\")
    
    print(r"\hline")
    print(f"\\multicolumn{{2}}{{r|}}{{\\textbf{{Total JDF:}}}} & \\textbf{{{total_jdf:.3f}}} \\\\")
    print(r"\end{tabular}")
    print(r"\caption{Joint Distance Function (JDF) computation for each data point. $D(\mathbf{x}) = \frac{d_1 d_2 d_3}{d_1 d_2 + d_1 d_3 + d_2 d_3}$.}")
    print(r"\label{tab:jdf}")
    print(r"\end{table}")
    print()


def main():
    # Example 1
    latex_distances_table(points, centers)
    latex_probabilities_table(points, centers)

    # Example 2
    latex_jdf_table(points, centers)
    
main()