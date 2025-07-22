import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
import time
import os

# --- Data loading and parsing ---
# data = """
# -1  1:-12   2:0.1
# -1  1:-12   2:0.2
# 1   1:-12   2:0.5
# 1   1:-12   2:0.6
# 1   1:-12   2:0.7
# 1   1:-12   2:0.8
# -1  1:-10   2:0.1
# -1  1:-10   2:0.2
# 1   1:-10   2:0.5
# 1   1:-10   2:0.6
# 1   1:-10   2:0.7
# 1   1:-10   2:0.8
# 1   1:-10   2:0.9
# -1  1:-10   2:1
# -1  1:-10   21.3
# -1  1:-10   2:1.4
# -1  1:-8    2:.1
# -1  1:-8    2:.2
# 1   1:-8    2:1.1
# 1   1:-8    2:1.2
# 1   1:-8    2:1.3
# -1  1:-7    2:.1
# -1  1:-7    2:.2
# 1   1:-7    2:.6
# 1   1:-7    2:.7
# 1   1:-7    2:.8
# 1   1:-7    2:1.2
# -1  1:-6    2:.1
# -1  1:-6    2:.2
# 1   1:-6    2:.5
# 1   1:-6    2:1.1
# 1   1:-6    2:1.2
# -1  1:-5    2:.1
# -1  1:-5    2:.2
# 1   1:-5    2:.8
# 1   1:-5    2:1
# 1   1:-5    2:1.2
# -1  1:-4    2:.1
# -1  1:-4    2:.2
# 1   1:-4    2:.8
# 1   1:-4    2:1
# -1  1:0     2:.1
# -1  1:0     2:.2
# -1  1:0     2:.3
# -1  1:0     2:.4
# -1  1:0     2:.5
# -1  1:0     2:.6
# -1  1:0     2:.7
# -1  1:0     2:.8
# -1  1:0     2:.9
# -1  1:0     2:1
# -1  1:0     2:1.1
# -1  1:0     2:1.2
# -1  1:0     2:1.3
# -1  1:0     2:1.4
# -1  1:-15   2:0
# -1  1:-15   2:.1
# -1  1:-15   2:.2
# -1  1:-15   2:.3
# -1  1:-15   2:.4
# -1  1:-15   2:.5
# -1  1:-15   2:.6
# -1  1:-15   2:.7
# -1  1:-15   2:.8
# -1  1:-15   2:.9
# -1  1:-15   2:1
# -1  1:-15   2:1.1
# -1  1:-15   2:1.2
# -1  1:-15   2:1.3
# -1  1:-15   2:1.4
# -1  1:-15   2:1.5
# -1  1:5 2:.1
# -1  1:5 2:.2
# -1  1:5 2:.3
# -1  1:5 2:.4
# -1  1:5 2:.5
# -1  1:5 2:.6
# -1  1:5 2:.7
# -1  1:5 2:.8
# -1  1:5 2:.9
# -1  1:5 2:1
# -1  1:5 2:1.1
# -1  1:5 2:1.2
# -1  1:5 2:1.3
# -1  1:5 2:1.4
# -1  1:-12   2:1.5
# -1  1:-10   2:0
# -1  1:-10   2:1.5
# -1  1:-8    2:0
# -1  1:-8    2:1.5
# -1  1:-7    2:0
# -1  1:-7    2:1.5
# -1  1:-6    2:0
# -1  1:-6    2:1.5
# -1  1:-4    2:0
# -1  1:-4    2:1.5
# -1  1:0 2:0
# -1  1:0 2:1.5
# -1  1:5 2:0
# -1  1:5 2:1.5
# -1  1:-5    2:0
# -1  1:-5    2:1.5
# -1  1:-12   2:0
# 1   1:-7.54 2:1.16
# -1  1:-11.14    2:1.17
# 1   1:-10.61    2:0.86
# 1   1:-10.85    2:0.68
# 1   1:-5.77 2:0.57
# 1   1:-8.26 2:0.95
# -1  1:-2.18 2:0.31
# 1   1:-7.56 2:0.86
# """

data = """
-1	1:-15	2:0
-1	1:-15	2:0.1
-1	1:-15	2:0.2
-1	1:-15	2:0.3
-1	1:-15	2:0.4
-1	1:-15	2:0.5
-1	1:-15	2:0.6
-1	1:-15	2:0.7
-1	1:-15	2:0.8
-1	1:-15	2:0.9
-1	1:-15	2:1
-1	1:-15	2:1.1
-1	1:-15	2:1.2
-1	1:-15	2:1.3
-1	1:-15	2:1.4
-1	1:-15	2:1.5
-1	1:5	2:0.1
-1	1:5	2:0.2
-1	1:5	2:0.3
-1	1:5	2:0.4
-1	1:5	2:0.5
-1	1:5	2:0.6
-1	1:5	2:0.7
-1	1:5	2:0.8
-1	1:5	2:0.9
-1	1:5	2:1
-1	1:5	2:1.1
-1	1:5	2:1.2
-1	1:5	2:1.3
-1	1:5	2:1.4
-1	1:-12	2:1.5
-1	1:-10	2:0
-1	1:-10	2:1.5
-1	1:-8	2:0
-1	1:-8	2:1.5
-1	1:-7	2:0
-1	1:-7	2:1.5
-1	1:-6	2:0
-1	1:-6	2:1.5
-1	1:-4	2:0
-1	1:-4	2:1.5
-1	1:0	2:0
-1	1:0	2:1.5
-1	1:5	2:0
-1	1:5	2:1.5
-1	1:-5	2:0
-1	1:-5	2:1.5
-1	1:-12	2:0
-1	1:-12	2:0.1
-1	1:-12	2:0.2
1	1:-12	2:0.5
-1	1:-12	2:0.6
-1	1:-12	2:0.7
-1	1:-12	2:0.8
-1	1:-12	2:0.9
-1	1:-12	2:1
-1	1:-12	2:1.1
-1	1:-12	2:1.2
-1	1:-12	2:1.3
-1	1:-12	2:1.4
-1	1:-10	2:0.1
-1	1:-10	2:0.2
1	1:-10	2:0.5
-1	1:-10	2:0.6
-1	1:-10	2:0.7
-1	1:-10	2:0.8
1	1:-10	2:0.9
-1	1:-10	2:1
-1	1:-10	2:1.1
-1	1:-10	2:1.2
-1	1:-10	2:1.3
-1	1:-10	2:1.4
-1	1:-8	2:.1
-1	1:-8	2:.2
-1	1:-8	2:1
1	1:-8	2:1.1
1	1:-8	2:1.2
-1	1:-7	2:0.1
-1	1:-7	2:0.2
1	1:-7	2:0.6
1	1:-7	2:0.7
-1	1:-7	2:0.8
-1	1:-6	2:0.1
-1	1:-6	2:0.2
1	1:-6	2:0.4
1	1:-6	2:0.5
-1	1:-5	2:0.1
-1	1:-5	2:0.2
-1	1:-5	2:0.6
-1	1:-5	2:0.8
-1	1:-5	2:1
-1	1:-4	2:0.1
-1	1:-4	2:0.2
-1	1:0	2:0.1
-1	1:0	2:0.2
-1	1:0	2:0.3
-1	1:0	2:0.4
-1	1:0	2:0.5
-1	1:0	2:0.6
-1	1:0	2:0.7
-1	1:0	2:0.8
-1	1:0	2:0.9
-1	1:0	2:1
-1	1:0	2:1.1
-1	1:0	2:1.2
-1	1:0	2:1.3
-1	1:0	2:1.4
1	1:-7.05	2:0.49
1	1:-6.71	2:0.63
1	1:-8.31	2:0.48
1	1:-8.12	2:1.10
-1	1:-5.13	2:1.35
-1	1:-10.68	2:1.13
-1	1:-5.21	2:0.87
"""

labels = []
features = []
for line in data.strip().split('\n'):
    parts = line.split()
    labels.append(int(parts[0]))
    features.append([float(p.split(':')[1]) for p in parts[1:]])

X = np.array(features)  # Feature matrix
y = np.array(labels)    # Label vector

# --- Matplotlib Global Settings ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows/Linux/MacOS Try bold typeface
plt.rcParams['axes.unicode_minus'] = False    # Fix the abnormal display of the negative sign

# --- Parametric grid search ---
# Define a wide range of parameters for the search
# Fix: Change 2 to 2.0 to avoid the error of negative powers of integers
c_params = [2.0 ** i for i in np.arange(-5, 15, 0.5)]
g_params = [2.0 ** i for i in np.arange(-15, 3, 0.5)]
param_grid = {'C': c_params, 'gamma': g_params}

if __name__ == '__main__':
    print("Grid search is currently being conducted using the retention method for cross-validation. Please wait a moment...")
    start_time = time.time()
    
    # Use GridSearchCV to find the optimal parameters
    grid_search = GridSearchCV(
        SVC(kernel='rbf'),
        param_grid,
        cv=LeaveOneOut(),  # Cross-validation is carried out using the one-leave method
        scoring='accuracy'
    )
    grid_search.fit(X, y)
    
    end_time = time.time()
    print(f"The grid search was completed, but it took a long time: {end_time - start_time:.2f} s")

####Save parameters and accuracy####
    # Save the search process data
    results = grid_search.cv_results_
    output_file = 'grid_search_results.txt'

    # Extract all parameter combinations and their corresponding scores
    param_records = []
    for i in range(len(results['params'])):
        params = results['params'][i]
        mean_score = results['mean_test_score'][i]
        param_records.append((params['C'], params['gamma'], mean_score))

    # 写入文件
    with open(output_file, 'w') as f:
        f.write(f"{'C':<15} {'gamma':<15} {'accuracy of verification':<20}\n")
        for C, gamma, score in param_records:
            f.write(f"{C:<15.6f} {gamma:<15.6f} {score:<20.6f}\n")

    print(f"\n>> The parameter search process data has been saved to: {output_file}")
####Save parameters and accuracy####
    best_c = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']
    
    print(f"\nGridSearchCV The optimal parameters found: C={best_c:.4f}, gamma={best_gamma:.4f}")
    print(f"The corresponding highest cross-validation accuracy rate: {grid_search.best_score_:.4f}")
    
    # Obtain the optimal model
    best_svm = grid_search.best_estimator_

    # --- Visualization of results ---
    # Create a grid for drawing decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Create a single-graph layout
    fig, ax = plt.subplots(figsize=(12, 9))  # Adjust the size to a square proportion

    # Draw the decision boundary of the optimal model
    Z_best_flat = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_best = Z_best_flat.reshape(xx.shape)
    contour = ax.contourf(xx, yy, Z_best, cmap='coolwarm', alpha=0.4)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm',
                         edgecolors='k', s=50, label='data point')
    # Add decision boundaries and titles
    ax.set_title(f'SVM decision boundary (C={best_c:.2f}, γ={best_gamma:.2f})', fontsize=16, pad=15)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    # Add legends and color bars
    ax.legend(*scatter.legend_elements(), title="category", loc='upper right')
    fig.colorbar(contour, ax=ax, label='decision value')

    data_to_save = np.column_stack((xx.ravel(), yy.ravel(), Z_best_flat))
    np.savetxt('demo_predictions_best_model.txt',
               data_to_save,
               fmt='%.6f %.6f %d',
               header='x_coordinate y_coordinate prediction_probability',  # 列标题
               comments='')  # 避免默认添加注释符
    print("\nThe prediction file has been generated using the optimal model: demo_predictions_best_model.txt")
    plt.tight_layout()
    plt.savefig('demo_svm_decision_boundary_comparison.png', dpi=300)
    plt.show()


