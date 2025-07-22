import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


# Dataset
data = """
-1	-12	0.1
-1	-12	0.2
1	-12	0.5
1	-12	0.6
1	-12	0.7
1	-12	0.8
-1	-10	0.1
-1	-10	0.2
1	-10	0.5
1	-10	0.6
1	-10	0.7
1	-10	0.8
1	-10	0.9
-1	-10	1
-1	-10	1.3
-1	-10	1.4
-1	-8	0.1
-1	-8	0.2
1	-8	1.1
1	-8	1.2
1	-8	1.3
-1	-7	0.1
-1	-7	0.2
1	-7	0.6
1	-7	0.7
1	-7	0.8
1	-7	1.2
-1	-6	0.1
-1	-6	0.2
1	-6	0.5
1	-6	1.1
1	-6	1.2
-1	-5	0.1
-1	-5	0.2
1	-5	0.8
1	-5	1
1	-5	1.2
-1	-4	0.1
-1	-4	0.2
1	-4	0.8
1	-4	1
-1	0	0.1
-1	0	0.2
-1	0	0.3
-1	0	0.4
-1	0	0.5
-1	0	0.6
-1	0	0.7
-1	0	0.8
-1	0	0.9
-1	0	1
-1	0	1.1
-1	0	1.2
-1	0	1.3
-1	0	1.4
1	-7.54	1.16
-1	-11.14	1.17
1	-10.61	0.86
1	-10.85	0.68
1	-5.77	0.57
1	-8.26	0.95
-1	-2.18	0.31
1	-7.56	0.86
-1	-15	0
-1	-15	0.1
-1	-15	0.2
-1	-15	0.3
-1	-15	0.4
-1	-15	0.5
-1	-15	0.6
-1	-15	0.7
-1	-15	0.8
-1	-15	0.9
-1	-15	1
-1	-15	1.1
-1	-15	1.2
-1	-15	1.3
-1	-15	1.4
-1	-15	1.5
-1	5	0.1
-1	5	0.2
-1	5	0.3
-1	5	0.4
-1	5	0.5
-1	5	0.6
-1	5	0.7
-1	5	0.8
-1	5	0.9
-1	5	1
-1	5	1.1
-1	5	1.2
-1	5	1.3
-1	5	1.4
-1	-12	1.5
-1	-10	0
-1	-10	1.5
-1	-8	0
-1	-8	1.5
-1	-7	0
-1	-7	1.5
-1	-6	0
-1	-6	1.5
-1	-4	0
-1	-4	1.5
-1	0	0
-1	0	1.5
-1	5	0
-1	5	1.5
-1	-5	0
-1	-5	1.5
-1	-12	0
"""

# data = """
# -1	-12	0.1
# -1	-12	0.2
# 1	-12	0.5
# -1	-12	0.6
# -1	-12	0.7
# -1	-12	0.8
# -1	-12	0.9
# -1	-12	1
# -1	-12	1.1
# -1	-12	1.2
# -1	-12	1.3
# -1	-12	1.4
# -1	-10	0.1
# -1	-10	0.2
# 1	-10	0.5
# -1	-10	0.6
# -1	-10	0.7
# -1	-10	0.8
# 1	-10	0.9
# -1	-10	1
# -1	-10	1.1
# -1	-10	1.2
# -1	-10	1.3
# -1	-10	1.4
# -1	-8	0.1
# -1	-8	0.2
# -1	-8	1
# 1	-8	1.1
# 1	-8	1.2
# -1	-7	0.1
# -1	-7	0.2
# 1	-7	0.6
# 1	-7	0.7
# -1	-7	0.8
# -1	-6	0.1
# -1	-6	0.2
# 1	-6	0.4
# 1	-6	0.5
# -1	-5	0.1
# -1	-5	0.2
# -1	-5	0.6
# -1	-5	0.8
# -1	-5	1
# -1	-4	0.1
# -1	-4	0.2
# -1	0	0.1
# -1	0	0.2
# -1	0	0.3
# -1	0	0.4
# -1	0	0.5
# -1	0	0.6
# -1	0	0.7
# -1	0	0.8
# -1	0	0.9
# -1	0	1
# -1	0	1.1
# -1	0	1.2
# -1	0	1.3
# -1	0	1.4
# 1	-7.05	0.49
# 1	-6.71	0.63
# 1	-8.31	0.48
# 1	-8.12	1.1
# -1	-5.13	1.35
# -1	-10.68	1.13
# -1	-5.21	0.87
# -1	-15	0
# -1	-15	0.1
# -1	-15	0.2
# -1	-15	0.3
# -1	-15	0.4
# -1	-15	0.5
# -1	-15	0.6
# -1	-15	0.7
# -1	-15	0.8
# -1	-15	0.9
# -1	-15	1
# -1	-15	1.1
# -1	-15	1.2
# -1	-15	1.3
# -1	-15	1.4
# -1	-15	1.5
# -1	5	0.1
# -1	5	0.2
# -1	5	0.3
# -1	5	0.4
# -1	5	0.5
# -1	5	0.6
# -1	5	0.7
# -1	5	0.8
# -1	5	0.9
# -1	5	1
# -1	5	1.1
# -1	5	1.2
# -1	5	1.3
# -1	5	1.4
# -1	-12	1.5
# -1	-10	0
# -1	-10	1.5
# -1	-8	0
# -1	-8	1.5
# -1	-7	0
# -1	-7	1.5
# -1	-6	0
# -1	-6	1.5
# -1	-4	0
# -1	-4	1.5
# -1	0	0
# -1	0	1.5
# -1	5	0
# -1	5	1.5
# -1	-5	0
# -1	-5	1.5
# -1	-12	0
# """


# Convert the data into a list form
data_lines = data.strip().split("\n")
parsed_data = []
for line in data_lines:
    parts = line.split()
    label = int(parts[0])  # The first column is the label
    feature1 = float(parts[1])  # The second column is Feature 1
    feature2 = float(parts[2])  # The third column is Feature 2
    parsed_data.append([label, feature1, feature2])

# Convert to a NumPy array
data_array = np.array(parsed_data)

# Separate features from labels
X = data_array[:, 1:]
y = data_array[:, 0]

# Divide the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=90)

# Initialization accuracy
acc_score = 0

# Initialize the highest accuracy rate
acc_score_best = 0

# Initialize the optimal parameters
n_estimators_best = 0

# Store the accuracy rates under different parameters
list_accuracy_1 = []

num = 0
for i in range(5, 101):
    num+=1

    # Initialize the random forest model
    clf = RandomForestClassifier(n_estimators=i, max_depth=20, min_samples_leaf=1, min_samples_split=4, max_features=1, n_jobs=-1, random_state=90)

    # Cross-validation is carried out using the one-leave method
    loo = LeaveOneOut()
    loo_scores = cross_val_score(clf, X_train, y_train, cv=loo, scoring='accuracy', n_jobs=-1)

    # The accuracy rate of this cycle
    acc_score = loo_scores.mean()

    # Add the parameters and accuracy rate of each loop to the list
    list_accuracy_1.append([num, i, acc_score])

    # Output result
    print(f"The {num} th cycle：The accuracy rate is {acc_score}")

    # Find out the best accuracy rate
    if acc_score >= acc_score_best:
        acc_score_best = acc_score
        n_estimators_best = i

# The optimal parameter combination and its accuracy rate
best_para = [n_estimators_best, acc_score_best]
print(best_para)

list_accuracy = np.array(list_accuracy_1)
data = pd.DataFrame(list_accuracy)

# Save the DataFrame to an Excel file using ExcelWriter
with pd.ExcelWriter('demo_estimator.xlsx', engine='openpyxl') as writer:  # lift recede的结果文件
    data.to_excel(writer, sheet_name='Sheet1')


# Initialize the random forest model with the best parameters
clf = RandomForestClassifier(n_estimators=n_estimators_best, max_depth=20, min_samples_leaf=1, min_samples_split=4, max_features=1, n_jobs=-1, random_state=90)

# Train the model
clf.fit(X_train, y_train)

# Predictive test set
y_pred = clf.predict(X_test)

# Evaluation model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred))

# Define the drawing decision boundary function
def plot_decision_boundary(model, X, y):
    # Set up the grid to calculate the model output
    x_min, x_max = -17.5, 7.5
    y_min, y_max = -0.15, 1.65
    h = 0.005
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Calculate the predicted values of the grid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.xlim(-17.5,7.5)
    plt.ylim(-0.15,1.65)
    x_major_locator = MultipleLocator(5)
    y_major_locator = MultipleLocator(0.3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    
    # Draw the decision boundary
    Colors=('#f14040','#37ad6b')
    plt.contourf(xx, yy, Z, alpha=1, colors=Colors)
    plt.show()

# Draw the decision boundary map
plot_decision_boundary(clf, X, y)