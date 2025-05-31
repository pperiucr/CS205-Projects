import math
import sys
import os
import pandas as pd
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_csv_data(filename, label_column="Engine Condition", max_rows=2048):
    df = pd.read_csv(filename)

    # Drop non-numeric columns
    df = df.select_dtypes(include=["number"])

    # Shuffle and limit rows
    df = df.sample(frac=1, random_state=42).head(max_rows)

    labels = df[label_column].astype(int).tolist()
    features = df.drop(columns=[label_column])

    # Normalize features
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(features)

    data = list(zip(labels, normalized.tolist()))
    return data
    
def load_txt_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            if line.strip():
                parts = list(map(float, line.strip().split()))
                label, features = int(parts[0]), parts[1:]
                data.append((label, features))
    return data

def leave_one_out(data, features):
    correct = 0
    for i in range(len(data)):
        test = data[i]
        best_dist = float('inf')
        best_label = None
        for j in range(len(data)):
            if i == j:
                continue
            dist = sum((test[1][f] - data[j][1][f])**2 for f in features)
            if dist < best_dist:
                best_dist = dist
                best_label = data[j][0]
        if best_label == test[0]:
            correct += 1
    return correct / len(data) * 100

def forward_selection(data):
    print("Beginning search:")
    num_features = len(data[0][1])
    current = []
    best_overall = 0
    accuracies = []
    feature_order = []

    for i in range(num_features):
        best_feature = None
        best_accuracy = 0
        for f in range(num_features):
            if f in current:
                continue
            trial = current + [f]
            acc = leave_one_out(data, trial)
            if(num_features < 15):
                print(f"Using feature(s) {[x+1 for x in trial]} accuracy is {acc:.1f}%")
            if acc > best_accuracy:
                best_accuracy = acc
                best_feature = f

        if best_feature is not None:
            current.append(best_feature)
            best_overall = best_accuracy
            accuracies.append(best_accuracy)
            feature_order.append(best_feature + 1)  # using 1-based indexing
            print(f"Feature set {[x+1 for x in current]} was best with accuracy {best_accuracy:.1f}%")
        else:
            break

    print("Finished search.")
    return feature_order, accuracies


def backward_elimination(data):
    print("Beginning search:")
    num_features = len(data[0][1])
    current = list(range(num_features))
    
    feature_order = []
    accuracies = []
    
    best_overall = leave_one_out(data, current)
    accuracies.append(best_overall)
    feature_order.append(None)  # No feature removed in full set

    print(f"Using feature(s) {[x+1 for x in current]} accuracy is {best_overall:.1f}%")

    for i in range(num_features - 1):
        best_set = None
        best_accuracy = 0
        feature_removed = None
        
        for f in current:
            trial = deepcopy(current)
            trial.remove(f)
            acc = leave_one_out(data, trial)
            if(num_features < 15):
                print(f"Using feature(s) {[x+1 for x in trial]} accuracy is {acc:.1f}%")
            if acc > best_accuracy:
                best_accuracy = acc
                best_set = trial
                feature_removed = f
        
        if best_set is not None and best_accuracy > best_overall:
            current = best_set
            feature_order.append(feature_removed)
            accuracies.append(best_accuracy)
            best_overall = best_accuracy
            print(f"Feature set {[x+1 for x in current]} was best with accuracy {best_accuracy:.1f}%")
        else:
            break
    
    print("Finished search.")
    return feature_order[1:], accuracies[1:]  # Skip the "None" full-set entry


def plot_feature_selection_progress(feature_order, accuracies, feature_names=None):
    print(f"[DEBUG] feature_order: {feature_order}")
    print(f"[DEBUG] accuracies: {accuracies}")

    if len(feature_order) != len(accuracies):
        raise ValueError(
            f"Mismatch: feature_order has {len(feature_order)} items, "
            f"but accuracies has {len(accuracies)} items."
        )

    x_ticks = list(range(1, len(accuracies) + 1))

    # Generate readable x-axis labels
    if feature_names:
        try:
            x_labels = [feature_names[i] for i in feature_order]
        except IndexError as e:
            raise IndexError("Feature index in feature_order exceeds feature_names length.") from e
    else:
        x_labels = [f + 1 for f in feature_order]  # 1-based index for display

    print(f"[DEBUG] x_labels: {x_labels}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_ticks, accuracies, marker='o')
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45)
    plt.xlabel("Feature Added/Removed")
    plt.ylabel("Accuracy (%)")
    plt.title("Feature Selection: Accuracy Progression")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 nn_feature_select.py <datafile> <forward|backward>")
        sys.exit(1)
        
    ext = os.path.splitext(sys.argv[1])[1].lower()
    if ext == ".txt":
        data = load_txt_data(sys.argv[1])
    elif ext == ".csv":
        data = load_csv_data(sys.argv[1])
    else:
        raise ValueError("Unsupported file format. Use .txt or .csv")
        
    if sys.argv[2] == "forward":
        feature_order, accuracies = forward_selection(data)
    elif sys.argv[2] == "backward":
        feature_order, accuracies = backward_elimination(data)
    else:
        print("Method must be 'forward' or 'backward'")
        
    plot_feature_selection_progress(feature_order, accuracies)
