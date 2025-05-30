import sys
import csv
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split



if len(sys.argv) != 2:
    print("Usage: python neural_network_classification.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
X = []
y = []

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 8:
            continue  # skip incomplete rows
        X.append([float(row[3]), float(row[4]), float(row[5])])
        y.append(int(row[7]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

activation = 'relu'
batch_size = 200
learning_rate_init = 0.01
max_iter = 100000
random_state = 2
tol = 0.0001
momentum = 0.1
n_iter_no_change = 10000

clf = MLPClassifier(
                    hidden_layer_sizes=(20, 20),
                    activation=activation,
                    solver='sgd',
                    alpha=0,                            #prolly 0 for simplicity of the model
                    batch_size=batch_size,             #default is 'auto'
                    # learning_rate='constant',         #prolly constant for simplicity of the model
                    learning_rate_init=learning_rate_init,    #default is 0.001
                    # power_t=0.5,                      #since learning rate is constant, power_t is not needed
                    max_iter=max_iter,                #default is 200
                    shuffle=True,               #default is True
                    random_state=random_state,             #default is None
                    tol=tol,                 #default is 0.0001
                    verbose=True,               #default is false
                    # warm_start=False,                 #prolly False for simplicity of the model
                    momentum=momentum,                         #prolly 0 for simplicity of the model
                    # nesterovs_momentum=True,          #since momentum will be 0, this is not needed
                    # early_stopping=False,             #prolly False for simplicity of the model
                    # validation_fraction=0.1,          #since early stopping is False, this is not needed
                    # beta_1=0.9,                       #since solver is not adam, this is not needed
                    # beta_2=0.999,                     #since solver is not adam, this is not needed
                    # epsilon=1e-08,                    #since solver is not adam, this is not needed
                    n_iter_no_change=n_iter_no_change,      #default is 10
                    # max_fun=15000                     #since solver is not lbfgs, this is not needed
                    )


clf.fit(X, y)

# Predict using the trained model
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

print("Training report:")
print(classification_report(y_train, y_pred_train, zero_division=0.0))
print("Test report:")
print(classification_report(y_test, y_pred_test,zero_division=0.0))



# # Ensure output directory exists
# os.makedirs('./output', exist_ok=True)

# output_file = './output/predictions_comparison.csv'


# # Write comparison to CSV
# with open(output_file, 'w', newline='', encoding='utf-8') as out_csv:
#     writer = csv.writer(out_csv)
#     writer.writerow(['Input1', 'Input2', 'Input3', 'Actual', 'Predicted'])
#     for inp, actual, pred in zip(X, y, predictions):
#         writer.writerow([inp[0], inp[1], inp[2], actual, pred])

# print(f"Comparison written to {output_file}")