def save_metrics_to_file(filename, accuracy, precision, recall, mse, mae):
    with open(filename, 'w') as file:
        file.write(f'Accuracy: {accuracy:.2f}\n')
        file.write(f'Precision: {precision:.2f}\n')
        file.write(f'Recall: {recall:.2f}\n')
        file.write(f'Mean Square Error (MSE): {mse_value:.2f}\n')
        file.write(f'Mean Absolute Error (MAE): {mae_value:.2f}\n')

def accuracy_score(labels, predicted_labels):
    if not len(predicted_labels) or not len(labels):
        raise ValueError('Lists must not be empty!')
    if len(predicted_labels) != len(labels):
        raise ValueError('Lists must have the same length!')

    accuracy = sum(1 for i in range(len(predicted_labels)) if predicted_labels[i] == labels[i]) / len(predicted_labels)
    return accuracy


def precision_recall_score(labels, predicted_labels):
    if not len(predicted_labels) or not len(labels):
        raise ValueError('Lists must not be empty!')
    if len(predicted_labels) != len(labels):
        raise ValueError('Lists must have the same length!')

    tp = sum(1 for i in range(len(predicted_labels)) if predicted_labels[i] == 1 and labels[i] == 1)
    fp = sum(1 for i in range(len(predicted_labels)) if predicted_labels[i] == 1 and labels[i] == 0)
    fn = sum(1 for i in range(len(predicted_labels)) if predicted_labels[i] == 0 and labels[i] == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


def mse(labels, predicted_labels):
    if not len(predicted_labels) or not len(labels):
        raise ValueError('Lists must not be empty!')
    if len(predicted_labels) != len(labels):
        raise ValueError('Lists must have the same length!')

    mse = sum((predicted_labels[i] - labels[i]) ** 2 for i in range(len(predicted_labels))) / len(predicted_labels)
    return mse


def mae(labels, predicted_labels):
    if not len(predicted_labels) or not len(labels):
        raise ValueError('Lists must not be empty!')
    if len(predicted_labels) != len(labels):
        raise ValueError('Lists must have the same length!')

    mae = sum(abs(predicted_labels[i] - labels[i]) for i in range(len(predicted_labels))) / len(predicted_labels)
    return mae


# Datele de test
y_pred = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Calcul metrici
accuracy = accuracy_score(y_true, y_pred)
print('Accuracy binary classifier: %.2f' % accuracy)
precision, recall = precision_recall_score(y_true, y_pred)
print('The precision is %.2f and the recall is %.2f - for binary classifier.' % (precision, recall))
mse = mse(y_true, y_pred)
print('Mean square error %.2f' % mse)
mae = mae(y_true, y_pred)
print('Mean absolute error %.2f' % mae)

# Salvare în fișier
save_metrics_to_file("metrici.txt", accuracy, precision, recall, mse, mae)

print("Metricile au fost salvate în 'metrici.txt'.")
