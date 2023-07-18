import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

tf.random.set_seed(42)


def load_data(identifier):
    ml_data_path = f"../ml_data/{identifier}/"
    train = np.load(ml_data_path + "train_data.npy", allow_pickle=True), \
        np.load(ml_data_path + "train_targets.npy", allow_pickle=True)
    val = np.load(ml_data_path + "val_data.npy", allow_pickle=True), \
        np.load(ml_data_path + "val_targets.npy", allow_pickle=True)
    test = np.load(ml_data_path + "test_data.npy", allow_pickle=True), \
        np.load(ml_data_path + "test_targets.npy", allow_pickle=True)
    return train, val, test


def base_model(patch_size):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(patch_size, patch_size, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def regularized_model(patch_size):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(patch_size, patch_size, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.SpatialDropout2D(0.1),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.SpatialDropout2D(0.1),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def build_model(patch_size):
    # model = base_model(patch_size)
    model = regularized_model(patch_size)
    model.compile(optimizer="adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train, val, epochs=100, patience=10):
    train_data, train_targets = train
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    return model.fit(train_data, train_targets, batch_size=256, epochs=epochs, validation_data=val, shuffle=True,
                     verbose=0, callbacks=callback)


def evaluate_training_model(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print("Training results:")
    print(f"- Training accuracy: {round(acc[-1], 5)}, Validation accuracy: {round(val_acc[-1], 5)}")
    print(f"- Training loss: {round(loss[-1], 5)}, Validation loss: {round(val_loss[-1], 5)}\n")

    epochs = range(len(acc))

    plt.plot(epochs, acc, label="train acc")
    plt.plot(epochs, val_acc, label="val acc")
    plt.title('Accuracy in training and validation')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.title('Loss in training and validation')
    plt.legend()


def plot_confusion_matrix(confusion_matrix):
    confusion_m_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    confusion_m_display.plot(cmap=plt.cm.Blues)


def calculate_evaluation_metrics(confusion_matrix):
    accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)
    mcc_nom = confusion_matrix[0, 0] * confusion_matrix[1, 1] - confusion_matrix[1, 0] * confusion_matrix[0, 1]
    mcc_denom = np.sqrt((confusion_matrix[0, 0] + confusion_matrix[0, 1])
                   * (confusion_matrix[0, 0] + confusion_matrix[1, 0])
                   * (confusion_matrix[1, 1] + confusion_matrix[0, 1])
                   * (confusion_matrix[1, 1] + confusion_matrix[1, 0]))
    if mcc_denom == 0:
        mcc = np.nan
    else:
        mcc = mcc_nom / mcc_denom
    return accuracy, mcc


def print_evaluation(text, predictions, test_y, plot=False):
    confusion_mat = confusion_matrix(test_y, predictions)
    if plot:
        plot_confusion_matrix(confusion_mat)

    accuracy, mcc = calculate_evaluation_metrics(confusion_mat)
    print(f"{text}Accuracy: {round(accuracy, 5)}, MCC: {round(mcc, 5)}")


def evaluate_test_data(model, test):
    test_x, test_y = test
    model_predictions_probability = model.predict(test_x, batch_size=10, verbose=0)
    model_predictions = np.rint(model_predictions_probability)[:, 0]
    print("Prediction results on the test set:")
    print_evaluation("- ", model_predictions, test_y, True)
    print()


def evaluate_baselines(test):
    _, test_y = test

    print("Baselines for evaluating performance:")
    random_prediction = np.random.randint(0, 2, len(test_y))
    print_evaluation("- Random class baseline - ", random_prediction, test_y)

    star_count = test_y.sum()
    gal_count = len(test_y) - star_count
    highest_class_prediction = np.zeros_like(test_y)
    if star_count > gal_count:
        highest_class_prediction = 1
    print_evaluation("- Mode class baseline - ", highest_class_prediction, test_y)
    print()


patch_size = 25
# identifier = f"patch_size{patch_size}_frames_10_ref_test"
# use this instead when you want to use augmented training data and have executed 04_augmentation.py
identifier = f"patch_size{patch_size}_frames_10_ref_test_augmented"

# identifier = f"patch_size{patch_size}_frames_3_ref_test"
# identifier = f"patch_size{patch_size}_frames_3_ref_test_augmented"

print(f"Building, training end evaluating ML model (binary classification):")
try:
    train, val, test = load_data(identifier)
    model = build_model(patch_size)
    # print(model.summary())

    print("- Training...", end=" ")
    history = train_model(model, train, val, epochs=1000, patience=10)
    print("done!")

    evaluate_training_model(history)
    evaluate_baselines(test)
    evaluate_test_data(model, test)
    plt.show()
    print(f"Model training and evaluation successful.")
except:
    print("Error in model training. Please try again.")
