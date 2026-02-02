# ===============================
# Main Execution File
# ===============================

from data_generator import get_data_generators
from model import build_model
from train import train_model
from evaluation import evaluate_model

def main():

    train_gen, val_gen, test_gen = get_data_generators()

    model = build_model()
    model.summary()

    history, history_fine = train_model(
        model,
        train_gen,
        val_gen
    )

    evaluate_model(model, test_gen)

    model.save("models/emotion_densenet_cnn.h5")


if __name__ == "__main__":
    main()
