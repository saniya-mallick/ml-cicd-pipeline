from src.train import train_model


def test_accuracy_above_threshold():
    _, accuracy = train_model()
    assert accuracy >= 0.9


def test_predictions_are_valid_classes():
    model, _ = train_model()
    sample = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
    predictions = model.predict(sample)
    assert set(predictions).issubset({0, 1, 2})


def test_setosa_is_predicted_correctly():
    model, _ = train_model()
    assert model.predict([[5.1, 3.5, 1.4, 0.2]])[0] == 0