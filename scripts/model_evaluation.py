def evaluate_model(model, X_test, y_test, batch_size=2048):
    """Evaluate the model performance on the test set."""
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print(f'Test score: {score}')
    print(f'Test accuracy: {acc}')
