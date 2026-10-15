from spam_classifier.modeling import build_pipeline, evaluate_model


def test_pipeline_trains_and_predicts():
    texts = ["team meeting tomorrow", "claim free cash now", "project update", "win prize click"]
    labels = [0, 1, 0, 1]
    model = build_pipeline("naive_bayes")
    model.fit(texts, labels)
    metrics = evaluate_model(model, texts, labels)
    assert set(metrics) == {"accuracy", "spam_precision", "spam_recall", "spam_f1", "report"}
    assert model.predict(["free prize"])[0] == 1
