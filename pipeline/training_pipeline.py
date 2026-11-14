from train import model
from load_data import load_data
from model.evaluate import evaluate_model

if __name__ == "__main__":
    df = load_data()
    x,y = df.drop("target",axis=1),df["target"]
    model, X_test, y_test = train_model(X, y)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy}")
