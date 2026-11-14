from data_processing.load_data import load_data
from data_processing.feature_engineering import add_features
from data_processing.preprocess import x_y, encoder
from model.train import kfold, model, cv_engine
from sklearn.base import clone


if __name__ == "__main__":
    # 1. Load the data
    train, test, orig = load_data()

    # 2. Add the engineered features
    train = add_features(train)

    # 3. Split into features (x) and target (y)
    x, y = x_y(train)

    # 4. Turn text and True/False columns into numbers
    x = encoder(x)

    # 5. Create the 5 folds
    kf = kfold()

    # 6. Run cross-validation and show the results
    results = cv_engine(x, y, model, kf)
    print(results)