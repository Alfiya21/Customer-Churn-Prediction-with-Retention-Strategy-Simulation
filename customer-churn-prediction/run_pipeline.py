from src.data_loader import load_data
from src.preprocessing import clean_data, encode_target
from src.feature_engineering import create_features
from src.train_model import prepare_train_test, train_random_forest
from src.evaluate_model import evaluate_model
from src.explainability import generate_shap_summary


DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main():
    # Load data
    df = load_data(DATA_PATH)

    # Preprocess
    df = clean_data(df)
    df = encode_target(df)

    # Feature engineering
    df = create_features(df)

    # Train model
    X_train, X_test, y_train, y_test, encoder = prepare_train_test(df)
    model = train_random_forest(X_train, y_train)

    # Evaluate
    y_prob = evaluate_model(model, X_test, y_test)

    # Explainability
    generate_shap_summary(model, X_train)



if __name__ == "__main__":
    main()
