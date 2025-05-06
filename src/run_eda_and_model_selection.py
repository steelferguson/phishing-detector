import utils as u
import exploratory_analysis as eda
from gradient_boosted_classifier import (
    select_gbm_parameters, create_gbm_with_set_hyperparameters, evaluate_gbm
)
from simple_nn import train_binary_classifier, evaluate_binary_classifier
import joblib
import config


if __name__ == "__main__":
    # Load
    df = u.load_data(location=config.DATA_INPUT_LOCATION)

    # Exploratory Data Analysi
    eda.run_explorations(df)
    df = u.add_ratio_features(df)
    
    # Split for training and keep that the same for models to compare
    X_train, X_test, y_train, y_test = u.get_train_and_test_from_df(df, random_state=config.RANDOM_STATE_TEST)

    # Select params for GBM
    # select_gbm_parameters(X_train, X_test, y_train, y_test) # grid search

    # GBM again with weighting
    # select_gbm_parameters(
    #     X_train, X_test, y_train, y_test, 
    #     output_csv=config.GBM_WEIGHTED_RESULTS_LOCATION, 
    #     use_sample_weights=True
    # ) 

    # Train unweighted
    # model1 = train_binary_classifier(
    #     X_train, y_train,
    #     reweight=False,
    #     epochs=20,
    #     plot_loss=True,
    #     loss_plot_path="data/outputs/nn_training_loss.png"
    # )
    # print("Unweighted simple nn")
    # print(evaluate_binary_classifier(model1, X_test, y_test))

    # Train with phishing class weighted 2x
    # model2 = train_binary_classifier(
    #     X_train, y_train,
    #     reweight=True,
    #     epochs=20,
    #     plot_loss=True,
    #     loss_plot_path="data/outputs/nn_reweighted_training_loss.png"
    # )
    # print("Re Weighted simple nn")
    # print(evaluate_binary_classifier(model2, X_test, y_test))
    # {'precision': 0.43775100401606426, 'recall': 0.15683453237410072, 'f1': 0.2309322033898305, 'roc_auc': np.float64(0.8409190199491305)}

    X_train, X_val, y_train, y_val = u.get_train_and_test_from_df(df, random_state=config.RANDOM_STATE_VALIDATION)
    # final_model = create_gbm_with_set_hyperparameters(
    #     X_train, 
    #     X_val, 
    #     y_train, 
    #     y_val, 
    #     config.GBM_SELECTED_HYPERPARAMS, 
    #     use_sample_weights=False
    # )

    final_model = joblib.load('src/model_store/final_model.pkl')
    metrics = evaluate_gbm(final_model, X_train, X_val, y_train, y_val, plot_path="data/outputs/pr_curve.png")
    print("final metrics")
    print(metrics)


    # joblib.dump(final_model, 'src/model_store/final_model.pkl')
    










