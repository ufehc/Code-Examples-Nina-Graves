from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from Global_configuartions import FOLDERNAME_MODELLING_DATA
import pandas as pd
import numpy as np
from tqdm import tqdm

Pisa_Type = ["MATH", "READ", "SCIENCE"]
model_run_name = "Random_Forest_no_restrictions_RS42_2"
path_0 = Path(FOLDERNAME_MODELLING_DATA)
path_results = Path(path_0 / "Results")
path_tree = Path(path_0 / "Trees")
path_input = Path(path_0 / "Input Data")
randomSearch = False
gridSearch = False
percentage_test_data = 15
model_run = model_run_name


pbar_pisa = tqdm(Pisa_Type)
for pisa_type in pbar_pisa:
    pbar_pisa.set_description(f"{pisa_type}")

    if pisa_type == "READ":
        input_data = "aggregated-years"
        iteration = 55
    else:
        input_data = "pisa-years"
        if pisa_type == "MATH":
            iteration = 65
        else:
            iteration = 50

    model_run = model_run_name

    # create directories
    path_results.mkdir(exist_ok=True)
    path_results_modelrun = Path(path_results / f"{model_run}")
    path_results_modelrun.mkdir(exist_ok=True)
    path_results_modelrun.mkdir(exist_ok=True)
    path_tree.mkdir(exist_ok=True)
    path_tree_modelrun = Path(path_tree / f"{model_run}")
    path_tree_modelrun.mkdir(exist_ok=True)

    # load input data and meta data
    data_for_model = pd.read_csv(Path(path_input / f"{input_data}") / f"{pisa_type}-{iteration}.csv")
    column_information = pd.read_csv(Path(path_input / f"{input_data}") / f"{pisa_type}-Column_information.csv")

    if "Unnamed: 0" in column_information.columns:
        column_information = column_information.rename(columns={"Unnamed: 0": "Index"})
    column_information = column_information.set_index("Index")
    data_for_model = data_for_model.set_index("Index")

    # prepare Data
    feature_columns = [x for x in data_for_model.columns if x != "PISA-SCORE" and x != "COUNTRY"]
    X = data_for_model[feature_columns]
    y = data_for_model["PISA-SCORE"]

    # Split Data
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, y, test_size=0.15,
                                                                                random_state=42)
    # create model
    model_forest = RandomForestRegressor(random_state=42)

    # train model
    model_forest.fit(X_train_final, y_train_final)

    # get child estimators
    child_estimators = model_forest.estimators_

    # evaluate model
    y_predict = model_forest.predict(X_test_final)

    # get feature permutation imortance on test-data
    perm_import_test = permutation_importance(model_forest, X_test_final, y_test_final, n_jobs=-1)
    perm_import_test_raw = pd.DataFrame(perm_import_test.importances)

    # get feature permutation imortance on training-data
    perm_import_train = permutation_importance(model_forest, X_train_final, y_train_final, n_jobs=-1)

    # save information about features
    features = X.columns.to_list()
    features_df = pd.DataFrame(index=features)
    features_df["Feature Importance"] = model_forest.feature_importances_
    features_df["TEST-Permutation Importance Mean"] = perm_import_test.importances_mean
    features_df["TEST-Permutation Importance SD"] = perm_import_test.importances_std
    features_df["TRAIN-Permutation Importance Mean"] = perm_import_train.importances_mean
    features_df["TRAIN-Permutation Importance SD"] = perm_import_train.importances_std
    features_df = features_df.sort_values("TEST-Permutation Importance Mean", ascending=False)
    features_merged = features_df.merge(column_information, how="left", left_index=True, right_index=True)
    features_merged.to_csv(path_results_modelrun / f"{pisa_type}_Feature_Analysis_{model_run}.csv")

    # store information on model run and estimator
    results = X_test_final.copy()
    parameters_model = model_forest.get_params()
    results = results.drop(columns=feature_columns)
    results["Modelrun"] = model_run
    results["Input-Data"] = input_data
    results["Amount of non-NA Values"] = iteration
    results = results.merge(data_for_model[["COUNTRY", "YEAR"]], on="Index")
    results["PISA-SCORE"] = y_test_final
    results["y_predict"] = y_predict
    results["Absoloute Error"] = abs(results["PISA-SCORE"] - results["y_predict"])
    results["Squared Error"] = abs(results["PISA-SCORE"] - results["y_predict"]) ** 2
    results["RMSE"] = np.sqrt(results["Squared Error"])
    results["R^2"] = np.nan
    new_row_df = pd.DataFrame(
        columns=["Modelrun", "COUNTRY", "YEAR", "PISA-SCORE", "y_predict", "Absoloute Error", "Squared Error", "RMSE",
                 "R^2"],
        data=[[model_run, np.nan, np.nan, np.nan, np.nan, sklearn.metrics.mean_absolute_error(y_test_final, y_predict),
               sklearn.metrics.mean_squared_error(y_test_final, y_predict),
               np.sqrt(sklearn.metrics.mean_squared_error(y_test_final, y_predict)),
               model_forest.score(X_test_final, y_test_final)]], index=[model_run])
    results_with_total_overview = pd.concat([new_row_df, results])
    results_with_total_overview.loc[model_run, "Input-Data"] = input_data
    results_with_total_overview.loc[model_run, "Amount of non-NA Values"] = iteration
    results_with_total_overview.loc[model_run, "Total available features"] = len(features)
    results_with_total_overview.loc[model_run, "RandomSearch"] = randomSearch
    results_with_total_overview.loc[model_run, "GridSearch"] = gridSearch
    results_with_total_overview.loc[model_run, "Percentage Test-Data"] = percentage_test_data
    results_with_total_overview.loc[model_run, "R2 Training Data"] = model_forest.score(X_train_final, y_train_final)
    results_with_total_overview.loc[model_run, "Depth"] = parameters_model["max_depth"]
    results_with_total_overview.loc[model_run, "min_samples_leaf"] = parameters_model["min_samples_leaf"]
    results_with_total_overview.loc[model_run, "min_samples_split"] = parameters_model["min_samples_split"]
    results_with_total_overview.loc[model_run, "ccp_alpha"] = parameters_model["ccp_alpha"]
    results_with_total_overview.loc[model_run, "Number of Estimators"] = parameters_model["n_estimators"]
    results_with_total_overview.loc[model_run, "Percentage of Features"] = parameters_model["max_features"]
    results_with_total_overview.loc[model_run, "Number of features used"] = model_forest.n_features_
    results_with_total_overview.to_csv(path_results_modelrun / f"{pisa_type}_Results_{model_run}.csv")

    # store information on child estimators
    child_estimator_info = pd.DataFrame(index=child_estimators)
    for estimator in child_estimators:
        params = estimator.get_params()
        child_estimator_info.loc[estimator, "Parameter"] = model_run
        child_estimator_info.loc[estimator, "Percentage non-NA"] = iteration
        child_estimator_info.loc[estimator, "R2"] = estimator.score(X_test_final, y_test_final)
        child_estimator_info.loc[estimator, "Random State"] = params["random_state"]
        child_estimator_info.loc[estimator, "Depth"] = estimator.get_depth()
        child_estimator_info.loc[estimator, "Number of Features"] = estimator.max_features_
        child_estimator_info.loc[estimator, "Percentage of Features"] = estimator.max_features
        child_estimator_info.loc[estimator, "Number Leaves"] = estimator.get_n_leaves()
        child_estimator_info.loc[estimator, "Feature Importance"] = str(
            estimator.feature_importances_)  # return to list with json lib -> json.loads()
        child_estimator_info.loc[estimator, "Predictions"] = str(
            estimator.predict(X_test_final))  # return to list with json lib -> json.loads()
    child_estimator_info.to_csv(path_results_modelrun / f"{pisa_type}_Tree-Childen-Info_{model_run}.csv")

    ######### Plots ##############
    path_results_modelrun_plots = Path(path_results_modelrun / f"{pisa_type}_Plots")
    path_results_modelrun_plots.mkdir(exist_ok=True)

    ###Prediction, True Scores ###
    plt.figure(iteration, figsize=(20, 10))
    plot_x_data = X_test_final.sort_values(["YEAR"])
    sc_plot_predict = sns.scatterplot(plot_x_data.index, y_test_final,
                                      hue=data_for_model.loc[X_test_final.index, "COUNTRY"])
    plt.xticks(rotation=45)
    plt.plot(y_predict, c="r", label="Prediction")
    plt.legend(loc=1)
    plt.savefig(path_results_modelrun_plots / f"RESULT_{pisa_type}_{model_run}.png")
    plt.close()

    ### Absolute Abweichung ###
    plt.figure(iteration + 4)
    plt.scatter(results["PISA-SCORE"], results["RMSE"])
    plt.xlabel("True PISA Score")
    plt.ylabel("RMSE")
    plt.title(f"{pisa_type}_{model_run}: RMSE per Score")
    plt.savefig(path_results_modelrun_plots / f"{pisa_type}_RMSE-True_Score_{model_run}.png")
    plt.close()

    ### Average Error per year ###
    plt.figure(iteration + 1)
    tmp = results[["YEAR", "RMSE"]].groupby("YEAR").mean()
    plt.plot(tmp.index, tmp, marker="o", label="Average Error")
    plt.hist(results["YEAR"], bins=100, align="left", label="Frequency in Test-Data")
    plt.xticks(tmp.index)
    plt.legend()
    plt.title(f"{pisa_type}_{model_run}: Average RMSE per year")
    plt.savefig(path_results_modelrun_plots / f"{pisa_type}_RMSE_per_year_{model_run}.png")
    plt.close()

    ### Robustness-Check ###
    plt.figure(iteration + 2, figsize=(30, 10))
    feature_names = features_merged.index.to_list()[:10]
    feature_names = [f"{x}: {feature_names[x]}" for x in range(len(feature_names))]
    feature_names = "\n".join(feature_names)
    data = features_merged[:10].copy()

    plt.plot(range(len(data.index)), data["Annual Mean"], marker="x", c="r", label="Annual Mean used for NaN")
    plt.hist(data.index, bins=range(len(data.index)), color="slategray", align="left", rwidth=0.5,
             weights=data["NA-Values"], label="# NA-Values")
    plt.legend()
    plt.xticks(range(len(data.index)), range(len(data.index)));
    plt.xlabel(feature_names, fontsize="x-small", ma="left")
    plt.title(f"{pisa_type}_{model_run}: Input Data Analysis")
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.savefig(path_results_modelrun_plots / f"{pisa_type}_Robustness_Check_{model_run}.png")
    plt.close()

    ###Feature Analysis###
    Importance_Measures = ["Feature Importance", "TEST-Permutation Importance Mean",
                           "TRAIN-Permutation Importance Mean"]
    imp_df = features_df.copy()
    all_measures = []
    colors = ["teal", "crimson", "limegreen", "indigo", "orange", "mediumblue", "salmon",
              "slategray", "lightseagreen", "darkgoldenrod"]
    for importance_measure in Importance_Measures:
        features_df = features_df.sort_values(importance_measure, ascending=False)
        data = features_df[:10].copy()
        data["name"] = [x.split(":")[0] for x in data.index]
        data["labels"] = [str(x) for x in range(len(data))]
        data["color"] = colors[:len(data)]
        data = data.sort_values(importance_measure)

        all_measures.extend(data.index.to_list())

        fig = plt.figure(figsize=(20, 5))
        for f in data.index:
            plt.barh(data.loc[f, "labels"], data.loc[f, importance_measure], align="center",
                     tick_label=data.loc[f, "name"], color=data.loc[f, "color"])

        plt.title(f"{pisa_type}-{input_data}_{iteration}: Top 10 {importance_measure}")
        plt.legend(labels=data.index, ncol=1, bbox_transform=fig.transFigure, loc="lower right",
                   fontsize='xx-small')
        plt.yticks(ticks=np.arange(len(data)), labels=data["name"])
        plt.savefig(path_results_modelrun_plots /
                    f"{pisa_type}_{importance_measure}_{model_run}.png")
        plt.close()

    all_measures = set(all_measures)
    plot_df = features_df.loc[all_measures, :].copy()
    names = [f"{x}: {plot_df.index.to_list()[x]}" for x in range(len(plot_df.index))]
    names_overview = "\n".join(names)

    fig = plt.figure(figsize=(10, 10))
    plt.title(f"{pisa_type}-{model_run}: Measures of Feature Importance")
    plt.scatter(x=range(len(names)), y=plot_df["Feature Importance"], label="Feature Importance")
    plt.scatter(x=range(len(names)), y=plot_df["TEST-Permutation Importance Mean"], label="TEST Permutation Importance")
    plt.scatter(x=range(len(names)), y=plot_df["TRAIN-Permutation Importance Mean"],
                label="TRAIN Permutation Importance")
    plt.xticks(ticks=range(len(names)), fontsize="small")
    plt.xlabel(names_overview, fontsize="x-small", ma="left")
    plt.legend(loc="upper left", fontsize="small")
    plt.gcf().subplots_adjust(bottom=0.3)
    fig.savefig(path_results_modelrun_plots / f"{pisa_type}_Overview_Importance_Measures_{model_run}.png")
    plt.close()

    ###Country-Plots individual###
    estimated_countries = list(results["COUNTRY"].unique())
    estimated_countries_data = data_for_model.loc[data_for_model["COUNTRY"].isin(estimated_countries),
                                                  ["PISA-SCORE", "YEAR", "COUNTRY"]]

    estimated_countries_data = estimated_countries_data.merge(results["y_predict"], how="left", left_index=True,
                                                              right_index=True)
    estimated_countries_data["new_pisa"] = estimated_countries_data["y_predict"]
    estimated_countries_data["new_pisa"] = estimated_countries_data["y_predict"].fillna(
        estimated_countries_data["PISA-SCORE"])
    estimated_countries_data = estimated_countries_data.sort_values(["YEAR"])

    years = list(estimated_countries_data["YEAR"].unique())

    for country in estimated_countries:
        tmp = estimated_countries_data.loc[estimated_countries_data["COUNTRY"] == country].copy()

        plt.figure(figsize=(20, 10))

        s = sns.lineplot(data=tmp, x="YEAR", y="new_pisa", color='r', linestyle='--')
        p = sns.lineplot(data=tmp, x="YEAR", y="PISA-SCORE", marker='o', label="True PISA Score")
        plt.scatter(tmp["YEAR"], tmp["y_predict"], color='r', marker='x', label="Predicted Values", s=100)
        plt.xticks(rotation=45)
        plt.ylabel("Score")
        plt.legend()
        plt.title(f" Score and Predictions for {country}")
        p.get_xaxis().get_major_formatter().set_useOffset(False)
        plt.savefig(path_results_modelrun_plots / f"{pisa_type}_Plot_{country}_{model_run}.png")
        plt.close()

    ###all countries in one plot###
    plt.figure(iteration + 3, figsize=(20, 40))
    for country in estimated_countries:
        tmp = estimated_countries_data.loc[estimated_countries_data["COUNTRY"] == country].copy()

        s = sns.lineplot(data=tmp, x="YEAR", y="new_pisa", color='r', linestyle='--')
        p = sns.lineplot(data=tmp, x="YEAR", y="PISA-SCORE", marker='o', label=f"True PISA Score {country}")
        plt.scatter(tmp["YEAR"], tmp["y_predict"], color='r', marker='x', label=f"Predicted Values {country}", s=100)
        p.get_xaxis().get_major_formatter().set_useOffset(False)

    plt.xticks(rotation=45)
    plt.title(f" Score and Predictions")
    plt.ylabel("Score")
    plt.legend(loc=4)
    plt.savefig(path_results_modelrun_plots / f"{pisa_type}_Plot_all_countries_{model_run}.png")
    plt.close()

results_comparative_modelrun = pd.DataFrame(index=Pisa_Type)

for pisa_type in Pisa_Type:

    if pisa_type == "READ":
        input_data = "aggregated-years"
        iteration = 55
    else:
        input_data = "pisa-years"
        if pisa_type == "MATH":
            iteration = 65
        else:
            iteration = 50

    features_df = pd.read_csv(
        path_results_modelrun / f"{pisa_type}_Feature_Analysis_{model_run}.csv")
    features_df = features_df.set_index("Unnamed: 0")
    results_df = pd.read_csv(path_results_modelrun / f"{pisa_type}_Results_{model_run}.csv")
    results_df = results_df.rename(columns={"Unnamed: 0": "Test_data"})
    results_df = results_df.set_index("Test_data")

    top_features = features_df.index.to_list()[:10]

    results_comparative_modelrun.loc[pisa_type, "Pisa-Type"] = pisa_type
    results_comparative_modelrun.loc[pisa_type, "Percentage non-NA"] = iteration
    results_comparative_modelrun.loc[pisa_type, f"Input-Data"] = input_data
    results_comparative_modelrun.loc[pisa_type, "RMSE"] = results_df.loc[model_run, "RMSE"]
    results_comparative_modelrun.loc[pisa_type, "R2"] = results_df.loc[model_run, "R^2"]
    results_comparative_modelrun.loc[pisa_type, "Total Features"] = results_df.loc[
        model_run, "Total available features"]
    results_comparative_modelrun.loc[pisa_type, "R2 Trainings Data"] = results_df.loc[
        model_run, "R2 Training Data"]
    results_comparative_modelrun.loc[pisa_type, "Depth"] = results_df.loc[model_run, "Depth"]
    results_comparative_modelrun.loc[pisa_type, "min_samples_leaf"] = results_df.loc[
        model_run, "min_samples_leaf"]
    results_comparative_modelrun.loc[pisa_type, "min_samples_split"] = results_df.loc[
        model_run, "min_samples_split"]
    results_comparative_modelrun.loc[pisa_type, "ccp_alpha"] = results_df.loc[model_run, "ccp_alpha"]
    results_comparative_modelrun.loc[pisa_type, "Number of Estimators"] = results_df.loc[
        model_run, "Number of Estimators"]
    results_comparative_modelrun.loc[pisa_type, "Available Features per Estimator"] = results_df.loc[
        model_run, "Percentage of Features"]
    results_comparative_modelrun.loc[pisa_type, "Mean_Approximated_Values[%]"] = features_df[
        "NA-Values [%]"].mean()
    for number in range(0, 10):
        results_comparative_modelrun.loc[pisa_type, f" Top {number + 1} feature, Test-PI"] = top_features[number]

results_comparative_modelrun.to_csv(path_results_modelrun / f"Overview_Random_Forest_Parameters.csv", index=False)
