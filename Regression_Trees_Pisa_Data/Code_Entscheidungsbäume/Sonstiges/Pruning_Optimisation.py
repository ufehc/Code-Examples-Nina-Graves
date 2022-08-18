from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import sklearn.metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import numpy as np
import pandas as pd
from Global_configuartions import FOLDERNAME_MODELLING_DATA
from tqdm import tqdm

Pisa_Type = ["READ", "MATH", "SCIENCE"]
path_0 = Path(FOLDERNAME_MODELLING_DATA)
path_results = Path(path_0 / "Results")
path_tree = Path(path_0 / "Trees")
path_input = Path(path_0 / "Input Data")
model_run = "Pruning_Optimisation"
percentage_test_data = 15

pbar2 = tqdm(Pisa_Type)
for pisa_type in pbar2:
    pbar2.set_description(pisa_type)

    if pisa_type == "READ":
        input_data = "aggregated-years"
        iteration = 55
    else:
        input_data = "pisa-years"
        if pisa_type == "MATH":
            iteration = 65
        else:
            iteration = 50

    # get values of optimal Hyperparameters
    optimal_params = pd.read_csv(Path(
        path_results / f"{input_data}_Optimal_Single_Tree") / f"{pisa_type}_Overview_{input_data}_Optimal_Single_Tree_RS42.csv")
    optimal_params = optimal_params.set_index("Unnamed: 0")

    # Hyper-Parameters of Optimal Single Tree without Pruning
    max_depth = int(optimal_params.loc[iteration, "max_Depth"])
    min_leaf = int(optimal_params.loc[iteration, "min_leaf"])
    min_split = int(optimal_params.loc[iteration, "min_split"])

    Max_Depth = [max(2, max_depth - 1), max(3, max_depth), max(4, max_depth + 1)]
    Min_Leaf = [max(1, min_leaf - 1), max(2, min_leaf), max(3, min_leaf + 1)]
    Min_Split = [max(2, min_split - 1), max(3, min_split), max(4, min_split + 1)]

    # create directories
    path_results.mkdir(exist_ok=True)
    path_results_modelrun = Path(path_results / model_run)
    path_results_modelrun.mkdir(exist_ok=True)
    path_results_modelrun.mkdir(exist_ok=True)
    path_tree.mkdir(exist_ok=True)
    path_tree_modelrun = Path(path_tree / model_run)
    path_tree_modelrun.mkdir(exist_ok=True)

    # load input data and meta data
    data_for_model = pd.read_csv(Path(path_input / input_data) / f"{pisa_type}-{iteration}.csv")
    column_information = pd.read_csv(Path(path_input / input_data) / f"{pisa_type}-Column_information.csv")

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

    # create and train model
    model_tree = DecisionTreeRegressor()
    path = model_tree.cost_complexity_pruning_path(X_train_final, y_train_final)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    alphas_impurities = dict(zip(ccp_alphas, impurities))

    ccp_alphas_selected = [x for x in ccp_alphas if 0.5 <= x <= 3]
    impurities_selected = [alphas_impurities[x] for x in ccp_alphas_selected]

    # Plot Imupority vs. effective alphas - Only for considered alphas

    plt.figure()
    plt.plot(ccp_alphas_selected, impurities_selected, marker='o', drawstyle="steps-post")
    plt.xlabel("effective alpha")
    plt.ylabel("total impurity of leaves")
    plt.title("Total Impurity on considered effective alpha for training set")
    plt.savefig(path_results_modelrun / f"Overview_effective-alphas_{pisa_type}-{iteration}.png")
    plt.close()

    params = {"max_depth": Max_Depth,
              "min_samples_leaf": Min_Leaf,
              "min_samples_split": Min_Split,
              "ccp_alpha": ccp_alphas_selected}

    # Get Hyperparams with Grid Search
    grid_tree = DecisionTreeRegressor()
    grid_search = GridSearchCV(grid_tree, params, n_jobs=-1)
    grid_search.fit(X_train_final, y_train_final)
    params_opt = grid_search.best_params_

    model_tree = DecisionTreeRegressor(max_depth=params_opt["max_depth"],
                                       min_samples_leaf=params_opt['min_samples_leaf'],
                                       min_samples_split=params_opt["min_samples_split"],
                                       ccp_alpha=params_opt["ccp_alpha"])

    model_tree = DecisionTreeRegressor()
    model_tree.fit(X_train_final, y_train_final)

    # evaluate model
    y_predict = model_tree.predict(X_test_final)

    # save tree
    total_nodes = model_tree.tree_.node_count
    features_tree = pd.DataFrame()
    feature_columns = list(X.columns)
    for node_id in range(0, total_nodes):
        if model_tree.tree_.feature[node_id] == -2:
            features_tree.loc[node_id, "Feature"] = "Leaf"
            features_tree.loc[node_id, "Threshold"] = np.nan
        else:
            features_tree.loc[node_id, "Feature"] = feature_columns[model_tree.tree_.feature[node_id]]
            features_tree.loc[node_id, "Threshold"] = model_tree.tree_.threshold[node_id]
        features_tree.loc[node_id, "Samples"] = model_tree.tree_.n_node_samples[node_id]
        features_tree.loc[node_id, "Value"] = model_tree.tree_.value[node_id][0]
    features_merged = features_tree.merge(column_information, how="left", left_on="Feature", right_index=True)
    features_merged.to_csv(
        path_tree_modelrun / f"{pisa_type}-{iteration}_{model_run}.csv")

    # create Feature Information
    tree_features = features_tree.drop(columns=["Threshold", "Value"], axis=1)
    tree_features = tree_features.groupby("Feature").sum()
    tree_features = tree_features.drop("Leaf")

    features = X.columns.to_list()
    features_df = pd.DataFrame(index=features)
    features_df["Feature Importance"] = model_tree.feature_importances_
    features_df = features_df.merge(tree_features["Samples"], how="left", left_index=True, right_index=True)
    features_df = features_df.merge(column_information, how="left", left_index=True, right_index=True)
    features_df["Robustness - Samples affected by NA [Samples]"] = features_df["Samples"] * features_df[
        "NA-Values [%]"] / 100
    features_df["Robustness - Samples affected by Annual Mean [Samples]"] = features_df["Samples"] * \
                                                                            features_df[
                                                                                "Annual Mean [%]"] / 100
    features_df = features_df.sort_values("Samples", ascending=False)
    features_df.to_csv(
        path_results_modelrun / f"{pisa_type}-{iteration}_Feature_Analysis_{model_run}.csv")

    # store model-run data
    alphas_impurities[params_opt["ccp_alpha"]] = 2
    results = X_test_final.copy()
    results = results.drop(columns=feature_columns)
    results["Modelrun"] = model_run
    results["Input-Data"] = input_data
    results = results.merge(data_for_model[["COUNTRY", "YEAR"]], on="Index")
    results["PISA-SCORE"] = y_test_final
    results["y_predict"] = y_predict
    results["Absoloute Error"] = abs(results["PISA-SCORE"] - results["y_predict"])
    results["Squared Error"] = abs(results["PISA-SCORE"] - results["y_predict"]) ** 2
    results["RMSE"] = np.sqrt(results["Squared Error"])
    results["R^2"] = np.nan
    results["Nodes_Tree"] = np.nan
    results["alpha_Pruning"] = np.nan
    new_row_df = pd.DataFrame(columns=["Modelrun", "COUNTRY", "PISA-SCORE", "y_predict", "Absoloute Error",
                                       "Squared Error", "RMSE", "R^2", "Nodes_Tree", ],
                              data=[[model_run, np.nan, np.nan, np.nan,
                                     sklearn.metrics.mean_absolute_error(y_test_final, y_predict),
                                     sklearn.metrics.mean_squared_error(y_test_final, y_predict),
                                     np.sqrt(sklearn.metrics.mean_squared_error(y_test_final, y_predict)),
                                     model_tree.score(X_test_final, y_test_final), total_nodes]],
                              index=[model_run])
    results_with_total_overview = pd.concat([new_row_df, results])
    results_with_total_overview.loc[model_run, "Impurity"] = alphas_impurities[params_opt["ccp_alpha"]]
    results_with_total_overview.loc[model_run, "Percentage Test-Data"] = percentage_test_data
    results_with_total_overview.loc[model_run, "Depth"] = model_tree.get_depth()
    results_with_total_overview.loc[model_run, "min_leaf"] = params_opt['min_samples_leaf']
    results_with_total_overview.loc[model_run, "min_split"] = params_opt['min_samples_split']
    results_with_total_overview.loc[model_run, "Alpha"] = params_opt['ccp_alpha']
    results_with_total_overview.loc[model_run, "Number of Features in Tree"] = len(tree_features)
    results_with_total_overview.to_csv(
        path_results_modelrun / f"{pisa_type}-{iteration}_Results_{model_run}.csv")

    ######### Plots #############
    path_results_modelrun_plots = Path(
        path_results_modelrun / f"{pisa_type}-{iteration}_Plots")
    path_results_modelrun_plots.mkdir(exist_ok=True)
    ###Prediction, True Scores ###
    plt.figure(iteration, figsize=(20, 10))
    plot_x_data = X_test_final.sort_values(["YEAR"])
    sc_plot_predict = sns.scatterplot(plot_x_data.index, y_test_final,
                                      hue=data_for_model.loc[X_test_final.index, "COUNTRY"])
    plt.xticks(rotation=45)
    plt.plot(y_predict, c="r", label="Prediction")
    plt.legend(loc=1)
    plt.savefig(path_results_modelrun_plots / f"RESULT_{pisa_type}-{iteration}.png")
    plt.close()

    ###Tree####
    export_graphviz(model_tree,
                    out_file=f"{pisa_type}-{iteration}_Tree_{model_run}_with_names.dot",
                    feature_names=feature_columns)
    export_graphviz(model_tree, out_file=f"{pisa_type}-{iteration}_Tree_{model_run}.dot")

    ### Absolute Abweichung ###
    plt.figure(iteration + 4)
    plt.scatter(results["PISA-SCORE"], results["RMSE"])
    plt.xlabel("True PISA Score")
    plt.ylabel("RMSE")
    plt.savefig(
        path_results_modelrun_plots / f"{pisa_type}-{iteration}_RMSE-True_Score.png")
    plt.close()

    ### Average Error per year ###
    plt.figure(iteration + 1)
    tmp = results[["YEAR", "RMSE"]].groupby("YEAR").mean()
    plt.plot(tmp.index, tmp, marker="o", label="Average Error")
    plt.hist(results["YEAR"], bins=100, align="left", label="Frequency in Test-Data")
    plt.xticks(tmp.index)
    plt.legend()
    plt.title("Average RMSE per year")
    plt.savefig(
        path_results_modelrun_plots / f"{pisa_type}-{iteration}_RMSE_per_year.png")
    plt.close()

    ### Robustness-Check ###
    plt.figure(iteration + 2, figsize=(30, 10))
    data = features_df[:10].copy()
    feature_names = features_df.index.to_list()[:10]
    feature_names = [f"{x}: {feature_names[x]}" for x in range(len(feature_names))]
    feature_names = "\n".join(feature_names)

    plt.plot(range(len(data)), data["Annual Mean"], marker="x", c="r", label="NAs approximated by annual mean")
    plt.plot(range(len(data)), data["NA-Values"], marker="x", c="b", label="NA-Values")
    plt.hist(data.index, bins=range(len(data)), color="slategray", align="left", rwidth=0.5,
             weights=data["Samples"], label="# NA-Values")
    plt.legend()
    plt.xticks(range(len(data.index)), range(len(data)));
    plt.xlabel(feature_names, fontsize="x-small", ma="left")
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.savefig(path_results_modelrun_plots / f"{pisa_type}-{iteration}_Robustness_Check_{model_run}.png")
    plt.close()

    ### Only Feature  ###
    colors = ["teal", "crimson", "limegreen", "indigo", "orange", "mediumblue", "salmon",
              "slategray", "lightseagreen", "darkgoldenrod"]
    data["name"] = [x.split(":")[0] for x in data.index]
    data["labels"] = [str(x) for x in range(len(data))]
    data["color"] = colors[:len(data)]

    fig = plt.figure(figsize=(20, 5))
    for f in data.index:
        plt.barh(data.loc[f, "labels"], data.loc[f, "Samples"], align="center",
                 tick_label=data.loc[f, "name"], color=data.loc[f, "color"])

    plt.title(f"{pisa_type}-{input_data}_{iteration}: Top 10 Samples")
    plt.legend(labels=data.index, ncol=1, bbox_transform=fig.transFigure, loc="upper right",
               fontsize='xx-small')
    plt.yticks(ticks=np.arange(len(data)), labels=data["name"])
    plt.savefig(path_results_modelrun_plots /
                f"{pisa_type}-{iteration}_Important_Features_{model_run}_.png")
    plt.close()

    ###Country-Plots individual###
    estimated_countries = list(results["COUNTRY"].unique())
    estimated_countries_data = data_for_model.loc[data_for_model["COUNTRY"].isin(estimated_countries),
                                                  ["PISA-SCORE", "YEAR", "COUNTRY"]]

    estimated_countries_data = estimated_countries_data.merge(results["y_predict"], how="left",
                                                              left_index=True,
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
        plt.savefig(
            path_results_modelrun_plots / f"{pisa_type}-{iteration}_Plot_{country}.png")
        plt.close()

    ###all countries in one plot###
    plt.figure(iteration + 3, figsize=(20, 40))
    for country in estimated_countries:
        tmp = estimated_countries_data.loc[estimated_countries_data["COUNTRY"] == country].copy()

        s = sns.lineplot(data=tmp, x="YEAR", y="new_pisa", color='r', linestyle='--')
        p = sns.lineplot(data=tmp, x="YEAR", y="PISA-SCORE", marker='o', label=f"True PISA Score {country}")
        plt.scatter(tmp["YEAR"], tmp["y_predict"], color='r', marker='x',
                    label=f"Predicted Values {country}", s=100)
        p.get_xaxis().get_major_formatter().set_useOffset(False)

    plt.xticks(rotation=45)
    plt.title(f" Score and Predictions")
    plt.ylabel("Score")
    plt.legend(loc=4)
    plt.savefig(
        path_results_modelrun_plots / f"{pisa_type}-{iteration}_Plot_all_countries.png")
    plt.close()

# Overview Table for all pisas
results_comparative_modelrun = pd.DataFrame(
    columns=["Percentage non-NA", "Total Features", "Features Tree", "RMSE",
             "R2", "Percentage_NA_mean"], index=Pisa_Type)

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

    total_features = pd.read_csv(
        Path(path_input / input_data) / f"{pisa_type}-{iteration}.csv")

    features_df = pd.read_csv(
        path_results_modelrun / f"{pisa_type}-{iteration}_Feature_Analysis_{model_run}.csv")
    top_features = features_df[:10]

    results_df = pd.read_csv(
        path_results_modelrun / f"{pisa_type}-{iteration}_Results_{model_run}.csv")
    results_df = results_df.rename(columns={"Unnamed: 0": "Test_data"})
    results_df = results_df.set_index("Test_data")

    results_comparative_modelrun.loc[pisa_type, "Input-Data"] = input_data
    results_comparative_modelrun.loc[pisa_type, "Percentage non-NA"] = iteration
    results_comparative_modelrun.loc[pisa_type, "RMSE"] = results_df.loc[model_run, "RMSE"]
    results_comparative_modelrun.loc[pisa_type, "R2"] = results_df.loc[model_run, "R^2"]
    results_comparative_modelrun.loc[pisa_type, "max_Depth"] = results_df.loc[model_run, "Depth"]
    results_comparative_modelrun.loc[pisa_type, "Nodes_Tree"] = results_df.loc[model_run, "Nodes_Tree"]
    results_comparative_modelrun.loc[pisa_type, "Impurity"] = results_df.loc[model_run, "Impurity"]
    results_comparative_modelrun.loc[pisa_type, "min_leaf"] = results_df.loc[model_run, "min_leaf"]
    results_comparative_modelrun.loc[pisa_type, "min_split"] = results_df.loc[model_run, "min_split"]
    results_comparative_modelrun.loc[pisa_type, "Alpha"] = results_df.loc[model_run, "Alpha"]
    results_comparative_modelrun.loc[pisa_type, "Number of Features in Tree"] = results_df.loc[
        model_run, "Number of Features in Tree"]
    results_comparative_modelrun.loc[pisa_type, "Total Features"] = len(total_features.columns) - 3
    results_comparative_modelrun.loc[pisa_type, "Percentage_NA_mean"] = features_df[
        "Annual Mean [%]"].mean()
    results_comparative_modelrun.loc[pisa_type, "Mean_Approximated_Values[%]"] = features_df[
        "NA-Values [%]"].mean()
    for number in range(0, 10):
        results_comparative_modelrun.loc[pisa_type, f" Top {number + 1} feature"] = top_features.loc[
            number, "Unnamed: 0"]

    results_comparative_modelrun.to_csv(
        path_results_modelrun / f"Overview_all_Pisas_{model_run}.csv")

