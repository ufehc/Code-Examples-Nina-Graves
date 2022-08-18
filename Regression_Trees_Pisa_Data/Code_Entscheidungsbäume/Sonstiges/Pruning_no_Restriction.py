from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from Global_configuartions import FOLDERNAME_MODELLING_DATA
from tqdm import tqdm
import pandas as pd
import numpy as np

Iterations = np.arange(50, 105, 5)
Input_Data = ["aggregated-years", "pisa-years"]
Pisa_Type = ["READ", "MATH", "SCIENCE"]
path_0 = Path(FOLDERNAME_MODELLING_DATA)
path_results = Path(path_0 / "Results")
path_tree = Path(path_0 / "Trees")
path_input = Path(path_0 / "Input Data")
model_run_name = "Pruning_no_restriction_RS42"
percentage_test_data = 15

pbar = tqdm(Input_Data)
for input_data in pbar:
    pbar.set_description(input_data)
    model_run = f"{input_data}_{model_run_name}"

    pbar2 = tqdm(Pisa_Type)
    for pisa_type in pbar2:
        pbar2.set_description(pisa_type)

        # create directories
        path_results.mkdir(exist_ok=True)
        path_results_modelrun = Path(path_results / model_run)
        path_results_modelrun.mkdir(exist_ok=True)
        path_results_modelrun.mkdir(exist_ok=True)
        path_tree.mkdir(exist_ok=True)
        path_tree_modelrun = Path(path_tree / model_run)
        path_tree_modelrun.mkdir(exist_ok=True)

        pbar3 = tqdm(Iterations)
        for iteration in pbar3:
            pbar.set_description(f"{input_data}: {pisa_type}-{iteration}")

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

            # Plot Impurity vs. effective alphas
            plt.figure()
            plt.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
            plt.xlabel("effective alpha")
            plt.ylabel("total impurity of leaves")
            plt.xscale("log")
            plt.title(f"Impurity on Alphas for PISA-{pisa_type}")
            plt.savefig(path_results_modelrun / f"Overview_xLog_effective-alphas_{pisa_type}-{iteration}.png")
            plt.close()

            plt.figure()
            plt.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
            plt.xlabel("effective alpha")
            plt.ylabel("total impurity of leaves")
            plt.xscale("log")
            plt.yscale("log")
            plt.title("Total Impurity vs effective alpha for training set")
            plt.savefig(path_results_modelrun / f"Overview_x_y_Log_effective-alphas_{pisa_type}-{iteration}.png")
            plt.close()

            plt.figure()
            plt.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
            plt.xlabel("effective alpha")
            plt.ylabel("total impurity of leaves")
            plt.title("Total Impurity vs effective alpha for training set")
            plt.savefig(path_results_modelrun / f"Overview_effective-alphas_{pisa_type}-{iteration}.png")
            plt.close()

            regressors = []
            for ccp_alpha in ccp_alphas:
                model_tree = DecisionTreeRegressor(ccp_alpha=ccp_alpha)
                model_tree.fit(X_train_final, y_train_final)
                regressors.append(model_tree)

                # evaluate model
                y_predict = model_tree.predict(X_test_final)

                # save tree
                total_nodes = model_tree.tree_.node_count
                features = pd.DataFrame()
                feature_columns = list(X.columns)
                for node_id in range(0, total_nodes):
                    if model_tree.tree_.feature[node_id] == -2:
                        features.loc[node_id, "Feature"] = "Leaf"
                        features.loc[node_id, "Threshold"] = np.nan
                    else:
                        features.loc[node_id, "Feature"] = feature_columns[model_tree.tree_.feature[node_id]]
                        features.loc[node_id, "Threshold"] = model_tree.tree_.threshold[node_id]
                    features.loc[node_id, "Samples"] = model_tree.tree_.n_node_samples[node_id]
                    features.loc[node_id, "Value"] = model_tree.tree_.value[node_id][0]
                features_merged = features.merge(column_information, how="left", left_on="Feature", right_index=True)
                features_merged.to_csv(
                    path_tree_modelrun / f"{pisa_type}-{iteration}_Alpha_{ccp_alpha}_{model_run}.csv")

                # create Robustness Information
                robust_check = features.drop(columns=["Threshold", "Value"], axis=1)
                robust_check = robust_check.groupby("Feature").sum()
                robust_check = robust_check.merge(column_information, how="left", left_on="Feature", right_index=True)
                robust_check["Robustness - Samples affected by NA [Samples]"] = robust_check["Samples"] * robust_check[
                    "NA-Values [%]"] / 100
                robust_check["Robustness - Samples affected by Annual Mean [Samples]"] = robust_check["Samples"] * \
                                                                                         robust_check[
                                                                                             "Annual Mean [%]"] / 100
                robust_check.to_csv(
                    path_results_modelrun / f"{pisa_type}-{iteration}_Feature_Analysis_Alpha_{ccp_alpha}_{model_run}.csv")

                # store model-run data
                results = X_test_final.copy()
                results = results.drop(columns=feature_columns)
                results["Modelrun"] = model_run
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
                                                   "Squared Error", "RMSE", "R^2", "Nodes_Tree", "alpha_Pruning"],
                                          data=[[model_run, np.nan, np.nan, np.nan,
                                                 sklearn.metrics.mean_absolute_error(y_test_final, y_predict),
                                                 sklearn.metrics.mean_squared_error(y_test_final, y_predict),
                                                 np.sqrt(sklearn.metrics.mean_squared_error(y_test_final, y_predict)),
                                                 model_tree.score(X_test_final, y_test_final), total_nodes, ccp_alpha]],
                                          index=[model_run])
                results_with_total_overview = pd.concat([new_row_df, results])
                results_with_total_overview.loc[model_run, "Impurity"] = alphas_impurities[ccp_alpha]
                results_with_total_overview.loc[model_run, "R2 Training Data"] = model_tree.score(X_train_final,
                                                                                                  y_train_final)
                results_with_total_overview.loc[model_run, "Percentage Test-Data"] = percentage_test_data
                results_with_total_overview.loc[model_run, "max_Depth"] = model_tree.get_depth()
                results_with_total_overview.to_csv(
                    path_results_modelrun / f"{pisa_type}-{iteration}_Results_Alpha_{ccp_alpha}_{model_run}.csv")

                ######### Plots #############
                path_results_modelrun_plots = Path(
                    path_results_modelrun / f"{pisa_type}-{iteration}_Plots_Alpha_{ccp_alpha}")
                path_results_modelrun_plots.mkdir(exist_ok=True)
                ###Prediction, True Scores ###
                plt.figure(iteration, figsize=(20, 10))
                plot_x_data = X_test_final.sort_values(["YEAR"])
                sc_plot_predict = sns.scatterplot(plot_x_data.index, y_test_final,
                                                  hue=data_for_model.loc[X_test_final.index, "COUNTRY"])
                plt.xticks(rotation=45)
                plt.plot(y_predict, c="r", label="Prediction")
                plt.legend(loc=1)
                plt.savefig(path_results_modelrun_plots / f"RESULT_{pisa_type}-{iteration}_Alpha_{ccp_alpha}.png")
                plt.close()

                ###Tree####
                export_graphviz(model_tree,
                                out_file=f"{pisa_type}-{iteration}_Tree_Alpha_{ccp_alpha}_{model_run}_with_names.dot",
                                feature_names=feature_columns)
                export_graphviz(model_tree, out_file=f"{pisa_type}-{iteration}_Tree_Alpha_{ccp_alpha}_{model_run}.dot")

                ### Absolute Abweichung ###
                plt.figure(iteration + 4)
                plt.scatter(results["PISA-SCORE"], results["RMSE"])
                plt.xlabel("True PISA Score")
                plt.ylabel("RMSE")
                plt.savefig(
                    path_results_modelrun_plots / f"{pisa_type}-{iteration}_Alpha_{ccp_alpha}_RMSE-True_Score.png")
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
                    path_results_modelrun_plots / f"{pisa_type}-{iteration}_Alpha_{ccp_alpha}_RMSE_per_year.png")
                plt.close()

                ### Robustness-Check ###
                plt.figure(iteration + 2, figsize=(30, 10))
                robust_check = robust_check.sort_values("Samples", ascending=False)
                relevant_features = robust_check.loc[robust_check["Samples"] >= 20]
                relevant_features = relevant_features.drop(["Leaf"], axis=0)
                plt.plot(range(len(relevant_features)), relevant_features["NA-Values"], marker="o", c="b",
                         label="# NA-Values")
                plt.plot(range(len(relevant_features)), relevant_features["Annual Mean"], marker="x", c="r",
                         label="Annual Mean used for NaN")
                plt.hist(relevant_features.index, bins=range(len(relevant_features)), color="grey", align="left",
                         rwidth=0.5,
                         weights=relevant_features["Samples"], label="Number of Samples using the feature")
                plt.legend()
                plt.xticks(range(len(relevant_features.index)), range(len(relevant_features)),
                           horizontalalignment="left", rotation=-45);
                plt.savefig(
                    path_results_modelrun_plots / f"{pisa_type}-{iteration}_Alpha_{ccp_alpha}_Robustness_Check.png")
                plt.close()

                ###Country-Plots individual###estimated_countries = list(results["COUNTRY"].unique())
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
                        path_results_modelrun_plots / f"{pisa_type}-{iteration}_Plot_{country}_Alpha_{ccp_alpha}.png")
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
                    path_results_modelrun_plots / f"{pisa_type}-{iteration}_Plot_all_countries_Alpha_{ccp_alpha}.png")
                plt.close()

            # when all alphas run through - Plots for different alphas
            # plot showing different depths
            regressors = regressors[:-1]
            ccp_alphas_no_root = ccp_alphas[:-1]

            node_counts = [regressor.tree_.node_count for regressor in regressors]
            depth = [regressor.tree_.max_depth for regressor in regressors]
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(ccp_alphas_no_root, node_counts, marker='o', drawstyle="steps-post")
            ax[0].set_xlabel("alpha")
            ax[0].set_ylabel("number of nodes")
            ax[0].set_title(f"Number of nodes per alpha for PISA-{pisa_type}")
            ax[1].plot(ccp_alphas_no_root, depth, marker='o', drawstyle="steps-post")
            ax[1].set_xlabel("alpha")
            ax[1].set_ylabel("depth of tree")
            ax[1].set_title(f"Depth per alpha for PISA-{pisa_type}")
            fig.savefig(path_results_modelrun_plots / f"{pisa_type}-{iteration}_Pruning_Overview_Nodes_Depth.png")

            # plot showing scores per alpha
            train_scores = [regressor.score(X_train_final, y_train_final) for regressor in regressors]
            test_scores = [regressor.score(X_test_final, y_test_final) for regressor in regressors]

            fig, ax = plt.subplots()
            ax.set_xlabel("alpha")
            ax.set_ylabel("R2")
            ax.set_title(f"R2 per alpha for training and testing sets for PISA-{pisa_type}")
            ax.plot(ccp_alphas_no_root, train_scores, marker='o', label="train",
                    drawstyle="steps-post")
            ax.plot(ccp_alphas_no_root, test_scores, marker='o', label="test",
                    drawstyle="steps-post")
            ax.legend()
            plt.show()
            fig.savefig(path_results_modelrun_plots / f"{pisa_type}-{iteration}_Pruning_Overview_Scores.png")

            # Overview Table for all alphas
            results_comparative_modelrun_alphas = pd.DataFrame(
                columns=["Percentage non-NA", "Total Features", "Features Tree", "RMSE",
                         "R2", "Percentage_NA_mean"], index=ccp_alphas)
            for ccp_alpha in ccp_alphas:

                if input_data == "pisa-years":
                    total_features = pd.read_csv(
                        Path(path_input / "0_no_aggregation_default") / f"{pisa_type}-{iteration}.csv")
                else:
                    total_features = pd.read_csv(
                        Path(path_input / "1_aggregation_weighted_average") / f"{pisa_type}-{iteration}.csv")

                features_df = pd.read_csv(
                    path_results_modelrun / f"{pisa_type}-{iteration}_Feature_Analysis_Alpha_{ccp_alpha}_{model_run}.csv")
                results_df = pd.read_csv(
                    path_results_modelrun / f"{pisa_type}-{iteration}_Results_Alpha_{ccp_alpha}_{model_run}.csv")
                results_df = results_df.rename(columns={"Unnamed: 0": "Test_data"})
                results_df = results_df.set_index("Test_data")

                results_comparative_modelrun_alphas.loc[ccp_alpha, "Alpha"] = ccp_alpha
                results_comparative_modelrun_alphas.loc[ccp_alpha, "Percentage non-NA"] = iteration
                results_comparative_modelrun_alphas.loc[ccp_alpha, "RMSE"] = results_df.loc[model_run, "RMSE"]
                results_comparative_modelrun_alphas.loc[ccp_alpha, "R2"] = results_df.loc[model_run, "R^2"]
                results_comparative_modelrun_alphas.loc[ccp_alpha, "max_Depth"] = results_df.loc[model_run, "Depth"]
                results_comparative_modelrun_alphas.loc[ccp_alpha, "Nodes_Tree"] = results_df.loc[
                    model_run, "Nodes_Tree"]
                results_comparative_modelrun_alphas.loc[ccp_alpha, "Impurity"] = results_df.loc[model_run, "Impurity"]
                results_comparative_modelrun_alphas.loc[ccp_alpha, "Features Tree"] = len(features_df)
                results_comparative_modelrun_alphas.loc[ccp_alpha, "Total Features"] = len(total_features.columns) - 3
                results_comparative_modelrun_alphas.loc[ccp_alpha, "R2 Training Data"] = results_df.loc[
                    model_run, "R2 Training Data"]
                results_comparative_modelrun_alphas.loc[ccp_alpha, "Percentage_NA_mean"] = features_df[
                    "Annual Mean [%]"].mean()
                results_comparative_modelrun_alphas.loc[ccp_alpha, "Mean_Approximated_Values[%]"] = features_df[
                    "NA-Values [%]"].mean()
                results_comparative_modelrun_alphas = results_comparative_modelrun_alphas.drop_duplicates()

            results_comparative_modelrun_alphas.to_csv(
                path_results_modelrun / f"{pisa_type}_{iteration}_Overview_All_Alphas_{model_run}.csv")

        # overview for all models of that pisa-type
        results_comparative_modelrun = pd.DataFrame(
            columns=["Percentage non-NA", "Alpha", "Total Features", "Features Tree",
                     "RMSE", "R2", "max_Depth", "Nodes_Tree", "Impurities",
                     "Percentage_NA_mean", "Mean_Approximated_Values[%]"])
        for iteration in Iterations:

            overview_iteration = pd.read_csv(
                path_results_modelrun / f"{pisa_type}_{iteration}_Overview_All_Alphas_{model_run}.csv")
            overview_iteration = overview_iteration.rename(columns={"Unnamed: 0": "ccp_alphas"})
            overview_iteration = overview_iteration.set_index("ccp_alphas")

            index = list(itertools.product([iteration], overview_iteration["Alpha"]))
            frame_iteration = pd.DataFrame(columns=["Percentage non-NA", "Alpha", "Total Features", "Features Tree",
                                                    "RMSE", "R2", "max_Depth", "Nodes_Tree", "Impurities",
                                                    "Percentage_NA_mean", "Mean_Approximated_Values[%]"],
                                           index=pd.MultiIndex.from_tuples(index))

            for alpha_it in overview_iteration.index:
                frame_iteration.loc[(iteration, alpha_it), "Percentage non-NA"] = iteration
                frame_iteration.loc[(iteration, alpha_it), "Alpha"] = alpha_it
                frame_iteration.loc[(iteration, alpha_it), "RMSE"] = overview_iteration.loc[alpha_it, "RMSE"]
                frame_iteration.loc[(iteration, alpha_it), "R2"] = overview_iteration.loc[alpha_it, "R2"]
                frame_iteration.loc[(iteration, alpha_it), "max_Depth"] = overview_iteration.loc[alpha_it, "max_Depth"]
                frame_iteration.loc[(iteration, alpha_it), "Nodes_Tree"] = overview_iteration.loc[
                    alpha_it, "Nodes_Tree"]
                frame_iteration.loc[(iteration, alpha_it), "Impurities"] = overview_iteration.loc[alpha_it, "Impurity"]
                frame_iteration.loc[(iteration, alpha_it), "Features Tree"] = overview_iteration.loc[
                    alpha_it, "Features Tree"]
                frame_iteration.loc[(iteration, alpha_it), "Total Features"] = overview_iteration.loc[
                    alpha_it, "Total Features"]
                frame_iteration.loc[(iteration, alpha_it), "Percentage_NA_mean"] = overview_iteration.loc[
                    alpha_it, "Percentage_NA_mean"]
                frame_iteration.loc[(iteration, alpha_it), "Mean_Approximated_Values[%]"] = overview_iteration.loc[
                    alpha_it, "Mean_Approximated_Values[%]"]

            results_comparative_modelrun = results_comparative_modelrun.append(frame_iteration)

        results_comparative_modelrun.to_csv(path_results_modelrun / f"{pisa_type}_Overview_{model_run}.csv")
