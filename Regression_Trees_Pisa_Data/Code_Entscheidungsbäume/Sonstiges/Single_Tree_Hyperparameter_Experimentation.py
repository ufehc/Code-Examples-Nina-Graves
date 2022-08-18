#%%

from pathlib import Path
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from Global_configuartions import FOLDERNAME_MODELLING_DATA
from tqdm import tqdm
import pandas as pd
import numpy as np


# Iterations = np.arange(50,105,5)
# Input_Data = ["aggregated-years", "pisa-years"]
# Pisa_Type = ["MATH", "READ", "SCIENCE"]
Iterations = [90]
Pisa_Type = ["MATH"]
Input_Data = ["pisa-years"]
Model_Run = ["criterion", "max_depth", "min_sample_split", "min_sample_leaf"]
Hyper_Parameter = {"criterion": ["mae", "mse", "friedman_mse"],
                      "max_depth": [4, 5, 6, 7, 8, 9, 10],
                      "min_sample_split": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                   "min_sample_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
path_0 = Path(FOLDERNAME_MODELLING_DATA)
path_results = Path(path_0/"Results")
path_tree = Path(path_0/"Trees")
path_input = Path(path_0/"Input Data")
percentage_test_data = 15

pbar = tqdm(Input_Data)
for input_data in pbar:
    pbar.set_description(input_data)

    pbar_pisa = tqdm(Pisa_Type)
    for pisa_type in pbar_pisa:
        pbar_pisa.set_description(f"{input_data}-{pisa_type}")
    pbar1 = tqdm(Model_Run)
    for model_run in pbar1:
        pbar1.set_description(model_run)


        pbar1 = tqdm(Hyper_Parameter[model_run])
        for h_param in pbar1:
            pbar1.set_description(f"{model_run}{h_param}")
            # create directories
            path_results.mkdir(exist_ok=True)
            path_results_modelrun = Path(path_results / f"{input_data}_Hyper_{model_run}")
            path_results_modelrun.mkdir(exist_ok=True)
            path_results_modelrun.mkdir(exist_ok=True)
            path_tree.mkdir(exist_ok=True)
            path_tree_modelrun = Path(path_tree/f"{input_data}_Hyper_{model_run}")
            path_tree_modelrun.mkdir(exist_ok=True)

            pbar3 = tqdm(Iterations)
            for iteration in pbar3:
                pbar3.set_description(f"{model_run}{h_param}{iteration}")

                # load input data and meta data
                data_for_model = pd.read_csv(Path(path_input/input_data)/f"{pisa_type}-{iteration}.csv")
                column_information = pd.read_csv(Path(path_input/input_data) / f"{pisa_type}-Column_information.csv")

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
                if model_run == "criterion":
                    model_tree = DecisionTreeRegressor(criterion=h_param)
                if model_run == "min_samples_split":
                    model_tree = DecisionTreeRegressor(min_samples_split=h_param)
                if model_run == "max_depth":
                    model_tree = DecisionTreeRegressor(max_depth=h_param)
                if model_run == "min_samples_leaf":
                    model_tree = DecisionTreeRegressor(min_samples_leaf=h_param)

                # rain model
                model_tree.fit(X_train_final, y_train_final)

                # evaluate model
                y_predict = model_tree.predict(X_test_final)

                # get tree information - iterate through the nodes of the Tree
                # merge with information about the feature
                # store as representation of tree
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
                features_merged.to_csv(path_tree_modelrun / f"{pisa_type}-{iteration}_{model_run}_{h_param}.csv")

                # create Robustness Information
                # Identify how often the different features appear in relation to the training samples in the tree
                # sort according to samples - importance of that feature pursumed by the total number of samples at the feature
                robust_check = features.drop(columns=["Threshold", "Value"], axis=1)
                robust_check = robust_check.groupby(
                    "Feature").sum()  # sum up the number of samples at nodes of the feature
                robust_check = robust_check.sort_values("Samples", ascending=False)
                robust_check = robust_check.drop("Leaf")  # drop leaf node - not relevant to feature analysis
                robust_check = robust_check.merge(column_information, how="left", left_on="Feature", right_index=True)

                # calculate how affected the samples passing through the node are by the approximation of  na values by the annual mean
                robust_check["Robustness - Samples affected by NA [Samples]"] = robust_check["Samples"] * robust_check["NA-Values [%]"] / 100
                robust_check["Robustness - Samples affected by Annual Mean [Samples]"] = robust_check["Samples"] * robust_check["Annual Mean [%]"] / 100
                robust_check.to_csv(path_results_modelrun/ f"{pisa_type}-{iteration}_Feature_Analysis_{model_run}_{h_param}.csv")

                # store model-run data
                results = X_test_final.copy()
                results = results.drop(columns=feature_columns)
                results["Modelrun"] = model_run
                results = results.merge(data_for_model[["COUNTRY", "YEAR"]], on="Index")
                results["PISA-SCORE"] = y_test_final
                results["y_predict"] = y_predict
                results["Absoloute Error"] = abs(results["PISA-SCORE"]-results["y_predict"])
                results["Squared Error"] = abs(results["PISA-SCORE"]-results["y_predict"])**2
                results["RMSE"] = np.sqrt(results["Squared Error"])
                results["R^2"] = np.nan
                new_row_df = pd.DataFrame(columns=["Modelrun", "COUNTRY", "PISA-SCORE", "y_predict", "Absoloute Error", "Squared Error", "RMSE", "R^2"],
                                       data=[[model_run, np.nan, np.nan, np.nan, sklearn.metrics.mean_absolute_error(y_test_final, y_predict),
                                        sklearn.metrics.mean_squared_error(y_test_final, y_predict),
                                        np.sqrt(sklearn.metrics.mean_squared_error(y_test_final, y_predict)),
                                        model_tree.score(X_test_final, y_test_final)]], index=[model_run])
                results_with_total_overview = pd.concat([new_row_df, results])
                results_with_total_overview.loc[model_run, "Percentage Test-Data"] = percentage_test_data
                results_with_total_overview.loc[model_run, "Depth"] = model_tree.get_depth()
                results_with_total_overview.loc[model_run, model_run] = h_param
                results_with_total_overview.to_csv(path_results_modelrun / f"{pisa_type}-{iteration}_Results_{model_run}_{h_param}.csv")

                ######### Plots ##############
                path_results_modelrun_plots = Path(path_results_modelrun / f"{pisa_type}-{iteration}_Plots")
                path_results_modelrun_plots.mkdir(exist_ok=True)
                ###Prediction, True Scores ###
                test_plot = X_test_final.copy()
                test_plot = test_plot.merge(data_for_model.loc[X_test_final.index, "COUNTRY"], how="left",
                                            left_index=True,
                                            right_index=True)
                test_plot = test_plot.merge(y_test_final, how="left", left_index=True, right_index=True)
                test_plot = test_plot.sort_values(["YEAR"])
                plt.figure(iteration, figsize=(20, 10))
                sc_plot_predict = sns.scatterplot(test_plot.index, test_plot["PISA-SCORE"],
                                                  hue=test_plot["COUNTRY"])
                plt.xticks(rotation=45)
                plt.plot(y_predict, c="r", label="Prediction")
                plt.legend(loc=1)
                plt.title(f"{pisa_type}-{input_data}_{iteration}_{model_run}_{h_param}: True Scores vs. Predictions")
                plt.savefig(path_results_modelrun_plots / f"RESULT_{pisa_type}-{iteration}_{model_run}_{h_param}.png")
                plt.close()

                ###Tree####
                export_graphviz(model_tree, out_file=f"{pisa_type}-{iteration}_Tree_{model_run}_{h_param}_with_names.dot", feature_names=feature_columns)
                export_graphviz(model_tree, out_file=f"{pisa_type}-{iteration}_Tree_{model_run}_{h_param}.dot")

                ### Absolute Abweichung ###
                plt.figure(iteration+4)
                plt.scatter(results["PISA-SCORE"], results["RMSE"])
                plt.xlabel("True PISA Score")
                plt.ylabel("RMSE")
                plt.savefig(path_results_modelrun_plots / f"{pisa_type}-{iteration}_RMSE-True_Score_{model_run}_{h_param}.png")
                plt.close()

                ### Average Error per year ###
                plt.figure(iteration+1)
                tmp = results[["YEAR", "RMSE"]].groupby("YEAR").mean()
                plt.plot(tmp.index, tmp, marker="o", label="Average Error")
                plt.hist(results["YEAR"], bins=100, align="left", label="Frequency in Test-Data")
                plt.xticks(tmp.index)
                plt.legend()
                plt.title("Average RMSE per year")
                plt.savefig(path_results_modelrun_plots / f"{pisa_type}-{iteration}_RMSE_per_year_{model_run}_{h_param}.png")
                plt.close()

                ### Robustness-Check ###
                plt.figure(iteration+2, figsize=(30,10))
                robust_check = robust_check.sort_values("Samples", ascending=False)
                relevant_features = robust_check.loc[robust_check["Samples"]>=20]
                relevant_features = relevant_features.drop(["Leaf"], axis=0)
                plt.plot(range(len(relevant_features)), relevant_features["NA-Values"], marker="o", c="b", label="# NA-Values")
                plt.plot(range(len(relevant_features)), relevant_features["Annual Mean"], marker="x", c="r", label="Annual Mean used for NaN")
                plt.hist(relevant_features.index, bins=range(len(relevant_features)), color="grey", align="left", rwidth=0.5,
                         weights=relevant_features["Samples"], label="Number of Samples using the feature")
                plt.legend()
                plt.xticks(range(len(relevant_features.index)), range(len(relevant_features)), horizontalalignment="left", rotation=-45);
                plt.savefig(path_results_modelrun_plots / f"{pisa_type}-{iteration}_Robustness_Check_{model_run}_{h_param}.png")
                plt.close()

                ###Country-Plots individual###estimated_countries = list(results["COUNTRY"].unique())
                estimated_countries = list(results["COUNTRY"].unique())
                estimated_countries_data = data_for_model.loc[data_for_model["COUNTRY"].isin(estimated_countries),
                                                                 ["PISA-SCORE", "YEAR", "COUNTRY"]]

                estimated_countries_data = estimated_countries_data.merge(results["y_predict"], how="left", left_index=True,
                                                                          right_index= True)
                estimated_countries_data["new_pisa"] = estimated_countries_data["y_predict"]
                estimated_countries_data["new_pisa"] = estimated_countries_data["y_predict"].fillna(estimated_countries_data["PISA-SCORE"])
                estimated_countries_data = estimated_countries_data.sort_values(["YEAR"])

                years = list(estimated_countries_data["YEAR"].unique())

                for country in  estimated_countries:

                    tmp = estimated_countries_data.loc[estimated_countries_data["COUNTRY"]== country].copy()

                    plt.figure(figsize=(20,10))


                    s = sns.lineplot(data = tmp, x="YEAR", y="new_pisa", color='r', linestyle='--')
                    p = sns.lineplot(data = tmp, x="YEAR", y="PISA-SCORE", marker='o', label="True PISA Score")
                    plt.scatter(tmp["YEAR"], tmp["y_predict"], color='r', marker='x', label="Predicted Values", s=100)
                    plt.xticks(rotation=45)
                    plt.ylabel("Score")
                    plt.legend()
                    plt.title(f" Score and Predictions for {country}")
                    p.get_xaxis().get_major_formatter().set_useOffset(False)
                    plt.savefig(path_results_modelrun_plots / f"{pisa_type}-{iteration}_Plot_{country}_{model_run}_{h_param}.png")
                    plt.close()

                ###all countries in one plot###
                plt.figure(iteration+3, figsize=(20,40))
                for country in  estimated_countries:

                    tmp = estimated_countries_data.loc[estimated_countries_data["COUNTRY"] == country].copy()

                    s = sns.lineplot(data = tmp, x="YEAR", y="new_pisa", color='r', linestyle='--')
                    p = sns.lineplot(data = tmp, x="YEAR", y="PISA-SCORE", marker='o', label=f"True PISA Score {country}")
                    plt.scatter(tmp["YEAR"], tmp["y_predict"], color='r', marker='x', label=f"Predicted Values {country}", s=100)
                    p.get_xaxis().get_major_formatter().set_useOffset(False)

                plt.xticks(rotation=45)
                plt.title(f" Score and Predictions")
                plt.ylabel("Score")
                plt.legend(loc=4)
                plt.savefig(path_results_modelrun_plots/ f"{pisa_type}-{iteration}_Plot_all_countries_{model_run}_{h_param}.png")
                plt.close()

    # create overview of modelrun
    index = list(itertools.product(Hyper_Parameter[model_run], Iterations))
    results_comparative_modelrun = pd.DataFrame(columns=[f"{model_run}", "Percentage non-NA", "Total Features", "Features Tree", "RMSE",
                                                             "R2", "Percentage_NA_mean"], index=pd.MultiIndex.from_tuples(index))
    for h_param in Hyper_Parameter[model_run]:
        for iteration in Iterations:
            ind = (h_param, iteration)


            total_features = pd.read_csv(Path(path_input/input_data)/ f"{pisa_type}-{iteration}.csv")


            features_df = pd.read_csv(path_results_modelrun/ f"{pisa_type}-{iteration}_Feature_Analysis_{model_run}_{h_param}.csv")
            results_df = pd.read_csv(path_results_modelrun / f"{pisa_type}-{iteration}_Results_{model_run}_{h_param}.csv")
            results_df = results_df.rename(columns={"Unnamed: 0": "Test_data"})
            results_df = results_df.set_index("Test_data")

            results_comparative_modelrun.loc[ind, f"{model_run}"] = h_param
            results_comparative_modelrun.loc[ind, "Percentage non-NA"] = iteration
            results_comparative_modelrun.loc[ind, "RMSE"] = results_df.loc[model_run, "RMSE"]
            results_comparative_modelrun.loc[ind, "R2"] = results_df.loc[model_run, "R^2"]
            results_comparative_modelrun.loc[ind, "Depth"] = results_df.loc[model_run, "Depth"]
            results_comparative_modelrun.loc[ind, "Features Tree"] = len(features_df)
            results_comparative_modelrun.loc[ind, "Total Features"] = len(total_features.columns)-3
            results_comparative_modelrun.loc[ind, "Percentage_NA_mean"] = features_df["Annual Mean [%]"].mean()
            results_comparative_modelrun.loc[iteration, "Mean_Approximated_Values[%]"] = features_df["NA-Values [%]"].mean()

    results_comparative_modelrun.to_csv(path_results_modelrun / f"{pisa_type}_Overview_{model_run}.csv")

