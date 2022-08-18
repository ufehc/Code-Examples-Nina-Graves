#%%
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import pandas as pd
import numpy as np
from Global_configuartions import FOLDERNAME_MODELLING_DATA

Iterations = np.arange(50, 105, 5)
Input_Data = ["aggregated-years", "pisa-years"]
Pisa_Type = ["READ", "MATH", "SCIENCE"]
path_0 = Path(FOLDERNAME_MODELLING_DATA)
path_results = Path(path_0/"Results")
path_tree = Path(path_0/"Trees")
path_input = Path(path_0/"Input Data")
model_run_name = "Optimal_Single_Tree_RS42_2"
randomSearch = False
gridSearch = True
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
        path_tree_modelrun = Path(path_tree/model_run)
        path_tree_modelrun.mkdir(exist_ok=True)

        pbar3 = tqdm(Iterations)
        for iteration in pbar3:
            pbar.set_description(f"{input_data}: {pisa_type}-{iteration}")

            # load input data and meta data
            data_for_model = pd.read_csv(Path(path_input/ input_data)/f"{pisa_type}-{iteration}.csv")
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


            # Params for Hyperparameter-optimisation
            params = {"max_depth": [4, 5, 6, 7, 8, 9, 10],
                      "min_samples_split": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                      "min_samples_leaf": [2, 3, 4, 5, 6, 7, 8, 9, 10]}

            # Get Hyperparams with Grid Search
            grid_tree = DecisionTreeRegressor(random_state=42)
            grid_search = GridSearchCV(grid_tree, params, n_jobs=-1, cv=5)
            grid_search.fit(X_train_final, y_train_final)
            params_opt = grid_search.best_params_


            # create and train model
            model_tree = DecisionTreeRegressor(min_samples_split=params_opt['min_samples_split'],
                                               max_depth=params_opt['max_depth'],
                                               min_samples_leaf=params_opt['min_samples_leaf'], random_state=42)
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
                if model_tree.tree_.feature[node_id] == -2:  # id of all Leaf nodes
                    features.loc[node_id, "Feature"] = "Leaf"
                    features.loc[node_id, "Threshold"] = np.nan
                else:
                    features.loc[node_id, "Feature"] = feature_columns[model_tree.tree_.feature[node_id]]
                    features.loc[node_id, "Threshold"] = model_tree.tree_.threshold[node_id]
                features.loc[node_id, "Samples"] = model_tree.tree_.n_node_samples[node_id]
                features.loc[node_id, "Value"] = model_tree.tree_.value[node_id][0]
            features_merged = features.merge(column_information, how="left", left_on="Feature", right_index=True)
            features_merged = features_merged.set_index("Feature")
            features_merged.to_csv(path_tree_modelrun / f"{pisa_type}-{iteration}_{model_run}.csv")

            # create Robustness Information
            # Identify how often the different features appear in relation to the training samples in the tree
            # sort according to samples - importance of that feature pursumed by the total number of samples at the feature
            robust_check = features.drop(columns=["Threshold", "Value"], axis=1)
            robust_check = robust_check.groupby("Feature").sum()  # sum up the number of samples at nodes of the feature
            robust_check = robust_check.sort_values("Samples", ascending=False)
            robust_check = robust_check.drop("Leaf")  # drop leaf node - not relevant to feature analysis
            robust_check = robust_check.merge(column_information, how="left", left_on="Feature", right_index=True)

            # calculate how affected the samples passing through the node are by the approximation of  na values by the annual mean
            robust_check["Robustness - Samples affected by NA [Samples]"] = robust_check["Samples"] * robust_check["NA-Values [%]"] / 100
            robust_check["Robustness - Samples affected by Annual Mean [Samples]"] = robust_check["Samples"] * robust_check["Annual Mean [%]"] / 100
            robust_check.to_csv(path_results_modelrun/ f"{pisa_type}-{iteration}_Feature_Analysis_{model_run}.csv")

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
            results_with_total_overview.loc[model_run, "RandomSearch"] = randomSearch
            results_with_total_overview.loc[model_run, "GridSearch"] = gridSearch
            results_with_total_overview.loc[model_run, "Percentage Test-Data"] = percentage_test_data
            results_with_total_overview.loc[model_run, "Optimal_Depth"] = model_tree.get_depth()
            results_with_total_overview.loc[model_run, "Optimal_min_leafs"] = params_opt['min_samples_leaf']
            results_with_total_overview.loc[model_run, "Optimal_min_splits"] = params_opt['min_samples_split']
            results_with_total_overview.loc[model_run, "R2 Training Data"] = model_tree.score(X_train_final, y_train_final)
            results_with_total_overview.to_csv(path_results_modelrun / f"{pisa_type}-{iteration}_Results_{model_run}.csv")

            ######### Plots ##############
            path_results_modelrun_plots = Path(path_results_modelrun / f"{pisa_type}-{iteration}_Plots")
            path_results_modelrun_plots.mkdir(exist_ok=True)
            ###Prediction, True Scores ###
            test_plot = X_test_final.copy()
            test_plot = test_plot.merge(data_for_model.loc[X_test_final.index, "COUNTRY"], how="left", left_index=True,
                                        right_index=True)
            test_plot = test_plot.merge(y_test_final, how="left", left_index=True, right_index=True)
            test_plot = test_plot.sort_values(["YEAR"])
            plt.figure(iteration, figsize=(20, 10))
            sc_plot_predict = sns.scatterplot(test_plot.index, test_plot["PISA-SCORE"],
                                              hue=test_plot["COUNTRY"])
            plt.xticks(rotation=45)
            plt.plot(y_predict, c="r", label="Prediction")
            plt.legend(loc=1)
            plt.title(f"{pisa_type}-{input_data}_{iteration}: True Scores vs. Predictions")
            plt.savefig(path_results_modelrun_plots / f"RESULT_{pisa_type}-{iteration}.png")
            plt.close()

            ###Export trees both with and without feature names####
            export_graphviz(model_tree, out_file=f"{pisa_type}-{iteration}_Tree_{model_run}_with_names.dot", feature_names=feature_columns)
            export_graphviz(model_tree, out_file=f"{pisa_type}-{iteration}_Tree_{model_run}.dot")

            ### Show abs. error per pisa score ###
            # see whether there is potential for improvement in specific areas of the pisa score
            plt.figure()
            plt.scatter(results["PISA-SCORE"], results["RMSE"])
            plt.xlabel("True PISA Score")
            plt.ylabel("RMSE")
            plt.title(f"{pisa_type}-{input_data}_{iteration}: RMSE per PISA")
            plt.savefig(path_results_modelrun_plots / f"{pisa_type}-{iteration}_RMSE-True_Score.png")
            plt.close()

            ### Average Error per year ###
            # identify systematic over/underprediction on yearly basis taking the frequency of the year's appearance into account
            plt.figure()
            tmp = results[["YEAR", "RMSE"]].groupby("YEAR").mean()
            plt.plot(tmp.index, tmp, marker="o", label="Average Error")
            plt.hist(results["YEAR"], bins=100, align="left", label="Frequency in Test-Data")
            plt.xticks(tmp.index)
            plt.legend()
            plt.title(f"{pisa_type}-{input_data}_{iteration}: Average RMSE per year")
            plt.savefig(path_results_modelrun_plots / f"{pisa_type}-{iteration}_RMSE_per_year.png")
            plt.close()

            ### Robustness-Check ###
            # display robustness of the best features - show how many na values they had and how many of the values were substituted by the annual mean
            plt.figure(figsize=(20, 20))
            if "Leaf" in robust_check.index:
                robust_check = robust_check.drop("Leaf")
            data = robust_check[:10].copy()
            feature_names = robust_check.index.to_list()[:10]
            feature_names = [f"{x}: {feature_names[x]}" for x in range(len(feature_names))]
            feature_names = "\n".join(feature_names)

            plt.plot(range(len(data)), data["Annual Mean"], marker="x", c="r", label="NAs approximated by annual mean")
            plt.plot(range(len(data)), data["NA-Values"], marker="x", c="b", label="NA-Values")
            plt.hist(data.index, bins=range(len(data)+1), color="slategray", align="left", rwidth=0.5,
                     weights=data["Samples"], label="# Samples")
            plt.legend()
            plt.xticks(range(len(data.index)), range(len(data)));
            plt.xlabel(feature_names, fontsize="x-small", ma="left")
            plt.gcf().subplots_adjust(bottom=0.5)
            plt.title(f"{pisa_type}-{input_data}_{iteration}: Substituted NAs")
            plt.savefig(path_results_modelrun_plots / f"{pisa_type}-{iteration}_Robustness_Check_{model_run}.png")
            plt.close()

            ###Features###
            # get overview of features with most samples in tree
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
            # plot predictions for countries to sidentify systematic over-underprediction on country level
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
                plt.savefig(path_results_modelrun_plots / f"{pisa_type}-{iteration}_Plot_{country}.png")
                plt.close()

            ###all countries in one plot###
            ###all countries in one plot - maybe an annual systematic error might be spottet on country basis###
            plt.figure(iteration+3, figsize=(20,40))
            for country in  estimated_countries:

                tmp = estimated_countries_data.loc[estimated_countries_data["COUNTRY"]== country].copy()

                s = sns.lineplot(data = tmp, x="YEAR", y="new_pisa", color='r', linestyle='--')
                p = sns.lineplot(data = tmp, x="YEAR", y="PISA-SCORE", marker='o', label=f"True PISA Score {country}")
                plt.scatter(tmp["YEAR"], tmp["y_predict"], color='r', marker='x', label=f"Predicted Values {country}", s=100)
                p.get_xaxis().get_major_formatter().set_useOffset(False)

            plt.xticks(rotation=45)
            plt.title(f" Score and Predictions")
            plt.ylabel("Score")
            plt.legend(loc=4)
            plt.savefig(path_results_modelrun_plots/ f"{pisa_type}-{iteration}_Plot_all_countries.png")
            plt.close()

        # create overview of modelrun
        results_comparative_modelrun = pd.DataFrame(columns=["Percentage non-NA", "Total Features", "Features Tree", "RMSE",
                                                             "R2", "Percentage_NA_mean"], index=Iterations)
        for iteration in Iterations:

            total_features = pd.read_csv(Path(path_input/input_data)/ f"{pisa_type}-{iteration}.csv")


            features_df = pd.read_csv(path_results_modelrun / f"{pisa_type}-{iteration}_Feature_Analysis_{model_run}.csv")
            features_df = features_df.set_index("Feature")
            if "Leaf" in features_df.index:
                features_df = features_df.drop("Leaf")
            results_df = pd.read_csv(path_results_modelrun / f"{pisa_type}-{iteration}_Results_{model_run}.csv")
            results_df = results_df.rename(columns={"Unnamed: 0": "Test_data"})
            results_df = results_df.set_index("Test_data")

            top_features = features_df.index.to_list()[:10]

            results_comparative_modelrun.loc[iteration, "Percentage non-NA"] = model_run
            results_comparative_modelrun.loc[iteration, "RMSE"] = results_df.loc[model_run, "RMSE"]
            results_comparative_modelrun.loc[iteration, "R2"] = results_df.loc[model_run, "R^2"]
            results_comparative_modelrun.loc[iteration, "max_Depth"] = results_df.loc[model_run, "Optimal_Depth"]
            results_comparative_modelrun.loc[iteration, "min_leaf"] = results_df.loc[model_run, "Optimal_min_leafs"]
            results_comparative_modelrun.loc[iteration, "min_split"] = results_df.loc[model_run, "Optimal_min_splits"]
            results_comparative_modelrun.loc[iteration, "Features Tree"] = len(features_df)
            results_comparative_modelrun.loc[iteration, "R2 Training Data"] = results_df.loc[model_run, "R2 Training Data"]
            results_comparative_modelrun.loc[iteration, "Total Features"] = len(total_features.columns)-3
            results_comparative_modelrun.loc[iteration, "Percentage_NA_mean"] = features_df["Annual Mean [%]"].mean()
            results_comparative_modelrun.loc[iteration, "Mean_Approximated_Values[%]"] = features_df["NA-Values [%]"].mean()
            for number in range(0, len(top_features)):
                results_comparative_modelrun.loc[iteration, f" Top {number + 1} feature"] = top_features[number]

        results_comparative_modelrun.to_csv(path_results_modelrun / f"{pisa_type}_Overview_{model_run}.csv")