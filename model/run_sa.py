import os
import parameters
import numpy as np
import pandas as pd
from model import HouseholdDecisionModel

# Note: the following code returns one csv for each "parameter_sa", containing
#  the results of "num_iters" instances of the model, at each step, for each combination of "parameter_sa" values

years = 30
parameters_sa = ["flatshare_size", "mean_rooms_baseline", "immigration_rate", "construction_rate_baseline", "construction_rate_A3", "construction_rate_B7", "construction_rate_A7", "construction_rate_B3"] # parameter to monitor for sensitivity analysis
num_iters = 100 # num of indipendent executions of the model

os.makedirs("../results/SA", exist_ok=True)
      
for parameter_sa in parameters_sa:
    results_df = pd.DataFrame()
    for iters in range(num_iters):
        print("////////  Starting iter " + str(iters) + " for parameter " + parameter_sa + "  ////////")
        if parameter_sa == "flatshare_size":
            for flat_size in range(1,11,1):
                np.random.seed(iters)
                model = HouseholdDecisionModel(seed=42, scenario="baseline", sensitivity_analysis=True, flat_size=flat_size)
                for month in range(12*years):
                    print("Step", month)
                    model.step()
                current_results = model.datacollector.get_model_vars_dataframe()
                current_results[parameter_sa] = flat_size
                current_results["model_id"] = iters
                results_df = pd.concat([results_df, current_results], sort=False)    # concatenate dataframes
        elif parameter_sa == "construction_rate_baseline":
            # scenario A construction rate
            for construct_rate in np.linspace(0.0, 0.05, num=11):
                np.random.seed(iters)
                model = HouseholdDecisionModel(seed=42, scenario="baseline", sensitivity_analysis=True, construction_rate=construct_rate)
                for month in range(12*years):
                    print("Step", month)
                    model.step()
                current_results = model.datacollector.get_model_vars_dataframe()
                current_results[parameter_sa] = construct_rate
                current_results["model_id"] = iters
                results_df = pd.concat([results_df, current_results], sort=False)    # concatenate
        elif parameter_sa == "construction_rate_A3":
            # scenario A construction rate
            for construct_rate in np.linspace(0.0, 0.05, num=11):
                np.random.seed(iters)
                model = HouseholdDecisionModel(seed=42, scenario="A3", sensitivity_analysis=True, construction_rate=construct_rate)
                for month in range(12*years):
                    print("Step", month)
                    model.step()
                current_results = model.datacollector.get_model_vars_dataframe()
                current_results[parameter_sa] = construct_rate
                current_results["model_id"] = iters
                results_df = pd.concat([results_df, current_results], sort=False)    # concatenate
        elif parameter_sa == "construction_rate_B7":    
            # scenario B construction rate
            for construct_rate in np.linspace(0.0, 0.05, num=11):
                np.random.seed(iters)
                model = HouseholdDecisionModel(seed=42, scenario="B7", sensitivity_analysis=True, construction_rate=construct_rate)
                for month in range(12*years):
                    print("Step", month)
                    model.step()
                current_results = model.datacollector.get_model_vars_dataframe()
                current_results[parameter_sa] = construct_rate
                current_results["model_id"] = iters
                results_df = pd.concat([results_df, current_results], sort=False)    # concatenate
        elif parameter_sa == "construction_rate_A7":
            # scenario A construction rate
            for construct_rate in np.linspace(0.0, 0.05, num=11):
                np.random.seed(iters)
                model = HouseholdDecisionModel(seed=42, scenario="A7", sensitivity_analysis=True, construction_rate=construct_rate)
                for month in range(12*years):
                    print("Step", month)
                    model.step()
                current_results = model.datacollector.get_model_vars_dataframe()
                current_results[parameter_sa] = construct_rate
                current_results["model_id"] = iters
                results_df = pd.concat([results_df, current_results], sort=False)    # concatenate
        elif parameter_sa == "construction_rate_B3":    
            # scenario B construction rate
            for construct_rate in np.linspace(0.0, 0.05, num=11):
                np.random.seed(iters)
                model = HouseholdDecisionModel(seed=42, scenario="B3", sensitivity_analysis=True, construction_rate=construct_rate)
                for month in range(12*years):
                    print("Step", month)
                    model.step()
                current_results = model.datacollector.get_model_vars_dataframe()
                current_results[parameter_sa] = construct_rate
                current_results["model_id"] = iters
                results_df = pd.concat([results_df, current_results], sort=False)    # concatenate
        elif parameter_sa == "immigration_rate":
            for immigration_rate in np.linspace(0.0, 0.1, num=11):
                np.random.seed(iters)
                model = HouseholdDecisionModel(seed=42, scenario="baseline", sensitivity_analysis=True, immigration_rate=immigration_rate)
                for month in range(12*years):
                    print("Step", month)
                    model.step()
                current_results = model.datacollector.get_model_vars_dataframe()
                current_results[parameter_sa] = immigration_rate
                current_results["model_id"] = iters
                results_df = pd.concat([results_df, current_results], sort=False)    # concatenate
        elif parameter_sa == "mean_rooms_baseline":
            for mean_rooms in np.linspace(1.5, 5.5, num=5):
                np.random.seed(iters)
                model = HouseholdDecisionModel(seed=42, scenario="baseline", sensitivity_analysis=True, mean_rooms=mean_rooms)
                for month in range(12*years):
                    print("Step", month)
                    model.step()
                current_results = model.datacollector.get_model_vars_dataframe()
                current_results[parameter_sa] = mean_rooms
                current_results["model_id"] = iters
                results_df = pd.concat([results_df, current_results], sort=False)    # concatenate
        else:
            raise ValueError("No SA with the given name.")

    print("Dumping results to csv..")
    results_df.to_csv ("../results/SA/full_results_" + parameter_sa + ".csv", header=True)