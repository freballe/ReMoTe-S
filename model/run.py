import os
import numpy as np
import pandas as pd
import parameters
from model import HouseholdDecisionModel

years = 30
num_iters = 100  # number of times we execute the full model to compute confidence intervals
scenarios = ["baseline", "A3", "B7", "A7", "B3", "A1", "B1", "C", "D"]

for scenario in scenarios:
    os.makedirs("../results/" + scenario, exist_ok=True)
    print("------------------------- STARTING SCENARIO "+ scenario +" ------------------------------------")
    results_df = pd.DataFrame()
    for iters in range(num_iters):
        print("------------------------- STARTING MODEL "+ str(iters) +" execution ------------------------------------")
        np.random.seed(iters)
        model = HouseholdDecisionModel(seed=42, scenario=scenario)
        for month in range(12*years):
            print("Step", month)
            model.step()
        current_results = model.datacollector.get_model_vars_dataframe()
        current_results["model_id"] = iters
        results_df = pd.concat([results_df, current_results], sort=False)    # concatenate dataframes
        model.dump_data_to_csv(r'../results/' + scenario + '/model_data_'+ str(iters) +'.csv', r'../results/' + scenario + '/households_data_'+ str(iters) +'.csv', r'../results/' + scenario + '/REAS_data_all.csv', r'../results/' + scenario + '/REAS_data_move.csv', iters)
    print("Dumping results to csv..")
    results_df.to_csv ("../results/" + scenario + "/full_results.csv", header=True)