# Single agent, simple prompt - Sharpe 1.0026
python3 scripts/run_scenario_list.py --config config/debate/baselines/single_agent_slim.yaml --scenarios config/scenarios/top_divergence/top_scenarios_l.json --parallel 4

# 3 Agents propose only, vote, simple prompts - Sharpe 0.9906
 python3 scripts/run_scenario_list.py --config config/debate/baselines/vote_slim_no_macro.yaml --scenarios config/scenarios/top_divergence/top_scenarios_l.json

# 3 Agent propose only, vote, causal prompts - Sharpe 1.0540
python3 scripts/run_scenario_list.py --config config/debate/baselines/vote_causal_no_macro_causal_out.yaml --scenarios config/scenarios/top_divergence/top_scenarios_l.json --parallel 2

# 3 Agent propose only, vote, causal prompts with rearranged output - Sharpe 0.9806
python3 scripts/run_scenario_list.py --config config/debate/baselines/vote_causal_no_macro_causal_out_cot.yaml --scenarios config/scenarios/top_divergence/top_scenarios_l.json --parallel 2

# 3 Agent propose only, vote, enriched prompts - Sharpe 0.9880
python3 scripts/run_scenario_list.py --config config/debate/baselines/vote_no_macro_enriched.yaml --scenarios config/scenarios/top_divergence/top_scenarios_l.json --parallel 2




# 3 Agents propose only, judge, causal prompts - Sharpe 1.0341
python3 scripts/run_scenario_list.py --config config/debate/baselines/judge_causal_no_macro_causal_out.yaml --scenarios config/scenarios/top_divergence/top_scenarios_l.json --parallel 2

# 3 Agents 1 round debate, judge, causal prompts - 
python3 scripts/run_scenario_list.py --config config/debate/baselines/debate_1_round_no_macro_causal_out.yaml --scenarios config/scenarios/top_divergence/top_scenarios_l.json --parallel 2



# cs372research/results_tracking_debate_1_round_no_macro.csv

# cs372research/results_tracking_debate_1_round_no_macro_rr_roles.csv

# cs372research/results_tracking_debate_1_round_rr_single_devil.csv


# #### NOT RUN YET ####

# # 3 Agents propose only, judge, causal prompts, rearranged output
# cs372research/results_tracking_judge_no_macro_causal_out_cot.csv

# # 3 Agent  2 rounds, judge, enriched prompts
# cs372research/results_tracking_debate_enriched_no_macro.csv

# 3 Agents propose only, judge, causal prompts with rearranged output
# cs372research/results_tracking_judge_causal_no_macro_causal_out_cot.csv
