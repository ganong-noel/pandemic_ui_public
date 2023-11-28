
load release_paths.mat

table_mpc_for_paper = readtable(fullfile(release_path_paper,'table_mpc_for_paper.csv'));

% Round everything to two decimal places and get in correct format with trailing zeroes
table_mpc_for_paper.MPC_model = round(table_mpc_for_paper.MPC_model, 2);
table_mpc_for_paper.MPC_model = num2str(table_mpc_for_paper.MPC_model, '%.2f    ');

% Drop first row (to save space)
table_mpc_for_paper = table_mpc_for_paper(:, [2:end]);

% Abbreviate text and add Latex escape character "\" for dollar sign
table_mpc_for_paper.Transfer = "\" + string(table_mpc_for_paper.Transfer);
table_mpc_for_paper.Transfer(table_mpc_for_paper.Transfer=="\$2400 Persistent+Regular Benefits") = "\$2400 Persistent+Reg UI";

table_mpc_for_paper.Who_Gets_Transfer = string(table_mpc_for_paper.Who_Gets_Transfer);
table_mpc_for_paper.Who_Gets_Transfer(table_mpc_for_paper.Who_Gets_Transfer=="Unemployed w/ No Benefits") = "Unemp no UI";
table_mpc_for_paper.Who_Gets_Transfer(table_mpc_for_paper.Who_Gets_Transfer=="Unemployed w/ Regular Benefits") = "Unemp w/ reg UI";

table_mpc_for_paper.Discount_Factor_Calibration = string(table_mpc_for_paper.Discount_Factor_Calibration);
table_mpc_for_paper.Discount_Factor_Calibration = extractBetween(table_mpc_for_paper.Discount_Factor_Calibration, "Target ", " MPC");
table_mpc_for_paper.Discount_Factor_Calibration(table_mpc_for_paper.Discount_Factor_Calibration=="Waiting Design") = "Waiting";
table_mpc_for_paper.Discount_Factor_Calibration(table_mpc_for_paper.Discount_Factor_Calibration=="$500") = "\$500";

table_mpc_for_paper.Economic_Environment = string(table_mpc_for_paper.Economic_Environment);
table_mpc_for_paper.Economic_Environment(table_mpc_for_paper.Economic_Environment=="Normal Times") = "Normal";

table_mpc_for_paper.MPC_data = string(table_mpc_for_paper.MPC_data);
table_mpc_for_paper.MPC_data(table_mpc_for_paper.MPC_data=="-") = "";
table_mpc_for_paper.MPC_model = string(table_mpc_for_paper.MPC_model);
table_mpc_for_paper.MPC_horizon = string(table_mpc_for_paper.MPC_horizon);

% Add row number in first column
row_numbers = table(["(1)"; "(2)"; "(3)"; "(4)"; "(5)"; "(6)"; "(7)"; "(8)"], 'VariableNames', " ");
table_mpc_for_paper = [row_numbers table_mpc_for_paper];

% Save to Latex
table2latex_numbers_only(table_mpc_for_paper, char(strcat(release_path_paper, '/table_mpc_for_paper.tex')));
