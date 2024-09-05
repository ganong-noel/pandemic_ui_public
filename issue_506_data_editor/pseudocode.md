## Pseudocode for Python Build in Chase

The following are the inputs for the R portion of the Chase analysis which are produced in the undisclosed Chase Python builds:

1.  `weekly_flows/{date}_weekly_flows_100pct.rds`
2.  `weekly_cp/{date}_weekly_inflows_by_cp_100pct.rds`
3.  `demog/{date}_monthly_demographics_100pct.rds`
4.  `eips/{date}_eips_list_100pct.rds`
5.  `eips/{date}_eip_rounds_list_100pct.rds`

### Setup: Customer List

For all of the files, our customer list consists of UI recipients who received UI in 2020, as well as a random sample of non-UI recipients. The primacy screen for the non-UI recipients is that we require them to receive \$12,000 of labor income in 2019.

### Flows and Demographics:

The following three files include information about cash flows and are grouped together:

-   `weekly_flows_100pct.rds` (Weekly Flows): This contains weekly information on all types of cash flows, inflow and outflow.
-   `weekly_inflows_by_cp_100pct.rds` (Weekly Inflows): Weekly information only on inflows.
-   `monthly_demographics_100pct.rds` (Demographics): Monthly information on all flows, along with comprehensive demographic information.

#### Weekly Flows

-   Chase Input file: Chase parquet datatable with information on customer transactions.
    -   Level of Aggregation: customer-by-day-by-flow
    -   Contents:
        -   This file takes the input Chase table and categorizes different inflow types (total inflows, UI inflows, labor income, transfers, tax refunds) by identifying the various transaction descriptions. Additionally, the transaction descriptions also help us identify the state in which the customer receives their UI cheque in. The data also identifies spending details for every customer and categorizes these outflows based on their type of spending category (durable, non-durable, essential, non-essential, debt payments) or mode of spending (debit card, credit card, cash, paper cheques).
-   Output file for analysis:
    -   Level of Aggregation: customer-by-week-by-flow
    -   Contents:
        -   We take the above input table, and aggregate it to a weekly level for analysis.

#### Weekly Inflows

-   Chase Input file: Chase parquet datatable with information on customer transactions.
    -   Level of Aggregation: customer-by-day-by-flow
    -   Contents:
        -   This is very similar to the weekly flows file, but includes information only on inflows, either labor or UI inflows. It takes the input Chase table and categorizes labor or UI inflows by identifying the various transaction descriptions.
-   Output file for analysis:
    -   Level of Aggregation: customer-by-week-by-inflow-by-counterparty
    -   Contents:
        -   We take the above input table, and aggregate it to a weekly level for analysis. We also include the counterparty depositing the inflow into the account: firm, if it is a labor inflow, or state governmental department, for UI inflows.

#### Demographics

-   Chase Input file: Chase parquet datatable with information on customer transactions, table with demographic data, and table with account balances.
    -   Level of Aggregation: customer-by-day-by-flow for customer transactions; by-customer for the demographics table; customer-by-month for account balance table.
    -   Contents:
        -   This takes the daily flows (input table for Weekly Flows above) and aggregates it to the monthly level. The demographics table and the account balance table, which contains information on checking account and total liquid balances, is then joined onto the monthly aggregated version of the transaction table.
-   Output file for analysis:
    -   Level of Aggregation: customer-by-month
    -   Contents:
        -   As described in the contents of the input table, it is aggregated up to the monthly level.

### EIP Specific:

The following two files are grouped together as they include information on EIP recipients:

-   `eips_list_100pct.rds` (EIP list): April 2020 EIP transactions information for just the one round.
-   `eip_rounds_list_100pct.rds` (EIP rounds): EIP transactions information for all 3 rounds.

#### EIP list and rounds

-   Chase Input file: Chase parquet datatable with information on customer transactions.
    -   Level of Aggregation: customer-by-day-by-flow for customer transactions
    -   Contents:
        -   This takes the transaction table and categorize transactions as EIPs (both direct deposit and paper cheque types) using transaction descriptions for UI recipients.
-   Output file for analysis:
    -   Level of Aggregation: customer-by-day-by-EIP_flow
    -   Contents:
        -   The EIP list output only contains deposit information (date and amount) on the first round. The EIP rounds output has deposit information (date and amount) all 3 rounds.
