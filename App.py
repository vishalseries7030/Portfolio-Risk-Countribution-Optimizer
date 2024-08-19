import pandas as pd
import numpy as np
import warnings
from flask import Flask, request, jsonify
warnings.filterwarnings("ignore")
app = Flask(__name__)

# Function to load CSV files
def load_csv_files():
    print("lodingcsv")
    rets_file = "csvs\\ind49_m_vw_rets.csv" 
    mkt_caps_file = "csvs\\ind49_m_size.csv" 
    
    ind_rets = pd.read_csv(rets_file,header=0)
    print(ind_rets)
    ind_mkt_caps = pd.read_csv(mkt_caps_file,header=0)
    print(ind_mkt_caps)
    
    return ind_rets, ind_mkt_caps

# Load data from CSV files
ind_rets, ind_mkt_caps = load_csv_files()

def portfolio_risk_contrib_optimizer(target_risk, cov_matrix):
    from scipy.optimize import minimize

    n = len(target_risk)

    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights

    def risk_contribution(weights):
        portfolio_var = portfolio_variance(weights)
        marginal_contrib = cov_matrix @ weights
        return weights * marginal_contrib / np.sqrt(portfolio_var)

    def risk_contribution_diff(weights):
        actual_risk_contrib = risk_contribution(weights)
        return np.sum((actual_risk_contrib - target_risk)**2)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(risk_contribution_diff, np.array(target_risk), method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def portfolio_risk_contributions(weights, cov_matrix):
    portfolio_var = weights.T @ cov_matrix @ weights
    marginal_contrib = cov_matrix @ weights
    return weights * marginal_contrib / np.sqrt(portfolio_var)

def enc(weights):
    return 1 / np.sum(weights**2)

def encb(risk_contrib):
    return 1 / np.sum(risk_contrib**2)

def optimize_portfolio(target_risk, amount, companies):
    nind = len(companies)
    ind_rets_subset = ind_rets[companies]
    print("here")
    print(ind_rets_subset.head(5))
    ind_mkt_caps_subset = ind_mkt_caps[companies]
    mat_cov = ind_rets_subset.cov()

    # Portfolio optimization based on target risk contributions
    weights = portfolio_risk_contrib_optimizer(target_risk, mat_cov)
    p_risk_contribs = portfolio_risk_contributions(weights, mat_cov)
    ENC = enc(weights)
    ENCB = encb(p_risk_contribs)
    
    # Calculate the amount to be invested in each company
    investment = weights * amount

    return {
        "weights": dict(zip(companies, weights)),
        "risk_contributions": dict(zip(companies, p_risk_contribs)),
        "ENC": ENC,
        "ENCB": ENCB,
        "investment": dict(zip(companies, investment))
    }

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.get_json()
    risk_percentage = data.get('risk_percentage')
    amount = data.get('amount')

    # Example list of industries (using the same industries as in the original code)
    companies = ['Beer', 'Hlth', 'Fin', 'Rtail', 'Whlsl']

    # Normalize risk percentage to sum to 1
    total_risk = sum(risk_percentage)
    target_risk = pd.Series([r / total_risk for r in risk_percentage], index=companies)
    
    # Optimize portfolio
    result = optimize_portfolio(target_risk, amount, companies)
   
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)


