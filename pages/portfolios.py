import numpy as np
import pandas as pd
import streamlit as st

from scipy.optimize import minimize
from typing import List

st.set_page_config(
    page_title="Portfolios",
    page_icon="💼",
    layout="wide",
)


@st.cache_data
def get_data() -> pd.DataFrame:
    # websocket
    # request rest api
    #
    return pd.read_csv("./files/data.csv").set_index("Date")


def optimize_portfolio(returns: pd.DataFrame, stocks: List[str]):
    rf = 0.1125 / 252

    returns = returns.loc[:, stocks]

    mean = returns.mean()
    cov = returns.cov()

    n = len(returns.columns)

    w0 = np.ones((n,)) / n
    bnds = ((1 / (2 * n), 1),) * n

    cons = ({
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1
            },)

    res = minimize(fun=sharpe_ratio,
                   x0=w0,
                   args=(mean, cov, rf),
                   bounds=bnds,
                   constraints=cons,
                   tol=1e-8)

    er = mean.T.dot(res.x)
    sigma = (res.x.T.dot(cov).dot(res.x)) ** 0.5

    weights_df = pd.DataFrame({"Stock": stocks, "Weight": res.x * 100})

    metrics = {"Sharpe": -res.fun, "ER": er, "Sigma": sigma}

    return weights_df, metrics


def sharpe_ratio(w: np.array, mean: np.array, cov_matrix: np.array, rf: float) -> float:
    er = mean.T.dot(w)
    sp = (w.T.dot(cov_matrix).dot(w)) ** 0.5
    sharpe = (er - rf) / sp
    return -sharpe


def add_stock(ticker: str):
    global selected_stocks

    if ticker in selected_stocks:
        selected_stocks.remove(ticker)
    else:
        selected_stocks.append(ticker)


selected_stocks = []

data = get_data()
data_rets = data.pct_change().dropna()

weights_df = pd.DataFrame({"Stock": [], "Weight": []})
metrics = {}

# APP CODE

with st.container():
    st.write("# Create your portfolio!")

    left_col, right_col = st.columns(2)

    with left_col:
        aapl_checkbox = st.checkbox("AAPL")
        amzn_checkbox = st.checkbox("AMZN")
        goog_checkbox = st.checkbox("GOOG")

    with right_col:
        nflx_checkbox = st.checkbox("NFLX")
        meta_checkbox = st.checkbox("META")
        tlsa_checkbox = st.checkbox("TSLA")

    if aapl_checkbox:
        if "AAPL" not in selected_stocks:
            selected_stocks.append("AAPL")
    else:
        if "AAPL" in selected_stocks:
            selected_stocks.remove("AAPL")

    if amzn_checkbox:
        if "AMZN" not in selected_stocks:
            selected_stocks.append("AMZN")
    else:
        if "AMZN" in selected_stocks:
            selected_stocks.remove("AMZN")

    if goog_checkbox:
        if "GOOG" not in selected_stocks:
            selected_stocks.append("GOOG")
    else:
        if "GOOG" in selected_stocks:
            selected_stocks.remove("GOOG")

    if nflx_checkbox:
        if "NFLX" not in selected_stocks:
            selected_stocks.append("NFLX")
    else:
        if "NFLX" in selected_stocks:
            selected_stocks.remove("NFLX")

    if meta_checkbox:
        if "META" not in selected_stocks:
            selected_stocks.append("META")
    else:
        if "META" in selected_stocks:
            selected_stocks.remove("META")

    if tlsa_checkbox:
        if "TSLA" not in selected_stocks:
            selected_stocks.append("TSLA")
    else:
        if "TSLA" in selected_stocks:
            selected_stocks.remove("TSLA")

    if len(selected_stocks) > 0:
        weights_df, metrics = optimize_portfolio(data_rets, selected_stocks)

        er_kpi, sigma_kpi, sharpe_kpi = st.columns(3)

        er_kpi.metric("Expected Returns", value=f"{metrics['ER'] * 252 * 100:.2f}%")
        sigma_kpi.metric("Portfolio Volatility", value=f"{metrics['Sigma'] * (252 ** 0.5) * 100:.2f}%")
        sharpe_kpi.metric("Sharpe", value=f"{metrics['Sharpe']:.4f}")

        st.bar_chart(data=weights_df, x="Stock", y="Weight")
        initial_price = data.iloc[0, :]

        cash = 1_000_000

        historical = pd.DataFrame()

        for stock in selected_stocks:
            weight = weights_df[weights_df["Stock"] == stock]["Weight"].values[0] / 100
            stock_price = initial_price[stock]
            num_shares = np.floor(cash * weight / stock_price)
            historical[stock] = data.loc[:, stock] * num_shares

        portfolio_value = historical.sum(axis=1)
        st.line_chart(portfolio_value)


