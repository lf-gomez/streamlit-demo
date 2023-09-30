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
    return pd.read_csv("./files/data.csv").set_index("Date")


def optimize_portfolio(returns: pd.DataFrame, stocks: List[str]):
    rf = 0.1125 / 252

    returns = returns.loc[:, stocks]

    mean = returns.mean()
    cov = returns.cov()

    n = len(returns.columns)

    w0 = np.ones((n,)) / n
    bnds = ((1 / (2*n), 1),) * n

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


# APP CODE

