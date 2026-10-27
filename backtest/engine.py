import pandas as pd

def simple_signal_backtest(
    df: pd.DataFrame,
    up_probability: pd.Series,
    threshold: float = 0.60,
    holding_days: int = 5,
    transaction_cost: float = 0.001,
):
    # Educational baseline only. A production engine needs
    # position sizing, slippage, liquidity, taxes, overlapping trades, etc.
    data = df.copy().reset_index(drop=True)
    data["signal"] = (up_probability >= threshold).astype(int)

    equity = [1_000_000.0]
    trades = []

    for i in range(len(data) - holding_days):
        capital = equity[-1]

        if data.loc[i, "signal"] == 1:
            # pyrefly: ignore [bad-argument-type]
            entry = float(data.loc[i, "close"])
            # pyrefly: ignore [bad-argument-type]
            exit_price = float(data.loc[i + holding_days, "close"])
            ret = exit_price / entry - 1 - transaction_cost
            capital *= (1 + ret)
            trades.append(ret)

        equity.append(capital)

    result = pd.DataFrame({"equity": equity})
    total_return = result["equity"].iloc[-1] / result["equity"].iloc[0] - 1
    return {
        "total_return": float(total_return),
        "final_equity": float(result["equity"].iloc[-1]),
        "trades": len(trades),
        "win_rate": float(sum(r > 0 for r in trades) / len(trades)) if trades else 0.0,
        "equity_curve": result,
    }
