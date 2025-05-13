import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
import logging

def calculate_time_metrics(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculates the total elapsed time in hours and the number of data periods per year.

    Args:
        df (pd.DataFrame): A DataFrame indexed by datetime.

    Returns:
        Tuple[float, float]: A tuple containing:
            - Total hours between the first and last timestamp.
            - Estimated number of periods per year based on sampling frequency.

    Example:
        >>> df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=24, freq="H"))
        >>> calculate_time_metrics(df)
        (23.0, 8760.0)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    if df.empty:
        return 0.0, 1.0
    total_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
    periods_per_year = (len(df) / total_hours) * 365 * 24 if total_hours > 0 else 1
    return total_hours, periods_per_year

def calculate_next_direction(df: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
    """
    Determines the direction of the next price movement based on log returns.

    Args:
        df (pd.DataFrame): A DataFrame containing a 'close' price column.
        threshold (float): The minimum absolute log return to classify movement.

    Returns:
        pd.Series: Series with values 1 (up), -1 (down), or 0 (neutral).

    Example:
        >>> df = pd.DataFrame({'close': [100, 101, 100.5, 100.4]})
        >>> calculate_next_direction(df)
        0    1
        1   -1
        2    0
        3    0
        dtype: int64
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if 'close' not in df.columns:
        raise KeyError("'close' column is required in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df['close']):
        raise TypeError("'close' column must be numeric.")
    if df.empty:
        return pd.Series(index=df.index, dtype='int64')
    next_direction = pd.Series(index=df.index, dtype='int64')
    next_log_return = np.log(df['close'].shift(-1) / df['close'])
    next_direction[next_log_return > threshold] = 1
    next_direction[next_log_return < -threshold] = -1
    next_direction[(next_log_return >= -threshold) & (next_log_return <= threshold)] = 0
    return next_direction.fillna(0).astype('int64')

def calculate_time_metrics(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculates the total elapsed time in hours and the number of data periods per year.

    Args:
        df (pd.DataFrame): A DataFrame indexed by datetime.

    Returns:
        Tuple[float, float]: A tuple containing:
            - Total hours between the first and last timestamp.
            - Estimated number of periods per year based on sampling frequency.

    Example:
        >>> df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=24, freq="H"))
        >>> calculate_time_metrics(df)
        (23.0, 8760.0)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    if df.empty:
        return 0.0, 1.0
    total_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
    periods_per_year = (len(df) / total_hours) * 365 * 24 if total_hours > 0 else 1
    return total_hours, periods_per_year

def calculate_next_direction(df: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
    """
    Determines the direction of the next price movement based on log returns.

    Args:
        df (pd.DataFrame): A DataFrame containing a 'close' price column.
        threshold (float): The minimum absolute log return to classify movement.

    Returns:
        pd.Series: Series with values 1 (up), -1 (down), or 0 (neutral).

    Example:
        >>> df = pd.DataFrame({'close': [100, 101, 100.5, 100.4]})
        >>> calculate_next_direction(df)
        0    1
        1   -1
        2    0
        3    0
        dtype: int64
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if 'close' not in df.columns:
        raise KeyError("'close' column is required in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df['close']):
        raise TypeError("'close' column must be numeric.")
    if df.empty:
        return pd.Series(index=df.index, dtype='int64')
    next_direction = pd.Series(index=df.index, dtype='int64')
    next_log_return = np.log(df['close'].shift(-1) / df['close'])
    next_direction[next_log_return > threshold] = 1
    next_direction[next_log_return < -threshold] = -1
    next_direction[(next_log_return >= -threshold) & (next_log_return <= threshold)] = 0
    return next_direction.fillna(0).astype('int64')

def calculate_buy_hold_annualized(buy_hold_return: Union[float, int], periods_per_year: Union[float, int] or 365, periods: Union[float, int]) -> float:
    """
    Converts a buy-and-hold return to an annualized return with robust error handling.

    Args:
        buy_hold_return (float | int): Total return from buy-and-hold (decimal, e.g., 0.25 for 25%).
        periods_per_year (float | int): Number of periods per year (e.g., 365 for daily, 252 for trading days).
        periods (float | int): Total number of periods in the sample.

    Returns:
        float: Annualized return (decimal), or -1 for complete loss, 0.0 for invalid inputs.

    Raises:
        TypeError: If inputs are not numeric.
        ValueError: If periods_per_year or periods are non-positive.

    Example:
        >>> calculate_buy_hold_annualized(0.2, 365, 366)
        0.1990429812746988
        >>> calculate_buy_hold_annualized(0.2, 8760, 876)
        0.7474220831252856
    """
    # Type checking
    if not all(isinstance(x, (float, int)) for x in [buy_hold_return, periods_per_year, periods]):
        logging.error(f"Invalid input types: buy_hold_return={type(buy_hold_return)}, "
                      f"periods_per_year={type(periods_per_year)}, periods={type(periods)}")
        raise TypeError("All inputs must be numeric (float or int).")

    # Value checking
    if periods_per_year <= 0:
        logging.error(f"periods_per_year must be positive, got {periods_per_year}")
        raise ValueError("periods_per_year must be positive.")
    if periods <= 0:
        logging.error(f"periods must be positive, got {periods}")
        raise ValueError("periods must be positive.")

    # Handle complete loss
    if buy_hold_return <= -1:
        logging.warning(f"buy_hold_return {buy_hold_return} indicates complete or greater loss, returning -1")
        return -1.0

    try:
        # Calculate annualized return
        exponent = periods_per_year / periods
        annualized_return = (1 + buy_hold_return) ** exponent - 1

        # Check for numerical issues
        if np.isnan(annualized_return) or np.isinf(annualized_return):
            logging.warning(f"Computed annualized return is invalid (NaN or inf) for buy_hold_return={buy_hold_return}, "
                            f"periods_per_year={periods_per_year}, periods={periods}")
            return 0.0

        logging.debug(f"Calculated annualized return: {annualized_return:.6f} from buy_hold_return={buy_hold_return:.6f}, "
                      f"periods_per_year={periods_per_year}, periods={periods}")
        return annualized_return

    except Exception as e:
        logging.error(f"Error computing annualized return: {e}, inputs: buy_hold_return={buy_hold_return}, "
                      f"periods_per_year={periods_per_year}, periods={periods}")
        return 0.0

def calculate_buy_hold_profit(df: pd.DataFrame, initial_cash: int) -> float:
    """
    Calculate the return for buy-and-hold strategy.

    Args:
        df (pd.DataFrame): DataFrame with 'close' prices.

    Returns:
        float: Buy-and-hold return (as a decimal, e.g., 0.25 for 25%), or 0.0 if not possible.
    """
    if df.empty or 'close' not in df.columns or df['close'].isna().all():
        logging.error("Invalid DataFrame for buy-and-hold")
        return 0.0

    # Get valid close prices
    valid_prices = df['close'].dropna()
    valid_prices = valid_prices[valid_prices > 0]
    if len(valid_prices) < 2:
        logging.warning("Insufficient valid price data for buy-and-hold")
        return 0.0

    # Use first and last valid prices
    initial_price = valid_prices.iloc[0]
    final_price = valid_prices.iloc[-1]

    # Calculate return
    bh_return = (final_price - initial_price) / initial_price if initial_price != 0 else 0.0
    bh_profit = bh_return * initial_cash
    bh_total_profit = bh_profit + initial_cash

    return bh_return, bh_profit, bh_total_profit

def calculate_portfolio_value(profit_series: pd.Series, initial_cash: float) -> pd.Series:
    """
    Calculates the portfolio value over time based on profits and initial capital.

    Args:
        profit_series (pd.Series): Series of cumulative profits.
        initial_cash (float): The starting capital.

    Returns:
        pd.Series: Portfolio value over time.

    Example:
        >>> profit_series = pd.Series([0, 10, 30, 50])
        >>> calculate_portfolio_value(profit_series, 100)
        0    100.0
        1    110.0
        2    130.0
        3    150.0
        dtype: float64
    """
    if not isinstance(profit_series, pd.Series):
        raise TypeError("profit_series must be a pandas Series.")
    if not isinstance(initial_cash, (int, float)):
        raise TypeError("initial_cash must be a number.")
    if profit_series.empty:
        return pd.Series([initial_cash], dtype='float64')
    return (profit_series + initial_cash).astype('float64')

def calculate_returns(portfolio_value: pd.Series, initial_cash: float = 10000.0) -> pd.Series:
    """
    Calculates returns from portfolio values, propagating trade returns until the next trade.

    Args:
        portfolio_value (pd.Series): Series of portfolio values.
        initial_cash (float): Initial portfolio value for reference. Defaults to 10000.0.

    Returns:
        pd.Series: Series of returns where each trade's return persists until the next trade.

    Example:
        >>> portfolio_value = pd.Series([10000, 10000, 4028.20, 4028.20, 4589.50],
        ...                            index=['2024-03-16', '2024-03-17', '2024-03-18', '2024-03-19', '2024-03-20'])
        >>> calculate_returns(portfolio_value)
        2024-03-16    0.000000
        2024-03-17    0.000000
        2024-03-18   -0.597180
        2024-03-19   -0.597180
        2024-03-20   -0.541050
        Freq: D, dtype: float64
    """
    if not isinstance(portfolio_value, pd.Series):
        raise TypeError("portfolio_value must be a pandas Series.")
    if len(portfolio_value) <= 1:
        return pd.Series(0.0, index=portfolio_value.index)
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")

    # Compute profit/loss relative to initial_cash
    profit = portfolio_value - initial_cash

    # Calculate returns as profit / initial_cash
    returns = profit / initial_cash

    # Only keep returns where portfolio_value changes
    delta = portfolio_value.diff().fillna(0)
    returns = returns.where(delta != 0)

    # Forward-fill non-zero returns to propagate trade effects
    returns = returns.ffill()

    # Fill initial NaN (before first trade) with 0
    returns = returns.fillna(0)

    # Ensure index alignment
    returns = returns.reindex(portfolio_value.index, fill_value=returns.iloc[-1])

    # Replace inf/-inf with 0 (handles edge cases)
    returns = returns.replace([np.inf, -np.inf], 0)

    return returns

def calculate_annualized_return(portfolio_value: pd.Series, initial_cash: float, periods_per_year: float, periods: int) -> float:
    """
    Computes annualized return from portfolio values.

    Args:
        portfolio_value (pd.Series): Series of portfolio values.
        initial_cash (float): Starting capital.
        periods_per_year (float): Number of data periods in one year.
        periods (int): Total number of data periods.

    Returns:
        float: Annualized return.

    Example:
        >>> pv = pd.Series([100, 120])
        >>> calculate_annualized_return(pv, 100, 8760, 876)
        0.7474220831252856
    """
    if not isinstance(portfolio_value, pd.Series):
        raise TypeError("portfolio_value must be a pandas Series.")
    if portfolio_value.empty or periods <= 0:
        return 0.0
    return ((portfolio_value.iloc[-1] / initial_cash) ** (periods_per_year / periods)) - 1 if portfolio_value.iloc[-1] > 0 else -1

def calculate_excess_return_over_bh(annualized_return: float, buy_hold_annualized: float) -> float:
    """
    Calculates strategy's annualized return in excess of buy-and-hold.

    Args:
        annualized_return (float): Annualized strategy return.
        buy_hold_annualized (float): Annualized return of buy-and-hold.

    Returns:
        float: Excess return.

    Example:
        >>> calculate_excess_return_over_bh(0.15, 0.10)
        0.05
    """
    return annualized_return - buy_hold_annualized

def calculate_std_return(returns: pd.Series) -> float:
    """
    Computes standard deviation of periodic returns.

    Args:
        returns (pd.Series): Series of returns.

    Returns:
        float: Standard deviation, or 0 if undefined.

    Example:
        >>> calculate_std_return(pd.Series([0.01, -0.02, 0.03]))
        0.025495097567963924
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    std = returns.std()
    return 0.0 if pd.isna(std) or std == 0 else std

def calculate_downside_deviation(returns: pd.Series) -> float:
    """
    Calculates the downside deviation (volatility of negative returns).

    Args:
        returns (pd.Series): Series of returns.

    Returns:
        float: Downside deviation, or a small value if insufficient data.

    Example:
        >>> calculate_downside_deviation(pd.Series([0.01, -0.02, 0.03, -0.01]))
        0.0070710678118654745
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    downside_returns = returns[returns < 0]
    return downside_returns.std() if len(downside_returns) > 1 else 1e-10

def calculate_sharpe_ratio(annualized_return: float, risk_free_rate: float, std_return: float, periods_per_year: float = 8760.0) -> float:
    """
    Calculates the Sharpe ratio of a strategy.

    Args:
        annualized_return (float): Strategy's annualized return.
        risk_free_rate (float): Risk-free rate of return.
        std_return (float): Standard deviation of returns.
        periods_per_year (float): Number of periods per year (default is hourly).

    Returns:
        float: Sharpe ratio, or 0 if std is 0.

    Example:
        >>> calculate_sharpe_ratio(0.15, 0.02, 0.01, 8760)
        4.216370213555072
    """
    if std_return < 0:
        std_return = 0.0
    if annualized_return > 500.0:
        annualized_return = 500.0
    annualized_std = std_return * np.sqrt(periods_per_year)
    excess_return = annualized_return - risk_free_rate
    return excess_return / annualized_std if annualized_std > 0 else 0.0

def calculate_sortino_ratio(annualized_return: float, risk_free_rate: float, downside_deviation: float, periods_per_year: int) -> float:
    """
    Calculates the Sortino ratio, adjusting for downside risk.

    Args:
        annualized_return (float): Strategy's annualized return.
        risk_free_rate (float): Risk-free rate of return.
        downside_deviation (float): Downside deviation of returns.
        periods_per_year (int): Number of periods per year.

    Returns:
        float: Sortino ratio, or 0 if downside deviation is too small.

    Example:
        >>> calculate_sortino_ratio(0.15, 0.02, 0.01, 8760)
        13.833788439141695
    """
    if downside_deviation < 0:
        downside_deviation = 0.0
    if annualized_return > 10.0:
        annualized_return = 10.0
    annualized_downside = downside_deviation * np.sqrt(periods_per_year)
    excess_return = annualized_return - risk_free_rate
    return excess_return / annualized_downside if annualized_downside > 1e-9 else 0.0

def calculate_max_drawdown(portfolio_value: pd.Series) -> float:
    """
    Calculates the maximum drawdown (peak-to-trough loss).

    Args:
        portfolio_value (pd.Series): Series of portfolio values.

    Returns:
        float: Max drawdown as a decimal.

    Example:
        >>> pv = pd.Series([100, 120, 110, 90, 95])
        >>> calculate_max_drawdown(pv)
        0.25
    """
    if not isinstance(portfolio_value, pd.Series):
        raise TypeError("portfolio_value must be a pandas Series.")
    if len(portfolio_value) <= 1:
        return 0.0
    return ((portfolio_value.cummax() - portfolio_value) / portfolio_value.cummax()).max() if portfolio_value.max() > 0 else 0

def calculate_hit_ratio(correct_preds: int, trade_counts: int) -> float:
    """
    Calculates the hit ratio (accuracy of trades).

    Args:
        correct_preds (int): Number of correct predictions.
        trade_counts (int): Total number of trades.

    Returns:
        float: Hit ratio, or 0 if no trades.

    Example:
        >>> calculate_hit_ratio(60, 100)
        0.6
    """
    if not all(isinstance(x, int) for x in [correct_preds, trade_counts]):
        raise TypeError("correct_preds and trade_counts must be integers.")
    return correct_preds / trade_counts if trade_counts > 0 else 0

def calculate_win_loss_ratio(correct_preds: Union[int, float], trade_counts: Union[int, float]) -> float:
    """
    Calculates the win/loss ratio of the strategy based on profitable trades.

    Args:
        correct_preds (int | float): Number of winning trades (trades with positive profit).
        trade_counts (int | float): Total number of trades.

    Returns:
        float: Win/loss ratio, or 0.0 if no trades or no losses with no wins.

    Raises:
        ValueError: If inputs are negative or trade_counts < correct_preds.

    Example:
        >>> calculate_win_loss_ratio(60, 100)
        1.5
        >>> calculate_win_loss_ratio(0, 1)
        0.0
    """
    # Type and value checking
    if not all(isinstance(x, (int, float)) for x in [correct_preds, trade_counts]):
        logging.error(f"Invalid input types: correct_preds={type(correct_preds)}, trade_counts={type(trade_counts)}")
        raise TypeError("Inputs must be int or float")
    if correct_preds < 0 or trade_counts < 0:
        logging.error(f"Negative inputs: correct_preds={correct_preds}, trade_counts={trade_counts}")
        raise ValueError("Inputs must be non-negative")
    if trade_counts < correct_preds:
        logging.error(f"correct_preds ({correct_preds}) exceeds trade_counts ({trade_counts})")
        raise ValueError("correct_preds cannot exceed trade_counts")

    win_trades = correct_preds
    loss_trades = trade_counts - win_trades

    # Handle edge cases
    if trade_counts == 0:
        logging.debug("No trades, returning Win/Loss Ratio: 0.0")
        return 0.0
    if loss_trades == 0:
        logging.debug(f"No loss trades, wins={win_trades}, returning {win_trades if win_trades > 0 else 0.0}")
        return float(win_trades) if win_trades > 0 else 0.0

    ratio = win_trades / loss_trades
    logging.debug(f"Win/Loss Ratio: {win_trades}/{loss_trades} = {ratio:.4f}")
    return ratio

def calculate_profit_weighted_accuracy(df: pd.DataFrame, profit_series: pd.Series, strategy: str) -> float:
    """
    Calculates accuracy weighted by profit based on trend correctness.

    Args:
        df (pd.DataFrame): DataFrame with strategy signals and trend.
        profit_series (pd.Series): Series of profit values.
        strategy (str): Strategy name suffix (e.g., 'basic').

    Returns:
        float: Profit-weighted accuracy.

    Example:
        >>> df = pd.DataFrame({
        ...     'buy_basic': [1, 0, 1],
        ...     'sell_basic': [0, 1, 0],
        ...     'trend_pct': [1, -1, 1]
        ... })
        >>> profits = pd.Series([0, 10, 30])
        >>> calculate_profit_weighted_accuracy(df, profits, 'basic')
        1.0
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(profit_series, pd.Series):
        raise TypeError("profit_series must be a pandas Series.")
    required_cols = [f'buy_{strategy}', f'sell_{strategy}', 'trend_pct']
    if not all(col in df.columns for col in required_cols):
        return 0.0
    profit_weight = profit_series.diff().fillna(0)
    trend_correct = ((df[f'buy_{strategy}'] == 1) & (df['trend_pct'] == 1)) | \
                    ((df[f'sell_{strategy}'] == 1) & (df['trend_pct'] == -1))
    return (trend_correct * profit_weight).sum() / profit_weight.abs().sum() if profit_weight.abs().sum() != 0 else 0

def calculate_predictive_hit_ratio(df: pd.DataFrame, next_direction: pd.Series, strategy: str) -> float:
    """
    Calculates how well the strategy's signals predicted the next direction.

    Args:
        df (pd.DataFrame): DataFrame with strategy signals.
        next_direction (pd.Series): Series indicating true future direction.
        strategy (str): Strategy name suffix.

    Returns:
        float: Predictive hit ratio.

    Example:
        >>> df = pd.DataFrame({
        ...     'buy_basic': [1, 0, 0],
        ...     'sell_basic': [0, 1, 0]
        ... })
        >>> next_dir = pd.Series([1, -1, 0])
        >>> calculate_predictive_hit_ratio(df, next_dir, 'basic')
        1.0
    """

    if not isinstance(df, pd.DataFrame) or not isinstance(next_direction, pd.Series):
        raise TypeError("df must be a DataFrame and next_direction must be a Series.")
    required_cols = [f'buy_{strategy}', f'sell_{strategy}']
    if not all(col in df.columns for col in required_cols):
        return 0.0
    direction_pred = pd.Series(0, index=df.index)
    direction_pred[df[f'buy_{strategy}'] == 1] = 1
    direction_pred[df[f'sell_{strategy}'] == 1] = -1
    hits = (direction_pred == next_direction)
    total_signals = (df[f'buy_{strategy}'] != 0) | (df[f'sell_{strategy}'] != 0)
    return hits[total_signals].mean() if total_signals.sum() > 0 else 0

def calculate_total_position_size(df: pd.DataFrame) -> float:
    """
    Calculates the total USD position size of all BTC trades.

    Args:
        df (pd.DataFrame): DataFrame with 'BTC Amount' and 'Entry Price'.

    Returns:
        float: Total position size in dollars.

    Example:
        >>> df = pd.DataFrame({'BTC Amount': [0.1, 0.2], 'Entry Price': [30000, 31000]})
        >>> calculate_total_position_size(df)
        9300.0
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not {'BTC Amount', 'Entry Price'}.issubset(df.columns):
        return 0.0
    return (df['BTC Amount'] * df['Entry Price']).sum() if not df[['BTC Amount', 'Entry Price']].isna().all().all() else 0.0

def calculate_total_margin(total_position_size: float, leverage: float) -> float:
    """
    Calculates total required margin given leverage.

    Args:
        total_position_size (float): Total size of all positions.
        leverage (float): Leverage used.

    Returns:
        float: Total margin required.

    Example:
        >>> calculate_total_margin(10000, 5)
        2000.0
    """

    if not all(isinstance(x, (int, float)) for x in [total_position_size, leverage]):
        raise TypeError("Inputs must be numeric.")
    if leverage <= 0:
        return 0.0
    return total_position_size / leverage

def calculate_avg_margin_per_trade(total_margin: float, trade_counts: int) -> float:
    """
    Calculates average margin required per trade.

    Args:
        total_margin (float): Total margin across all trades.
        trade_counts (int): Total number of trades.

    Returns:
        float: Average margin per trade.

    Example:
        >>> calculate_avg_margin_per_trade(2000, 10)
        200.0
    """

    if not isinstance(total_margin, (float, int)) or not isinstance(trade_counts, int):
        raise TypeError("Inputs must be numeric.")
    return total_margin / trade_counts if trade_counts > 0 else 0

def calculate_roi(initial_investment: float, final_value: float) -> float:
    """
    Calculates the Return on Investment (ROI) as a percentage.

    ROI is a measure of the profitability of an investment, expressed as the
    percentage increase or decrease in value relative to the initial investment.

    Args:
        initial_investment (float): The amount of capital initially invested.
        final_value (float): The final value of the investment after gains or losses.

    Returns:
        float: The ROI as a percentage.
            - A positive value indicates profit.
            - A negative value indicates loss.

    Example:
        >>> initial_investment = 1000
        >>> final_value = 1200
        >>> roi = calculate_roi(initial_investment, final_value)
        >>> print(f"ROI: {roi}%")
        ROI: 20.0%
    """
    if initial_investment <= 0:
        raise ValueError("Initial investment must be greater than zero.")

    roi = ((final_value - initial_investment) / initial_investment) * 100
    return roi



def calculate_strategy_metrics(df: pd.DataFrame, profit_dict: Dict[str, pd.Series],
                               strategies: List[str], periods_per_year: float,
                               next_direction: pd.Series, trade_counts: Dict[str, int],
                               correct_preds: Dict[str, int], risk_free_rate: float = 0.02,
                               leverage: float = 20, initial_cash: float = 10000,
                               total_margin_dict: Dict[str, float] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculates various performance metrics for trading strategies.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing historical price data of the asset being traded.
    profit_dict : Dict[str, pd.Series]
        A dictionary of profit series for each strategy.
    strategies : List[str]
        List of strategy names to evaluate.
    periods_per_year : float
        Number of trading periods per year (typically 252 for daily data).
    next_direction : pd.Series
        Series indicating the predicted direction for each period.
    trade_counts : Dict[str, int]
        A dictionary with the number of trades executed by each strategy.
    correct_preds : Dict[str, int]
        A dictionary with the number of correct predictions for each strategy.
    risk_free_rate : float, default=0.02
        The risk-free rate used in calculating Sharpe and Sortino ratios (annualized).
    leverage : float, default=20
        The leverage used in trading, affecting margin and position size.
    initial_cash : float, default=10000
        The starting capital for the trading strategies.
    total_margin_dict : Dict[str, float], optional, default=None
        A dictionary with the total margin used for each strategy.

    Returns:
    --------
    Dict[str, Dict[str, float]]
        A dictionary containing performance metrics for each strategy.

    Metrics Explained:
    --------------------
    - **Sharpe Ratio**: Measures the risk-adjusted return of a strategy. It is the ratio of the strategy's excess return over the risk-free rate to its standard deviation. A higher Sharpe ratio indicates better risk-adjusted performance.
    - **Sortino Ratio**: A variation of the Sharpe ratio that only considers downside risk (negative returns). This is useful in assessing strategies that are sensitive to drawdowns.
    - **Max Drawdown (MDD)**: The maximum peak-to-trough decline in the portfolio value. It indicates the largest loss from the highest point to the lowest point in the value of the portfolio.
    - **Leverage**: The amount of borrowed capital used to trade. Higher leverage allows you to control larger positions with a smaller amount of capital. However, it also increases risk, as both gains and losses are amplified.
    - **Margin**: The collateral required to open a leveraged position. The amount of margin used depends on the leverage, position size, and risk per trade.
        - At **100x leverage**, you only need a small margin to control a larger position.
        - At **4x leverage**, the margin requirement increases, as you control a smaller position for the same dollar risk.
    - **Profit-Weighted Accuracy**: A measure of how well the strategy performs in relation to the magnitude of profits. It emphasizes accuracy in high-return trades.
    - **Hit Ratio**: The proportion of correct trades (i.e., profitable trades) relative to the total number of trades.
    - **Win/Loss Ratio**: The ratio of winning trades to losing trades. A higher ratio indicates a strategy's ability to generate more winning trades than losing ones.
    - **Total Margin Used**: The total collateral used across all trades for a given strategy. This value depends on the leverage and position size.
    - **Average Margin per Trade**: The average amount of margin required for each trade, which scales down with higher leverage.

    Example Interpretation:
    ------------------------
    - **Leverage and Margin**:
        - At **100x leverage**, even if you're risking $30 per trade, the margin requirement could be as low as $1.50 per trade, because you're controlling a larger position.
        - At **4x leverage**, the margin requirement increases to $37.50 per trade for the same risk amount, since you're controlling a smaller position.
    - **Sharpe and Sortino Ratios**:
        - A higher **Sharpe ratio** (e.g., 1.0 or above) indicates that the strategy is providing higher returns for each unit of risk.
        - A higher **Sortino ratio** (e.g., 1.0 or above) is especially valuable when the strategy is more sensitive to drawdowns.
    - **Max Drawdown (MDD)**:
        - A **lower MDD** (e.g., 0.2) indicates that the strategy has had relatively small declines in portfolio value, while a high MDD (e.g., 0.8) indicates significant drawdowns.
    - **Hit Ratio and Win/Loss Ratio**:
        - A **higher hit ratio** (e.g., 0.7 or 70%) suggests that the strategy is more successful in predicting profitable trades.
        - A **higher win/loss ratio** (e.g., 2:1) indicates that the strategy generates more winning trades compared to losing ones.

    """
    metrics = {}
    # Calculate buy-and-hold metrics once
    buy_hold_return, buy_hold_profit, _ = calculate_buy_hold_profit(df, initial_cash)
    buy_hold_annualized = calculate_buy_hold_annualized(buy_hold_return, periods_per_year, periods)


    for strategy in strategies:
        try:
            print(f"\nDebug: Analyzing strategy: {strategy}")
            profit_series = profit_dict[strategy]
            portfolio_value = calculate_portfolio_value(profit_series, initial_cash)

            if len(portfolio_value) <= 1:
                print("Debug: Portfolio too short, skipping detailed metrics.")
                continue

            returns = calculate_returns(portfolio_value)
            annualized_return = calculate_annualized_return(portfolio_value, initial_cash, periods_per_year, len(df))
            std_return = calculate_std_return(returns)
            downside_deviation = calculate_downside_deviation(returns)

            sharpe = calculate_sharpe_ratio(annualized_return, risk_free_rate, std_return, periods_per_year) \
                if std_return > 0 else 0.0
            sortino = calculate_sortino_ratio(annualized_return, risk_free_rate, downside_deviation, periods_per_year) \
                if downside_deviation > 0 else 0.0
            mdd = calculate_max_drawdown(portfolio_value) if len(portfolio_value) > 1 else 0.0

            hit_ratio = calculate_hit_ratio(correct_preds[strategy], trade_counts[strategy])
            win_loss_ratio = calculate_win_loss_ratio(correct_preds[strategy], trade_counts[strategy])
            profit_weighted_acc = calculate_profit_weighted_accuracy(df, profit_series, strategy)
            predictive_hit_ratio = calculate_predictive_hit_ratio(df, next_direction, strategy)
            total_margin = total_margin_dict.get(strategy, 0.0) if total_margin_dict else 0.0
            avg_margin_per_trade = calculate_avg_margin_per_trade(total_margin, trade_counts[strategy])

            final_profit = profit_series.iloc[-1]
            # Calculate ROI
            roi = calculate_roi(initial_cash, final_profit + initial_cash)

            metrics[strategy] = {
                'Profit-Weighted Acc': profit_weighted_acc,
                'Sharpe': sharpe,
                'Sortino': sortino,
                'MDD': mdd,
                'Hit Ratio': hit_ratio,
                'Win/Loss Ratio': win_loss_ratio,
                'Trades': trade_counts[strategy],
                'Final Profit': final_profit,
                'ROI (%)': roi,
                'Predictive Hit Ratio': predictive_hit_ratio,
                'Buy-and-Hold Profit': buy_hold_profit,
                'Excess Return Over Buy-and-Hold': calculate_excess_return_over_bh(annualized_return, buy_hold_annualized),
                'Leverage': leverage,
                'Total Margin Used': total_margin,
                'Avg Margin per Trade': avg_margin_per_trade
            }

        except Exception as e:
            print(f"Error processing strategy {strategy}: {e}")

    return metrics


def summarize_strategy_results(
        df: pd.DataFrame,
        profit_dict: Dict[str, pd.Series],
        trade_log: List[Dict],
        strategies: List[str],
        initial_cash: float = 10000,
        leverage: float = 4,
        risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Summarizes the results of multiple trading strategies.

    This function processes the results of backtesting multiple trading strategies, calculates key performance
    metrics for each strategy, and returns a summary dataframe that includes information such as the number of trades,
    total profit, average profit per trade, win rate, Sharpe ratio, ROI, and other important performance indicators.

    Args:
        df (pd.DataFrame): The historical price data used for backtesting, typically with columns such as 'Close', 'Volume', etc.
        profit_dict (Dict[str, pd.Series]): A dictionary containing profit/loss data for each strategy, keyed by strategy name.
        trade_log (List[Dict]): A list of dictionaries representing individual trades executed during the backtest.
        strategies (List[str]): A list of strategy names to generate summaries for.
        initial_cash (float, optional): The initial cash available for each strategy's backtest (default is 1000).
        leverage (float, optional): The leverage used in the backtest (default is 20).

    Returns:
        pd.DataFrame: A dataframe containing the summarized results for each strategy. The dataframe includes:
            - 'Strategy': The name of the trading strategy.
            - 'Number of Trades': The total number of trades executed by the strategy.
            - 'Total Profit': The total profit earned by the strategy.
            - 'Avg Profit per Trade': The average profit per trade.
            - 'Avg Pct Change': The average percentage change in price per trade.
            - 'Max Pct Change': The maximum percentage change in price observed in any trade.
            - 'Min Pct Change': The minimum percentage change in price observed in any trade.
            - 'Std Dev Pct Change': The standard deviation of percentage changes across trades.
            - 'Avg Duration (hours)': The average duration of trades in hours.
            - 'Total BTC Traded': The total amount of BTC traded by the strategy.
            - 'Start Date': The start date of the first trade in the backtest.
            - 'End Date': The end date of the last trade in the backtest.
            - 'Win Rate (%)': The percentage of trades that were profitable.
            - 'Sharpe Ratio': The Sharpe ratio of the strategy (measure of risk-adjusted return).
            - 'Profit Factor': The ratio of gross profit to gross loss for the strategy.
            - 'Profit per BTC': The total profit earned per BTC traded by the strategy.
            - 'Initial Capital': The initial capital used for the backtest.
            - 'ROI (%)': The Return on Investment (ROI) as a percentage of the initial capital.

    Example:
        df = pd.read_csv('backtest_data.csv')
        profit_dict = {'strategy1': pd.Series(...), 'strategy2': pd.Series(...)}
        trade_log = [{'Strategy': 'strategy1', 'Profit': 100.0, 'Pct Change': 2.0, ...}]
        strategies = ['strategy1', 'strategy2']
        initial_cash = 1000
        leverage = 20

        summary_df = summarize_strategy_results(df, profit_dict, trade_log, strategies, initial_cash, leverage)
        print(summary_df)
    """
    total_hours, periods_per_year = calculate_time_metrics(df)
    periods = len(df)

    # Calculate buy-and-hold metrics once
    buy_hold_return, buy_hold_profit, _ = calculate_buy_hold_profit(df, initial_cash)
    buy_hold_annualized = calculate_buy_hold_annualized(buy_hold_return, periods_per_year, periods)

    # Initialize counters and metrics
    trade_counts = {strat: 0 for strat in strategies}
    correct_preds = {strat: 0 for strat in strategies}
    total_margin_dict = {strat: 0.0 for strat in strategies}
    exit_stats = {strat: {'trailing_stop': 0, 'profit_target': 0, 'other': 0} for strat in strategies}
    gross_profit = {strat: 0.0 for strat in strategies}
    gross_loss = {strat: 0.0 for strat in strategies}

    # Process trade log
    for trade in trade_log:
        strat = trade.get('Strategy', 'unknown')
        if strat in strategies:
            trade_counts[strat] += 1
            profit = trade.get('Profit', 0.0)
            correct_preds[strat] += 1 if profit > 0 else 0
            total_margin_dict[strat] += trade.get('Margin Used', 0.0) or 0.0
            exit_reason = trade.get('Exit Reason', 'other')
            exit_stats[strat][exit_reason] = exit_stats[strat].get(exit_reason, 0) + 1
            if profit > 0:
                gross_profit[strat] += profit
            else:
                gross_loss[strat] += abs(profit)

    # Compute metrics for each strategy
    summary_data = []
    for strat in strategies:
        # Trade-level stats
        total_trades = trade_counts[strat]
        profits = [t.get('Profit', 0.0) for t in trade_log if t.get('Strategy') == strat]
        total_profit = sum(profits) if profits else 0.0

        # Profit series and portfolio metrics
        profit_series = profit_dict.get(strat, pd.Series(np.zeros(len(df)), index=df.index))
        portfolio_value = calculate_portfolio_value(profit_series, initial_cash)
        returns = calculate_returns(portfolio_value)
        annualized_return = calculate_annualized_return(portfolio_value, initial_cash, periods_per_year, periods)
        std_return = calculate_std_return(returns)
        downside_deviation = calculate_downside_deviation(returns)
        sharpe = calculate_sharpe_ratio(annualized_return, risk_free_rate, std_return, periods_per_year)
        sortino = calculate_sortino_ratio(annualized_return, risk_free_rate, downside_deviation, periods_per_year)
        max_drawdown = calculate_max_drawdown(portfolio_value)

        # Additional metrics using helpers
        win_rate = calculate_hit_ratio(correct_preds[strat], total_trades) * 100  # Convert to %
        profit_factor = gross_profit[strat] / gross_loss[strat] if gross_loss[strat] > 0 else (np.inf if gross_profit[strat] > 0 else 0.0)
        excess_return = calculate_excess_return_over_bh(annualized_return, buy_hold_annualized)
        final_profit = portfolio_value.iloc[-1] if not portfolio_value.empty else initial_cash
        roi = calculate_roi(initial_cash, final_profit)

        # Exit statistics
        exit_summary = {
            'trailing_stop': f"{exit_stats[strat]['trailing_stop']} trades ({exit_stats[strat]['trailing_stop'] / total_trades * 100:.1f}%)" if total_trades > 0 else "0 trades (0.0%)",
            'profit_target': f"{exit_stats[strat]['profit_target']} trades ({exit_stats[strat]['profit_target'] / total_trades * 100:.1f}%)" if total_trades > 0 else "0 trades (0.0%)",
            'other': f"{exit_stats[strat]['other']} trades ({exit_stats[strat]['other'] / total_trades * 100:.1f}%)" if total_trades > 0 else "0 trades (0.0%)"
        }

        # Compile summary row
        summary_row = {
            'Strategy': strat,
            'Number of Trades': total_trades,
            'Total Profit': total_profit,
            'Avg Profit per Trade': total_profit / total_trades if total_trades > 0 else 0.0,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_drawdown,
            'Initial Capital': initial_cash,
            'Final Profit': final_profit,
            'ROI (%)': roi,
            'Buy-and-Hold Profit': buy_hold_profit,
            'Excess Return Over Buy-and-Hold': excess_return,
            'Total Margin Used': total_margin_dict[strat],
            'Leverage': leverage,
            'Exit Statistics': exit_summary
        }
        summary_data.append(summary_row)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    return summary_df
