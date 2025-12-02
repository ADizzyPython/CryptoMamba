
def buy_sell_smart(today, pred, balance, shares, risk=5):
    diff = pred * risk / 100
    if today > pred + diff:
        balance += shares * today
        shares = 0
    elif today > pred:
        factor = (today - pred) / diff
        balance += shares * factor * today
        shares *= (1 - factor)
    elif today > pred - diff:
        factor = (pred - today) / diff
        shares += balance * factor / today
        balance *= (1 - factor)
    else:
        shares += balance / today
        balance = 0
    return balance, shares

def buy_sell_smart_w_short(today, pred, balance, shares, risk=5, max_n_btc=0.002):
    diff = pred * risk / 100
    if today < pred - diff:
        shares += balance / today
        balance = 0
    elif today < pred:
        factor = (pred - today) / diff
        shares += balance * factor / today
        balance *= (1 - factor)
    elif today < pred + diff:
        if shares > 0:
            factor = (today - pred) / diff
            balance += shares * factor * today
            shares *= (1 - factor)
    else:
        balance += (shares + max_n_btc) * today
        shares = -max_n_btc
    return balance, shares

def buy_sell_vanilla(today, pred, balance, shares, tr=0.01):
    tmp = abs((pred - today) / today)
    if tmp < tr:
        return balance, shares
    if pred > today:
        shares += balance / today
        balance = 0
    else:
        balance += shares * today
        shares = 0
    return balance, shares


def buy_sell_smart_prob(today, expected_return, balance, shares, risk=5):
    """
    Trading strategy based on expected return from classification probabilities.
    
    Args:
        today: Current price
        expected_return: Expected % return calculated from class probabilities
        balance: Current cash balance
        shares: Current shares held
        risk: Threshold for taking action (e.g., 0.5 for 0.5% expected return)
    """
    # If expected return is strongly positive (Buy)
    if expected_return > risk:
        # Full buy
        shares += balance / today
        balance = 0
        
    # If expected return is strongly negative (Sell)
    elif expected_return < -risk:
        # Full sell
        balance += shares * today
        shares = 0
        
    # If expected return is small (Neutral/Hold)
    # We could implement scaling here, but for now we just hold
    else:
        pass
        
    return balance, shares

def trade(data, time_key, timstamps, targets, preds, balance=100, mode='smart_v2', risk=5, y_key='Close', is_classification=False):
    balance_in_time = [balance]
    shares = 0

    for ts, target, pred in zip(timstamps, targets, preds):
        # Find the index of the current timestamp
        current_indices = data.index[data[time_key] == int(ts)].tolist()
        if not current_indices:
            continue # Skip if timestamp not found
        
        current_idx = current_indices[0]
        
        # Ensure we have a previous index
        if current_idx == 0:
            continue
            
        # Get the previous day's close (or whatever y_key is)
        today = data.iloc[current_idx - 1][y_key]
        
        if is_classification:
             # For classification, 'pred' is expected return or probability structure
             # 'smart_prob' mode uses expected return directly
             if mode == 'smart_prob':
                 balance, shares = buy_sell_smart_prob(today, pred, balance, shares, risk=risk)
        else:
            if mode == 'smart':
                balance, shares = buy_sell_smart(today, pred, balance, shares, risk=risk)
            if mode == 'smart_w_short':
                balance, shares = buy_sell_smart_w_short(today, pred, balance, shares, risk=risk, max_n_btc=0.002)
            elif mode == 'vanilla':
                balance, shares = buy_sell_vanilla(today, pred, balance, shares)
            elif mode == 'no_strategy':
                shares += balance / today
                balance = 0
                
        balance_in_time.append(shares * today + balance)

    # Final balance calculation
    # Note: targets[-1] might be class index for classification, need actual price
    # We use the price at the last timestamp
    last_price = data[data[time_key] == int(timstamps[-1])].iloc[0][y_key]
    balance += shares * last_price
    return balance, balance_in_time