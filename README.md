# TODO

__Goal:__ Create a class to test a investment strategy
 An investment strategy outputs Calls:

 Strategy:
    inputs: df
    output: a Call 

    - buy: coins to buy -> Call 
    - sell: coins to sell -> Call
    - liquid: coins to sell if no more liquid -> Call

 A Call:
    - timestamp
    - trading pair
    - buy / sell
    - how much asset
    - platform


Portfolio:
    - balance_states
    - pnl_states


BacktestingEngine:
    - date_from
    - date_to
    - trading pair
    - platform
    
    - apply(strategy, portfolio, date_from, date_to) -> return PnL
      * init Portfolio
      * for t in [date_from, date_to, freq]
        * sell() 
        * 


Strategy: method of simulator that can access the balance
---------

- df_in:
one can add as many columns as he likes

------------------------------------------------------------------------------------------------------------------------
timestamp | pair | asset | quote | price_open | price_high | price_close | price_low | volume | buy | sell | quantity

- df_out

----------------------------------------------------------
timestamp | pair | asset | quote | buy | sell | quantity

Simulator:
----------

- balance
- tick(window, step): timestamp changes, df_in changes, apply strategy(df_in), update balance
- balance_hist
- pnl_hist


#### 

cache disk: symbol interval