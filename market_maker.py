import random
import math
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
def call_delta(S,K,T, r, sigma):
      T=T/(252*6.5*60*60*100) # need to convert the seconds to years
      d1 = (math.log(S/K) + (r+.5*sigma**2)*T) / (sigma * math.sqrt(T))
      return norm.cdf(d1)

def add_to_book(b, OQ, OP, SQ, SP):
  b.append({'OQ': OQ, 'OP':OP, 'SQ': SQ, 'SP': SP})
  return b

def market_order(bid_spread_prop, ask_spread_prop):
  
  bid_prob = 1- bid_spread_prop / (bid_spread_prop + ask_spread_prop) # lower is better
  ask_prob = 1 - ask_spread_prop / (bid_spread_prop + ask_spread_prop)# lower is better

  side = random.choices(population =['buy','sell'], weights = [bid_prob, ask_prob], k=1)[0]
  size =  random.randint(1, 10)

  # fill is based on the bid fair ask prob: 
  r = random.random()
  if side == 'buy':

    # my fill probs is just a function I plotted and think it is somewhat reasonable, but there is definitely room for improvement. The thing is that modeling a very good fill function is out of the scope of this project
    if r> math.exp(-30*bid_spread_prop) or bid_spread_prop > .1: 
      print("not crazy")
      return 0 # the proportion of the spread being .1 is unreasonable so it will just auto reject
  else: 
    if r > math.exp(-30*ask_spread_prop) or ask_spread_prop > .1: 
      print("not crazy2 ")
      return 0

  return {'side': side, 'size': size}

def process_market_order(long_book, short_book, m, cash, bid, ask, Stock_Price, delta):
    side = m['side']   # buy or sell 50 50 shot 
    size = m['size'] 


    # side is buy so market maker is selling
    if side == 'buy': 
      # we need to check our long books size
      if len(long_book) >0:
        # we have inventory in our long book so we will sell from there and liquidate corresponding hedge

        # looping through long book until the order is processed or we need to create calls
        remaining = size
        while remaining > 0 and len(long_book) > 0:
          top_order = long_book[0] # this is by fifo 
          units = min(remaining, top_order['OQ'])

          # We need to credit ask * units to unwind the long position
          cash += ask*units*100

          # We need to unwind our hedge at that price. Since we are short stock in long positions, we need to buy stock in a proportional manner to the number of stock / option we bought before
          hedge_shares_to_buy = (top_order['SQ'] / top_order['OQ']) * units

          cash -= Stock_Price * hedge_shares_to_buy 
          top_order['OQ'] -= units

          if top_order['OQ'] == 0: # if the OQ is 0 we want to take it away from the long book and decrease remaining by units which would be oq
            long_book.pop(0)
            remaining -= units
            continue
          top_order['SQ'] -= hedge_shares_to_buy # stock quantity exposure needs to decrease by the number of shares we hedged with
          remaining -= units 

        if remaining > 0: # this means that long_book is empty so it needs to be handled
          size = remaining 
          cash += ask*size*100 # credit us yes 

          
          hedge_shares = delta*size*100 # since an option represents 100 units of the underlying
          cash -= Stock_Price*hedge_shares # we bought that many shares

          # adding to the short book
          short_book = add_to_book(short_book, size, ask, hedge_shares, Stock_Price)
      else:
        # our long book is empty so we process the order by selling an option add its info to the short book
        # credit the sell
        cash += ask * size*100

        # delta hedge the sell
        hedge_shares =delta*size*100 # since an option represents 100 units of the underlying      
        cash -= Stock_Price*hedge_shares # we bought that many shares

        # adding to the short book  # logging the order to the short book
        short_book = add_to_book(short_book, size, ask, hedge_shares, Stock_Price)
    else:  # market maker is a buyer 
      #trader wants to sell so mkt maker is a buyer
      # we are short some options, so we want to buy to cover 
      if len(short_book) > 0:
        remaining = size
        while remaining >0 and len(short_book) > 0:
          top_order = short_book[0]
          units = min(remaining, top_order['OQ'])

          # We need to debit bid*units to cover the short position
          cash -= bid*units*100

          #We need to unwind our hedge at that price. Since we are long stock in short options positions, we need to short stock in a proportional manner to the number of stock / option we bought before
          hedge_shares_to_sell = (top_order['SQ'] / top_order['OQ']) * units 

          cash += Stock_Price * hedge_shares_to_sell
          top_order['OQ'] -= units

          if top_order['OQ'] == 0:
            short_book.pop(0)
            remaining -= units
            continue
          top_order['SQ'] -= hedge_shares_to_sell

          remaining -= units

        # if this is met then the code is identical to what happens in the else statement below so long as the size = remaining
        if remaining > 0:
          size = remaining
          cash -= bid * size*100
         
          hedge_shares =delta*size*100 # since an option represents 100 units of the underlying
          cash += Stock_Price*hedge_shares # so really we are being credited here

          # adding to the long book
          long_book = add_to_book(long_book, size, bid, hedge_shares, Stock_Price)
      else:
        # our short book is empty, so we process the order by buying the option
        # credit the sell

        # We buy options so we are long delta meaning we need to hedge by shorting it
        cash -= bid * size*100

       
        hedge_shares = delta*size*100 # since an option represents 100 units of the underlying
        cash += Stock_Price*hedge_shares # we are being credited here

        # adding to the long book
        long_book = add_to_book(long_book, size, bid, hedge_shares, Stock_Price)

    return long_book, short_book, cash

def BS(S, K, T, r, sigma): ## had to change comment structure to hashtag because of streamlit reading quotes as something for it to display
  #S: Stock Price
  #K: Strike Price
  #T: Time until expiration passed in as centi-seconds, but converted to years below
  #r: T bill rate
  #sigma: Volatility of Underlying

  #d1, and d2 are variables that are composed of the ones above and are standard practice in BS. 
  #d1: A measure used to calculate the risk adjusted likelihood that the option will finish in the money
  #d2: The probability the option will expire in the money adjusting for volatility over time
 
  T = T / (252*6.5*60*60*100) # Converting time from centi second to years
  d1 = (math.log(S/K) + (r+.5*sigma**2)*T)/(sigma*math.sqrt(T))
  d2 = d1 - sigma * math.sqrt(T)
  return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)

def rehedge_book(long_book, short_book, Stock_Price, delta, cash): 
    # the way long and short books are defined, either 1 will have size or neither will
    current_total_hedge = 0
    target_hedge = 0
    if len(long_book) > 0: 
      current_total_hedge = sum(item['SQ'] for item in long_book)
      target_hedge = delta * 100 * sum(item['OQ'] for item in long_book)
      Cum_Options_Quant = sum(item['OQ'] for item in long_book)

      hedge_diff = target_hedge - current_total_hedge

      if hedge_diff > 0: # we need to add to our position, since our long book is short shares we need to short more shares
        cash += hedge_diff * Stock_Price 
        
        # updating position, since we are adding to the short position
        for o in long_book: 
          # adjusting entry stock price for each item in the short book
          Respective_Prop = o['OQ'] / Cum_Options_Quant
          Increase_In_SQ = hedge_diff*Respective_Prop
          o['SP'] = (o['SP']*o['SQ'] + Stock_Price * Increase_In_SQ) / (o['SQ'] + Increase_In_SQ)
          # updating SQ
          o['SQ'] +=Increase_In_SQ

      else: 
         # We have to buy stock 
        cash -= abs(hedge_diff)*Stock_Price

        # updating stock quantity
        for o in long_book: 
          Respective_Prop = o['OQ'] / Cum_Options_Quant
          Decrease_In_SQ = abs(hedge_diff)*Respective_Prop
          o['SQ']-= Decrease_In_SQ

    elif len(short_book) > 0: 

      current_total_hedge = sum(item['SQ'] for item in short_book)
      target_hedge = delta * 100 * sum(item['OQ'] for item in short_book)
      Cum_Options_Quant = sum(item['OQ'] for item in short_book)

      hedge_diff = target_hedge - current_total_hedge

      if hedge_diff > 0: 
        # we need to buy more shares 
        cash -= hedge_diff * Stock_Price

        # need to update stock quantity and entry price accross all orders
       
        for o in short_book: 
          # adjusting entry stock price for each item in the short book
          Respective_Prop = o['OQ'] / Cum_Options_Quant
          Increase_In_SQ = hedge_diff*Respective_Prop
          o['SP'] = (o['SP']*o['SQ'] + Stock_Price * Increase_In_SQ) / (o['SQ'] + Increase_In_SQ)
          # updating SQ
          o['SQ'] +=Increase_In_SQ

      else: 
        # We have to sell off 
        cash += abs(hedge_diff)*Stock_Price

        # updating stock quantity
        for o in short_book: 
          Respective_Prop = o['OQ'] / Cum_Options_Quant
          Decrease_In_SQ = abs(hedge_diff)*Respective_Prop
          o['SQ']-= Decrease_In_SQ  
    return long_book, short_book, cash, current_total_hedge, target_hedge


def current_pnl(long_book, short_book, Stock_Price, BS_Call_Price, cash): 
  pnl = cash - 5000000 

  if len(long_book) > 0: 
    for pos in long_book: 
     # option_pnl = (BS_Call_Price - pos['OP'])*100*pos['OQ']
      #stock_pnl = (pos['SP'] - Stock_Price) * pos['SQ']

      option_exposure = BS_Call_Price*100*pos['OQ']
      stock_exposure = Stock_Price * pos['SQ'] # since I own options and short stocks, I need to do option - stock price 
      net_exposure = option_exposure - stock_exposure

      pnl += net_exposure # option_pnl + stock_pnl +
  elif len(short_book) > 0: 
    for pos in short_book: 
      #option_pnl = (pos['OP'] - BS_Call_Price)*100*pos['OQ']
      #stock_pnl = (Stock_Price -pos['SP'])* pos['SQ']

      option_exposure = BS_Call_Price*100*pos['OQ']
      stock_exposure = Stock_Price * pos['SQ'] # since I own options and short stocks, I need to do option - stock price 
      net_exposure =  stock_exposure - option_exposure

      pnl += net_exposure #option_pnl + stock_pnl + 
  return pnl

def gbm_step(S_t, mu, vol):
  dt = 1/(589680000) # this represents 1/100th of a second in terms of trading days by trading hours by minutes by seconds by a 100th of second
  Z = np.random.normal()
  S_t1 = S_t * np.exp((mu - 0.5 * vol**2)*dt + vol * np.sqrt(dt) * Z)
  return S_t1

# every second this thing is going to show 1/60th of second jumps 
def simulation(delta_path, price_series, mu, vol, option_price_series, K, T, r, sigma, long_book, short_book, cash, fair_value_ask_spread, bid_fair_value_spread):
  for i in range(100): 
    p = gbm_step(price_series[-1], mu, vol)
    T -=1
    price_series.append(p)
    # add all logic in here
    option_price = BS(p, K, T, r, sigma)
    option_price_series.append(option_price)

    bid_prop = bid_fair_value_spread / option_price
    ask_prop = fair_value_ask_spread / option_price
    m = market_order(bid_prop, ask_prop)
    if m == 0: # market_order returns 0 when they don't put in an order because your spread is too greedy
      continue
    delta = call_delta(p, K,T, r, sigma)
    delta_path.append(delta)
    bid = option_price - bid_fair_value_spread
    ask = option_price + fair_value_ask_spread
    long_book, short_book, cash = process_market_order(long_book, short_book, m, cash, bid, ask, p, delta)  

  return T, long_book, short_book, cash

# GUI session initialization
import streamlit as st
st.set_page_config(layout="wide")
st.title("Black-Scholes market making simulation for a single call option including delta hedging")
st.write("Prior to running the simulation, please feel free to visit this public google colab notebook I made that Derives the Black-Scholes Pricing model, explains the greeks, and explains delta hedging https://colab.research.google.com/drive/1MyxcOBN823yFH8SaGyyUcOAChQyHPBRm?usp=sharing")
st.write(" ")
st.write("Your goal is to make as much money as you can. Since we are simulating the underlying with GBM (see colab notebook), you aren't really exposed to any risk, so the name of the game is finding the sweetspot for your bid to Black-Scholes spread and Black-Scholes to ask spread that gives you the steapest pnl curve. Fill probabilities are modeled in a way that the tighter your spread is the more likely you are to get filled exponentially with a max of 100 orders per second that have a quanitity of 1 to 10 units per order. This is because each time step is a centi-second, but you see updates and rehedge every second for a total of 60 seconds. You should also try to balance your inventory, but there is no explicit constraints that force you to (This is something that could be added in a future revision). \n \n " )
st.title("IMPORTANT: The first few runs are laggy (wait until time step 6 before you click off at the bare minumum, though it should only take 3 ish) and the charts don't fully load in. Also sometimes the charts load in weird with scientific notation above, please just accept this. Have fun! \n \n PS: You can zoom in on charts and tables if you hover over them and click the expanding box button. To exit its at the top right of your screen.  \n \n \n Also, if for some reason the PnL curve is negative or flat or crashes, it is because the streamlit lagged weirdly and the spread value was not properly accounted for. Just rerun the sim if this happens. Feel free to check the code. There is a readme on my github")
st.title("Instructions: Start the simulation. Once the game starts adjust your bid-BS fair value spread and BS fair value - ask spreads as the simulation runs.")
col1, col2, col3, col4 = st.columns([1,1,1,1])
if 'running' not in st.session_state: 
  st.session_state.running = False

if 'step' not in st.session_state: 
    st.session_state.step = 0
if "started" not in st.session_state:
    st.session_state.started = False

start_placeholder = st.empty()

if not st.session_state.started:
  if start_placeholder.button("Start Simulation"):
      st.session_state.started = True
      st.session_state.running = True
      st.session_state.long_book = []
      st.session_state.short_book = []
      st.session_state.cash = 5_000_000
      st.session_state.Stock_Price = round(random.uniform(10, 500), 2)
      st.session_state.Strike_Price = round(st.session_state.Stock_Price * random.uniform(0.7, 1.3), 2)
      st.session_state.Time_to_Maturity = random.randint(30, 60)*6.5*60*60*100
      st.session_state.vol = round(random.uniform(0.1, 0.9), 4)
      st.session_state.risk_free_rate = round(random.uniform(0.005, 0.06), 4)
      st.session_state.pnl_history = []
      st.set_page_config(layout="wide")
      st.session_state.BS_Call_Price = 0
      
      st.session_state.gbm_path = []
      st.session_state.option_path = []
      st.session_state.mu = random.uniform(-10,10)
      st.session_state.prev_hedge = 0
      st.session_state.new_hedge = 0
      st.session_state.new_hedge_minus_prev_hedge= []
      st.session_state.delta_path = []
      st.session_state.option_exposure = []
      st.session_state.stock_exposure = []
      st.session_state.net_exposure = []
      st.session_state.spread_max = round(BS(st.session_state.Stock_Price, st.session_state.Strike_Price, st.session_state.Time_to_Maturity, st.session_state.risk_free_rate, st.session_state.vol), 2)*.1
      start_placeholder.empty()
      st.rerun()

 


if st.session_state.running: 
  BS_Call_Price = round(BS(st.session_state.Stock_Price, st.session_state.Strike_Price, st.session_state.Time_to_Maturity, st.session_state.risk_free_rate, st.session_state.vol), 2)
  st.session_state.BS_Call_Price = BS_Call_Price


  with col1:
      st.markdown(f"### Second: {st.session_state.step}")
      st.write(f"Stock Price: ${st.session_state.Stock_Price:.2f}")
      st.write(f"Strike Price: ${st.session_state.Strike_Price:.2f}")
      st.write(f"Time to Maturity (centi-seconds): {st.session_state.Time_to_Maturity}")
      st.write(f"Time_to_Maturity (days) {st.session_state.Time_to_Maturity/(6.5*60*60*100)}")
      st.write(f"Volatility: {st.session_state.vol:.2f}")
      st.write(f"Risk-free rate: {st.session_state.risk_free_rate:.3f}")
      st.write(f"Cash: ${st.session_state.cash:,.2f}")
      st.write(f"Current_hedge: {st.session_state.prev_hedge}")
      st.write(f"Target_hedge:  {st.session_state.new_hedge}")
      st.write(f"Target_hedge - Current_hedge = {st.session_state.new_hedge - st.session_state.prev_hedge}")
      st.markdown(f"### OPTION PRICE AKA BS_Price: {BS_Call_Price} \n if you feel this is too low feel free to restart")
      

      st.number_input(
          "Decrease this spread and increase the spread below to offload your long book (Options you own). Adjust this as you go to try and maximize profits. You can use your KEYBOARD. Fills are probabilistic \n Bid-BS fair value spread ($)", min_value=0.01, max_value= st.session_state.spread_max, key='bid_fair_value_spread', step=0.01)
      st.number_input(
          "Decrease this spread and increase the spread above to cover your short book (Options you owe). Adjust this as you go to try and maximize profits. You can use your KEYBOARD. Fills are probabilistic \n BS fair value spread ($)", min_value=0.01, max_value= st.session_state.spread_max, key = 'fair_value_ask_spread', step=0.01)
      st.session_state.prev_bid_fair_value_spread =  st.session_state.bid_fair_value_spread 
      st.session_state.prev_fair_value_ask_spread = st.session_state.fair_value_ask_spread
  with col2: 

    Stock_Plot = plt.figure(figsize = (5,2))
    plt.plot(st.session_state.gbm_path)
    plt.title("Stock Price", fontsize = 5)
    plt.xlabel("Time in centi-seconds", fontsize = 5)
    plt.ylabel("Price", fontsize = 5)
    plt.xticks(fontsize = 4)
    plt.yticks(fontsize = 4)
    st.pyplot(Stock_Plot)

    Option_Plot = plt.figure(figsize = (5,2))
    plt.plot(st.session_state.option_path)
    plt.title("Call Option Price", fontsize = 5)
    plt.xlabel("Time in centi-seconds", fontsize = 5)
    plt.ylabel("Price", fontsize = 5)
    plt.xticks(fontsize = 4)
    plt.yticks(fontsize = 4)
    st.pyplot(Option_Plot)

    
    
  with col3: 

    pnl_plot = plt.figure(figsize = (5, 2))
    plt.plot(st.session_state.pnl_history)
    plt.title("PnL", fontsize = 5)
    plt.xlabel("Time in seconds", fontsize = 5)
    plt.ylabel("$", fontsize = 5)
    plt.xticks(fontsize = 4)
    plt.yticks(fontsize = 4)
    st.pyplot(pnl_plot)

    hedge_plot = plt.figure(figsize = (5, 2))
    plt.plot(st.session_state.new_hedge_minus_prev_hedge)
    plt.xlabel("Time", fontsize = 5)
    plt.ylabel("Hedge_Quantity", fontsize = 5)
    plt.title("Target hedge - Current hedge", fontsize = 5)
    plt.xticks(fontsize = 4)
    plt.yticks(fontsize = 4)
    st.pyplot(hedge_plot)

    

  with col4: 
    Delta_plot = plt.figure(figsize = (5, 2))
    plt.plot(st.session_state.delta_path)
    plt.title("Call Delta", fontsize = 5)
    plt.xlabel('Time in centi-seconds', fontsize = 5)
    plt.ylabel("Delta", fontsize = 5)
    plt.xticks(fontsize = 5)
    plt.yticks(fontsize = 5)
    plt.legend()
    st.pyplot(Delta_plot)


    import pandas as pd
    st.subheader("Long Book (Bought Options)")
    df_long = pd.DataFrame(st.session_state.long_book)
    if not df_long.empty:
      df_long.sort_index(ascending = False, inplace = True)
      df_long['OP'] = df_long['OP'].round(2)
      df_long['SP'] = df_long['SP'].round(2)

      st.dataframe(df_long)

    st.subheader("Short Book (Sold Options)")
    df_short = pd.DataFrame(st.session_state.short_book)
    if not df_short.empty:
      df_short.sort_index(ascending = False, inplace = True)
      df_short['OP'] = df_short['OP'].round(2)
      df_short['SP'] = df_short['SP'].round(2)
      
      st.dataframe(df_short)



  if st.session_state.step == 0: 
    pass
  if st.session_state.step == 1: # step 1 is really step 0 here because I was trying to fix bug of jumping from 0 to 3
    st.session_state.gbm_path.append(st.session_state.Stock_Price)
    st.session_state.option_path.append(BS(st.session_state.Stock_Price, st.session_state.Strike_Price, st.session_state.Time_to_Maturity, st.session_state.risk_free_rate, st.session_state.vol))
    st.session_state.Stock_Price = st.session_state.gbm_path[-1]

  if st.session_state.step > 1:
      # leaving this for simplicity, but it will be different later
    # bid = BS_Call_Price - .5
      #ask = BS_Call_Price + .5
      
      # this is going to run and process orders over the minute then user will rehedge and be shown pnl, the chart shows it after it happend, may cause issues with not seeing last thing so think about that later
      st.session_state.Time_to_Maturity, st.session_state.long_book, st.session_state.short_book, st.session_state.cash = simulation(st.session_state.delta_path, st.session_state.gbm_path, st.session_state.mu, st.session_state.vol,st.session_state.option_path, st.session_state.Strike_Price, st.session_state.Time_to_Maturity, st.session_state.risk_free_rate, st.session_state.vol, st.session_state.long_book, st.session_state.short_book, st.session_state.cash, st.session_state.fair_value_ask_spread, st.session_state.bid_fair_value_spread)
      st.session_state.Stock_Price = st.session_state.gbm_path[-1]
      BS_Call_Price = st.session_state.option_path[-1]

      # hedging and pnl calculation happens here at each time step 
      delta = call_delta(st.session_state.Stock_Price, st.session_state.Strike_Price, st.session_state.Time_to_Maturity, st.session_state.risk_free_rate, st.session_state.vol)
      st.session_state.long_book, st.session_state.short_book, st.session_state.cash, st.session_state.prev_hedge, st.session_state.new_hedge = rehedge_book(st.session_state.long_book, st.session_state.short_book, st.session_state.Stock_Price, delta, st.session_state.cash)

      st.session_state.new_hedge_minus_prev_hedge.append(st.session_state.new_hedge - st.session_state.prev_hedge)
    
      # calculate pnl
      st.session_state.pnl = current_pnl(st.session_state.long_book, st.session_state.short_book, st.session_state.Stock_Price, BS_Call_Price, st.session_state.cash)
      st.session_state.pnl_history.append(st.session_state.pnl)

  if st.session_state.step < 60: 
    time.sleep(1)
    st.session_state.step += 1
    st.rerun()
      

    
  
  if st.session_state.step == 60:
    st.write(f"total PNL = {st.session_state.pnl}")