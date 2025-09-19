README FOR MARKET\_MAKER.PY





This project simulates an options market maker trading a single call option. The market maker hedges delta whenever they get an order and rehedges at every 1 second time step. The underlying is modeled as GBM and the option is modeled by Black Scholes. The simulation is intended for educational purposes, and the colab link provided when you go to the Streamlit link touches on some of the down falls of Black-Scholes, and why it is unrealistic. 



Here is a list of all the functions and what they do. The code has plenty of comments to fill in the gaps. 



**call\_delta:** 

Takes in Stock price, Strike, Time till expiry, risk free rate, and stock volatility

returns the delta of the stock 



**add\_to\_book:** 

Takes in the book of holdings, quantity of options someone ordered, the price of options, the stock quantity and the stock price, and adds the order to the book 

returns the updated book



**market\_order:**

Decides the probability of a market order and the side you are going to get based on your bid BS value spread and BS value ask spread. It works using e^-30(respective spread) for both. If you get an order the method randomly picks a size between 1 and 10

returns the side and size or 0 if no order on that pass through



**process\_market\_order:**

Takes in the long\_book, short\_book, market order, cash, bid, ask, stock price, call delta



The long book is a book of long options and short stock and the short book is a book of short options and long stock. 



process\_market\_order checks the side of the order then processes it by adding it to either the long book or short book and adjusting cash, or taking from inventory which is represented on the long or short book and covering the balance with cash and adding to opposite book. Whenever we process an order we hedge it by taking an opposing position in the stock times delta \* 100



returns the updated long book, short book, and cash



**BS:** 

Takes in the stock price, strike price, time until expiration in centi seconds, risk free rate, and vol of stock



converts time until expiry from centi-seconds to years then computes the call options value using the Black-Scholes pricing model 



returns the price of the option



**rehedge\_book**

Takes in the long\_book, short\_book, Stock\_Price, call delta, and cash



Note that the long\_book and short\_book are defined so only 1 can have items in it at a time. 



This method goes through the book and takes the current total hedge that the book has and compares it to the target hedge. How we were hedged, vs where we need to be, then adjusts the book and cash accordingly



returns long\_book, short\_book,cash, curretn\_total\_hedge,target\_hedge



**current\_pnl**

Takes in long\_book, short\_book, Stock\_price, BS\_call\_price, cash



It is important to know how cash works in this simulation. If I buy an option I debit cash then credit cash corresponding to my hedge at time t = 0, so if we step forward to t = 1 and the prices change, I can compute my pnl by doing current cash - initial stake, then I need to mark to market the value of the option and the call and credit cash corresponding to the options fair value (BS value), and debit cash corresponding to the stock positions current value. This way I am left with profits that have already been made, and in thoery half the spread per position.





**gbm\_step**



This is kind of weird because gbm is continuous by nature, but computationally I just say 1/100th of a second is a time step, which is pretty small for the purposes of a simple educational simulator. This uses Euler-Maruyama discretization of GBM, which makes the sim reasonably continuous for demo purposes



Takes in Stock price, trend, and vol

returns new stock price according to gbm



**simulation:** 

Takes in delta path, price series, mu, vol, option\_price\_series,Strike, Time until maturity, sigma, long\_book, short\_book, and cash. 



Builds up a price series by appending to price series. Does 100 loops calling gbm\_step to give the price, then calls BS to get a corresponding options price and then gives random market orders based on how 'fair' the quotes are.  

