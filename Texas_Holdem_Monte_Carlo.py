from deuces import Deck, Card, Evaluator
# can clean all of the code up in the future if desired but doesnt matter

import numpy as np
import random 
import matplotlib.pyplot as plt
import pandas as pd 

class sim:
    #fix later some bs issue
    def __init__(self): 
        self.community_cards = []
        self.human_cards = []
        self.evaluator = Evaluator()
        self.numOpps = int(input("Enter the number of oponents you have: "))
        self.numCom = 0

    def get_human_cards(self): 
        print("DONT MESS UP NO PRECAUTIONS IN PLACE: Enter your first card then type enter button as follows 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A then s, d, h, c for example 4h is four of hearts")
        card1 = Card.new(input("Enter your first card: "))
        card2 = Card.new(input("Enter your second card: "))

        self.human_cards = [card1, card2]
    
    def get_community_cards(self): 
        self.numCom = int(input("Enter the number of community cards on the board: "))

        for i in range(self.numCom):
            self.community_cards.append(Card.new(input("Enter a card same formatting as b4 no mistakes allowed: ")))

    
    def Monte_Carlo_Flop(self): # hc human cards, cc community cards
            hc : list[int]= self.human_cards
            cc :list[int] = self.community_cards
            print(f"hc {hc} cc {cc}")
            wins, ties, losses = 0,0,0

            num_simulations = 10000

            results = np.zeros(num_simulations)

            hand_freq = {
                "High Card": 0,
                "One Pair": 0,
                "Two Pair": 0,
                "Three of a Kind": 0,
                "Straight": 0,
                "Flush": 0,
                "Full House": 0,
                "Four of a Kind": 0,
                "Straight Flush": 0,
                "Royal Flush": 0
                }

            for _ in range(num_simulations):

                # this clones the deck by removing the known cards
                montydeck = Deck()
                known_cards = hc + cc
                for card in known_cards: 
                    montydeck.cards.remove(card)

                # deal opponents hole cards for 3 opponents
                opponent_hands = [montydeck.draw(2) for _ in range(self.numOpps)]

                # deal rest of board turn and river
                if(self.numCom == 3):
                    remaining_board = montydeck.draw(2) 
                    full_board = cc+ remaining_board
                elif(self.numCom == 4): 
                    remaining_board = montydeck.draw(1) 
                    full_board = cc +remaining_board
                else: 
                    full_board = cc
                # Evaluate teh players hand 

                full_board: list[int] = cc

                player_score = self.evaluator.evaluate(hc, full_board)
                hand_type = self.evaluator.get_rank_class(player_score)
                
                
                hand_freq [self.get_hand(hand_type)] += 1


                opponent_scores = [ self.evaluator.evaluate(opp_hand, full_board) for opp_hand in opponent_hands]


                if player_score < min(opponent_scores): 
                    results[_] = 1 # winner 
                elif player_score == min(opponent_scores): 
                    if opponent_scores.count(player_score) == 1: 
                        results [_] = .5 # tie
                    else: 
                        results[_] = -1 # loss for multi tie ( not sure about this have to thnk through more but leaving for now )
                
                else: 
                    results[_] = -1 # loss

                wins = np.sum(results == 1) 
                ties = np.sum(results == .5)
                losses = np.sum(results == -1)


            total = len(results) 
            win_prob = (wins/total) * 100 
            tie_prob = (ties/total) *100
            loss_prob = (losses/total)*100

            hand_df = pd.DataFrame(list(hand_freq.items()), columns = ["Hand Type", "Frequency"])

            fig, axes = plt.subplots(1, 2, figsize = (14, 6))


            # create a plot 
            categories = ["Win", "Tie", "Loss"]
            probabilities = [win_prob, tie_prob, loss_prob]
            #Plotting 
            bars = axes[0].bar(categories, probabilities, color = ['green', 'blue', 'red'])
            axes[0].set_title('Monte Carlo Simulation Results')
            axes[0].set_xlabel('Outcome')
            axes[0].set_ylabel('Prob %')
            axes[0].set_ylim(0,100)



            # Add the probability percentage above each bar
            for bar in bars:
                yval = bar.get_height()  # Get the height of the bar (the probability)
                axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10)
        
            font_size = 12
        # Plot Hnad Freq
            bars2 = axes[1].bar(hand_df["Hand Type"], hand_df["Frequency"], color = 'lightblue')
            axes[1].set_title('Hand Frequencies')
            axes[1].set_xlabel("Hand Type")
            axes[1].set_ylabel("Frequency")
            axes[1].tick_params(axis = 'x', labelsize = 6)

            # Add frequency values above each bar
            for bar in bars2:
                height = bar.get_height()  # Get the height of each bar
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,  # x position (center of the bar)
                    height + 0.5,  # y position (slightly above the bar)
                    f'{int(height)}',  # Text to display (convert to integer if needed)
                    ha='center',  # Horizontal alignment
                    va='bottom',  # Vertical alignment
                    fontsize=8  # Smaller font size for the numbers
                )



                # Add a textbox panel below
            textbox_ax = fig.add_axes([0.1, 0.02, 0.8, 0.1])  # [left, bottom, width, height]
            textbox_ax.axis("off")  # Turn off the axes
            textbox_ax.text(
                0.5,
                0.5,
                f" community cards: {"  ".join([Card.int_to_pretty_str(card) for card in cc])} \n \n your cards: {"  ".join([Card.int_to_pretty_str(card) for card in hc])} ",
                ha="center",
                va="center",
                fontsize=12,
                # can adjust font color later if want but not really necessary
            )

            plt.tight_layout(rect=[0, 0.1, 1, 1]) 
            plt.show()

    def get_hand(self, hand_rank): 
        if hand_rank == 0:
            return "Royal Flush"
        elif hand_rank == 1:
            return "Straight Flush"
        elif hand_rank == 2:
            return "Four of a Kind"
        elif hand_rank == 3:
            return "Full House"
        elif hand_rank == 4:
            return "Flush"
        elif hand_rank == 5:
            return "Straight"
        elif hand_rank == 6:
            return "Three of a Kind"
        elif hand_rank == 7:
            return "Two Pair"
        elif hand_rank == 8:
            return "One Pair"
        else:
            return "High Card"


if __name__ == "__main__": 
   s = sim()
   s.get_human_cards()
   s.get_community_cards()
   s.Monte_Carlo_Flop()
