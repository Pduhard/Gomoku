import pygame
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


class Graph:
    """
        2 graph type:
            - Graphs over the game/turns. Use all snapshots data
                Save a snapshot id representing the beginning of graphs/game
            - Graphs about one turn/snapshot data
                Always use the last snapshot

        Graphs:
            - Player 0: StateQuality + heuristic
            - Player 1: StateQuality + heuristic


        'G' pressed modify internal state. ON/OFF
        
        draw():
            Fetch data we want to print in the future
            ON:
                Dynamic update of all graphs
    """
    def __init__(self):

        self.graphs = {
            'p0': {
                'plot': None,
                'stateQualities': [0],
                'heuristics': [0.5]
            }
        }
        self.show = False
        if self.show:
            self.init_graphs()
    
    def init_graphs(self):
        self.fig, ((p0, _), (_, _)) = plt.subplots(2, 2)

        p0.set(title='Player 0 data', xlabel='Turn', ylabel='Quality')
        self.graphs['p0']['plot'] = p0

        plt.legend()
        plt.ion()
        plt.show()
        plt.draw()

    def keyboard_handler(self, event):
        if event.key == pygame.K_g:
            self.show = not self.show
            if self.show:
                self.init_graphs()
            else:
                plt.close()

            print(f"You press G key, show={self.show}")

    def init_event(self, manager):
        manager.register(pygame.KEYDOWN, self.keyboard_handler)

    def draw(self, **kwargs):

        self.save_datas(**kwargs)
        
        if self.show:
            self.display_graphs()

            plt.pause(0.05)

        print(f"MATHPLTLIB SHOW ?")

    def save_datas(self, ss_data: dict, ss_i: int, **kwargs):
        try:
            state_data = ss_data['mcts_state_data'][0]
        except:
            state_data = ss_data.get('mcts_state_data', None)

        if state_data:
            try:    
                s_n, s_v, (sa_n, sa_v) = state_data['Visits'], state_data['Rewards'], state_data['StateAction']
            except:
                s_n, s_v, (sa_n, sa_v) = state_data['visits'], state_data['rewards'], state_data['stateAction']

            p0_stateQuality = s_v / s_n
            if ss_i >= len(self.graphs['p0']['stateQualities']):
                self.graphs['p0']['stateQualities'].append(p0_stateQuality)
            else:
                self.graphs['p0']['stateQualities'][ss_i] = p0_stateQuality

        # else:
        #     print(f"No state_data:\n{ss_data}")

    def display_graphs(self):

        p0_plot = self.graphs['p0']['plot']
        p0_stateQualities = self.graphs['p0']['stateQualities']

        X = np.arange(len(p0_stateQualities))

        p0_plot.plot(
            X,
            p0_stateQualities,
            color='b',
            label='State qualities'
        )
