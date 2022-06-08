import pygame
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


class Graph:
    """
        2 graph type:
            - Graphs over the game/turns. Use all snapshots data
                Save a snapshot id representing the beginning of graphs/game
            - Graphs about one turn/snapshot data (not yet)
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
            0: {
                'axe': None,
                'stateQualities': [],
                'heuristics': []
            },
            1: {
                'axe': None,
                'stateQualities': [],
                'heuristics': []
            }
        }
        self.show = False
        if self.show:
            self.init_graphs()
    
    def init_graphs(self):
        self.fig, ((a0, a1), (_, _)) = plt.subplots(2, 2, sharey='row')
        # self.fig, ((a0, a1), (_, _)) = plt.subplots(2, 2)

        a0.set(title='Player 0 data', xlabel='Turns', ylabel='Qualities')
        a1.set(title='Player 1 data', xlabel='Turns', ylabel='Qualities')
        a0.legend()
        a1.legend()

        self.graphs[0]['axe'] = a0
        self.graphs[1]['axe'] = a1

        plt.ion()   # Interactive mode, draw is now in non blocking mode
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

    def del_mem(self, ss_i: int):
        for _, p_mem in self.graphs.items():
            del p_mem['stateQualities'][ss_i + 1:]
            del p_mem['heuristics'][ss_i + 1:]

    def save_datas(self, ss_data: dict, ss_i: int, **kwargs):
        player_idx = ss_data.get('player_idx', 0)

        try:
            state_data = ss_data['mcts_state_data'][0]
        except:
            state_data = ss_data.get('mcts_state_data', None)

        if state_data:
            try:    
                s_n, s_v, (sa_n, sa_v) = state_data['Visits'], state_data['Rewards'], state_data['StateAction']
            except:
                s_n, s_v, (sa_n, sa_v) = state_data['visits'], state_data['rewards'], state_data['stateAction']

            arr = [
                (self.graphs[player_idx]['stateQualities'], s_v / s_n),
                (self.graphs[player_idx]['heuristics'], ss_data.get('heuristic', -42))
            ]
            for mem, data in arr:
                if ss_i >= len(mem):
                    mem.append(data)
                else:
                    mem[ss_i] = data

    def display_graphs(self):

        for player_idx, graph in self.graphs.items():

            graph['axe'].clear()
            graph['axe'].plot(
                np.arange(len(graph['stateQualities'])),
                graph['stateQualities'],
                color='b',
                label=f'StateQuality player {player_idx}'
            )

            graph['axe'].plot(
                np.arange(len(graph['heuristics'])),
                graph['heuristics'],
                color='g',
                label=f'Heuristic player {player_idx}'
            )
            graph['axe'].set_ylim(0, 1)
            graph['axe'].legend()

        plt.pause(0.005)