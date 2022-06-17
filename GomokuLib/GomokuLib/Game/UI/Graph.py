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
                'heuristics': [],
            },
            1: {
                'axe': None,
                'stateQualities': [],
                'heuristics': [],
            },
            2: {
                'axe': None,
                'depth': [
                    [],
                    []
                ]
            },
        }
        self.show = False
        if self.show:
            self.init_graphs()
    
    def init_graphs(self):
        self.fig, ((a0, a1), (a2, _)) = plt.subplots(2, 2, sharey='row')
        # self.fig, ((a0, a1), (_, _)) = plt.subplots(2, 2)

        a0.set(title='Player 0 data', xlabel='Turns', ylabel='Qualities')
        a1.set(title='Player 1 data', xlabel='Turns', ylabel='Qualities')
        a0.legend()
        a1.legend()

        self.graphs[0]['axe'] = a0
        self.graphs[1]['axe'] = a1
        self.graphs[2]['axe'] = a2

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
        try:
            del self.graphs[0]['stateQualities'][ss_i + 1:]
            del self.graphs[0]['heuristics'][ss_i + 1:]
            del self.graphs[1]['stateQualities'][ss_i + 1:]
            del self.graphs[1]['heuristics'][ss_i + 1:]
            del self.graphs[2]['depth'][0][ss_i + 1:]
            del self.graphs[2]['depth'][1][ss_i + 1:]
        except Exception as e:
            print("Graph: Unable to delete graph memory:\n\t", e)

    def save_datas(self, ss_data: dict, ss_i: int, **kwargs):
        try:
            if 'mcts_state_data' in ss_data:

                player_idx = ss_data.get('player_idx', 0)
                state_data = ss_data['mcts_state_data'][0]
                s_n, s_v, h, max_depth = state_data['visits'], state_data['rewards'], state_data['heuristic'], state_data['max_depth']

                if 'heuristic' in ss_data:  # MCTSNjit instance from UIManager has the priority
                    h = ss_data['heuristic']

                arr = [
                    (self.graphs[player_idx]['stateQualities'], s_v / s_n),
                    (self.graphs[player_idx]['heuristics'], h),
                    (self.graphs[2]['depth'][player_idx], max_depth)
                ]
                try:
                    for mem, data in arr:
                        if ss_i >= len(mem):
                            mem.append(data)
                        else:
                            mem[ss_i] = data
                        # print(mem)
                except Exception as e:
                    print("Graph: Unable to save data:\n\t", e)
                    return

        except Exception as e:
            print("Graph: Unable to fetch MCTS data:\n\t", e)

    def display_graphs(self):

        ## GAME graphs
        arr = [
            (0, self.graphs[0]),
            (1, self.graphs[1]),
        ]
        for player_idx, graph in arr:

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

        ## DEPTH graph
        axe = self.graphs[2]['axe']
        axe.clear()

        data_len = max(len(self.graphs[2]['depth'][0]), len(self.graphs[2]['depth'][1]))
        axe.plot(
            np.arange(data_len),
            [10] * data_len,
            color='darkgray',
            label=f'Depth 10'
        )

        arr = [
            (0, self.graphs[2]['depth'][0], 'darkred'),
            (1, self.graphs[2]['depth'][1], 'darkorange'),
        ]
        for player_idx, data, color in arr:
            axe.plot(
                np.arange(len(data)),
                data,
                color=color,
                label=f'Tree depth player {player_idx}'
            )
        axe.set_ylim(0, 15)
        axe.legend()

        plt.pause(0.005)
