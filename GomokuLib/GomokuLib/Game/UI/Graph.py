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
            0: {
                'plot': None,
                'stateQualities': [0.5],
                'heuristics': [0.5]
            },
            1: {
                'plot': None,
                'stateQualities': [0.5],
                'heuristics': [0.5]
            }
        }
        self.graph_uptodate = True
        self.show = False
        if self.show:
            self.init_graphs()
    
    def init_graphs(self):
        # self.fig, ((p0, p1), (_, _)) = plt.subplots(2, 2, sharey='row')
        self.fig, ((p0, p1), (_, _)) = plt.subplots(2, 2)

        p0.set(title='Player 0 data', xlabel='Turns', ylabel='Qualities', ylim=(0, 1))
        p1.set(title='Player 1 data', xlabel='Turns', ylabel='Qualities', ylim=(0, 1))
        p0.legend()
        p1.legend()

        self.graphs[0]['plot'] = p0
        self.graphs[1]['plot'] = p1

        self.display_graphs()
        plt.ion()
        # plt.show()
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

            plt.pause(0.01)

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

            # print(f"Graph: new data", ss_i)
            arr = [
                (self.graphs[player_idx]['stateQualities'], s_v / s_n),
                (self.graphs[player_idx]['heuristics'], ss_data.get('heuristic', -42))
            ]
            for mem, data in arr:
                if ss_i >= len(mem):
                    mem.append(data)
                else:
                    mem[ss_i] = data
                    del mem[ss_i + 1:]
                    self.graph_uptodate = False

        # else:
        #     print(f"No state_data:\n{ss_data}")

    def display_graphs(self):

        if not self.graph_uptodate:
            plt.close()
            self.init_graphs()

        for player_idx, graph in self.graphs.items():

            graph['plot'].plot(
                np.arange(len(graph['stateQualities'])),
                graph['stateQualities'],
                color='b',
                label=f'StateQuality player {player_idx}'
            )

            graph['plot'].plot(
                np.arange(len(graph['heuristics'])),
                graph['heuristics'],
                color='g',
                label=f'Heuristic player {player_idx}'
            )
