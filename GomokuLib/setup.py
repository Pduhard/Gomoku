from setuptools import setup

setup(
    name='GomokuLib',
    version='0.1.0',
    author='Pduhard - Agiordan',
    description='',
    keywords='lib',
    packages=[
        'GomokuLib',
        'GomokuLib.AI',
        'GomokuLib.AI.Agent',
        'GomokuLib.AI.Model',
        'GomokuLib.AI.Dataset',
        'GomokuLib.Algo',
        'GomokuLib.Game',
        'GomokuLib.Game.Action',
        'GomokuLib.Game.GameEngine',
        'GomokuLib.Game.Rules',
        'GomokuLib.Game.State',
        'GomokuLib.Game.UI',
        'GomokuLib.Media',
        'GomokuLib.Player'
    ],

    # long_description=open('README.md').read(),
    install_requires=[
        # 'numpy==1.19.3',
        # 'numba==0.43.1'
    ]
)