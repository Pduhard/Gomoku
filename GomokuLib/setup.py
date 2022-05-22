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
        'GomokuLib.Game.GameEngine',
        'GomokuLib.Game.Rules',
        # 'GomokuLib.Game.UI',
        'GomokuLib.Media',
        'GomokuLib.Player',
        'GomokuLib.Typing',
        'GomokuLib.Sockets',
    ],
    cffi_modules=[
        "./GomokuLib/Game/Rules/C/_build.py:ffibuilder",
        "./GomokuLib/Algo/C/_build.py:ffibuilder",
    ],

    # long_description=open('README.md').read(),
    install_requires=[
    ]
)