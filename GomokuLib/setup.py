from setuptools import setup

setup(
    name='GomokuLib',
    version='1.0.0',
    author='Pduhard - Agiordan',
    description='',
    keywords='lib',
    packages=[
        'GomokuLib',
        'GomokuLib.Algo',
        'GomokuLib.Game',
        'GomokuLib.Game.GameEngine',
        'GomokuLib.Game.Rules',
        'GomokuLib.Game.UI',
        'GomokuLib.Media',
        'GomokuLib.Player',
        'GomokuLib.Typing',
        'GomokuLib.Sockets',
    ],
    cffi_modules=[
        "./GomokuLib/Game/Rules/C/_build.py:ffibuilder",
        "./GomokuLib/Algo/C/_build.py:ffibuilder",
    ],
    package_data={'GomokuLib': ['GomokuLib/Media/Images/*']},
    include_package_data=True,
    install_requires=[
        'scipy>=0.8',
        'numpy>=1.18,<1.23'
        'numba>=0.55',
        'matplotlib',
        'cffi',
        'pygame'
    ]
)
