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
    #  'numpy<1.22,>=1.18',
    #   'numba>=0.55'
    #    'matplotlib',
    #    'cffi',
    #    'pygame==1.23.0'
    ]
)
