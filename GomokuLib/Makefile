.PHONY = help setup

.DEFAULT_GOAL = setup
UNAME := $(shell uname)

PYTHON=python

help:
	@echo ---------------HELP-----------------
	@echo To setup the project and build packages run make setup
	@echo ------------------------------------

setup:
	@echo $(PYTHON)
	@echo $(UNAME)
	$(PYTHON) setup.py build
	$(PYTHON) setup.py install

# run:
# 	@python Tests/train.py -n Tests/data.csv

# test:
# 	@python Tests/train.py -n Tests/data.csv

# test_conv:
# 	@python Tests/train_conv.py -n Tests/mnist_784.csv