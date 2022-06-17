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
