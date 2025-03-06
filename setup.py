from setuptools import setup
from pathlib import Path


# read in requirements. this gets used in the setup function
requirements_dir = f'{Path(__file__).parent}/requirements.txt'
with open(requirements_dir, 'r') as f:
	requirements = f.read().splitlines()

setup(
	name='llm-eval',
	version='0.0.1',
	py_modules=['llm-eval'],
	install_requires=requirements,
	entry_points={
		'console_scripts' : [
			'llm_eval = src:main',
		],
	}
)
