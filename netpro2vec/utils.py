# @Time    : 14/08/2020 00:48
# @Email   : ichcha.manipur@gmail.com
# @File    : utils.py
# file for utility functions

def nop(it, *a, **k):
    return it

def vprint(obj, verbose, end='\n'):
	if verbose:
		print(obj, end=end)
	return
