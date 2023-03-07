

def progressBar(iterable, prefix='', suffix='', decimals=1, length=50, fill='=', arrow='>', printEnd='\r'):
        total = len(iterable)
        def printProgressBar (iteration): # Progress Bar Printing Function
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * (filledLength-1) + arrow + '-' * (length - filledLength)
            print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
        printProgressBar(0) # Initial Call
        # Update Progress Bar
        for i, item in enumerate(iterable):
            yield item
            printProgressBar(i + 1)
        print() # Print New Line on Complete

import numpy as np
def double_gaussian(t, t0, T1, T2, amplitude):
    
    return amplitude*(\
                      np.exp(-(t-t0)**2/2./T1**2)*(t<t0)+\
                      np.exp(-(t-t0)**2/2./T2**2)*(t>t0))
        