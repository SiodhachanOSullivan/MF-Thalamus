

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
        