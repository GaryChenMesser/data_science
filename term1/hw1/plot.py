import matplotlib.pyplot as plot
import sys

with open(sys.argv[1], "r") as f:
    d = list(map(float, f.read().split("  ")))

plot.hist(d, bins=10)
plot.show()
