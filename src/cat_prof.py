import pstats
import sys

args = sys.argv
sts = pstats.Stats(args[1])
sts.strip_dirs().sort_stats(-1).print_stats()