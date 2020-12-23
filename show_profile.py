import pstats


name = '3.profile'
out = pstats.Stats(name)

print('sort by cumulatvie (the function itself plus all child functions called inside)')  # noqa
out.sort_stats('cumulative').print_stats(20)

print('sort by total time (only the function itself not its childrens)')
out.sort_stats('time').print_stats(20)
