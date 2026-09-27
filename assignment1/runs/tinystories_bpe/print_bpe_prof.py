import pstats
from pstats import SortKey

path = r"D:\desktop\data\tinystories\tokenizer_train_10k.prof"
stats = pstats.Stats(path)
print("total_tt", round(stats.total_tt, 3))
print("--- tottime ---")
stats.sort_stats(SortKey.TIME).print_stats(25)
print("--- cumulative ---")
stats.sort_stats(SortKey.CUMULATIVE).print_stats(25)
