from datasets import load_dataset

ds = load_dataset("vCache/SemBenchmarkLmArena")

print(ds)
print(ds["train"][0])

ds.save_to_disk("SemBenchmarkLmArena")