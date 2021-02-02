from tqdm import tqdm

name = "execute_jobs"

def execute_jobs(repeated_jobs:list, verbose:bool=False): #counter:object, 
	aiqc.BaseModel._meta.database.close()
	aiqc.BaseModel._meta.database = aiqc.get_db()
	for j in tqdm(
		repeated_jobs
		, desc = "🔮 Training Models 🔮"
		, ncols = 100
	):
		job = j['job']
		repeat_index = j['repeat_index']
		job.run(verbose=verbose, repeat_index=repeat_index)
		#counter.value += 1