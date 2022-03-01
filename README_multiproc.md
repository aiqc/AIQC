https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
- 'fork' makes all variables on main process available to child process. OS attempts not to duplicate all variables.
- 'spawn' requires that variables be passed to child as args, and seems to play by pickle's rules (e.g. no func in func).

- In Python 3.8, macOS changed default from 'fork' to 'spawn' , which is how I learned all this.
- Windows does not support 'fork'. It supports 'spawn'. So basically I have to play by spawn/ pickle rules.
- Spawn/ pickle dictates (1) where execute_jobs func is placed, (2) if MetricsCutoff func works, (3) if tqdm output is visible.
- Update: now MetricsCutoff is not working in `fork` mode. probably pickling.
- Wrote the `poll_progress` func for 'spawn' situations.
- Tried `concurrent.futures` but it only works with `.py` from command line.

```python
if (os.name != 'nt'):
	# If `force=False`, then `importlib.reload(aiqc)` triggers `RuntimeError: context already set`.
	multiprocessing.set_start_method('fork', force=True)
```

	"""
	# This is related to background processing. After I decoupled Jobs, I never reenabled background processing.
	def poll_statuses(id:int, as_pandas:bool=False):
		queue = Queue.get_by_id(id)
		repeat_count = queue.repeat_count
		statuses = []
		for i in range(repeat_count):
			for j in queue.jobs:
				# Check if there is a Predictor with a matching repeat_index
				matching_predictor = Predictor.select().join(Job).join(Queue).where(
					Queue.id==queue.id, Job.id==j.id, Predictor.repeat_index==i
				)
				if (len(matching_predictor) == 1):
					r_id = matching_predictor[0].id
				elif (len(matching_predictor) == 0):
					r_id = None
				job_dct = {"job_id":j.id, "repeat_index":i, "predictor_id": r_id}
				statuses.append(job_dct)

		if (as_pandas==True):
			df = pd.DataFrame.from_records(statuses, columns=['job_id', 'repeat_index', 'predictor_id'])
			return df.round()
		elif (as_pandas==False):
			return statuses

	
	# This is related to background processing. After I decoupled Jobs, I never reenabled background processing.
	def poll_progress(id:int, raw:bool=False, loop:bool=False, loop_delay:int=3):
		# - For background_process execution where progress bar not visible.
		# - Could also be used for cloud jobs though.
		if (loop==False):
			statuses = Queue.poll_statuses(id)
			total = len(statuses)
			done_count = len([s for s in statuses if s['predictor_id'] is not None]) 
			percent_done = done_count / total

			if (raw==True):
				return percent_done
			elif (raw==False):
				done_pt05 = round(round(percent_done / 0.05) * 0.05, -int(math.floor(math.log10(0.05))))
				bars_filled = int(done_pt05 * 20)
				bars_blank = 20 - bars_filled
				meter = '|'
				for i in range(bars_filled):
					meter += '██'
				for i in range(bars_blank):
					meter += '--'
				meter += '|'
				print(f"🔮 Training Models 🔮 {meter} {done_count}/{total} : {int(percent_done*100)}%")
		elif (loop==True):
			while (loop==True):
				statuses = Queue.poll_statuses(id)
				total = len(statuses)
				done_count = len([s for s in statuses if s['predictor_id'] is not None]) 
				percent_done = done_count / total
				if (raw==True):
					return percent_done
				elif (raw==False):
					done_pt05 = round(round(percent_done / 0.05) * 0.05, -int(math.floor(math.log10(0.05))))
					bars_filled = int(done_pt05 * 20)
					bars_blank = 20 - bars_filled
					meter = '|'
					for i in range(bars_filled):
						meter += '██'
					for i in range(bars_blank):
						meter += '--'
					meter += '|'
					print(f"🔮 Training Models 🔮 {meter} {done_count}/{total} : {int(percent_done*100)}%", end='\r')
					#print()

				if (done_count == total):
					loop = False
					os.system("say Model training completed")
					break
				time.sleep(loop_delay)
	"""


	"""
	def stop_jobs(id:int):
		# This is related to background processing. After I decoupled Jobs, I never reenabled background processing.
		# SQLite is ACID (D = Durable). If transaction is interrupted mid-write, then it is rolled back.
		queue = Queue.get_by_id(id)
		
		proc_name = f"aiqc_queue_{queue.id}"
		current_procs = [p.name for p in multiprocessing.active_children()]
		if (proc_name not in current_procs):
			raise ValueError(f"\nYikes - Cannot terminate `multiprocessing.Process.name` '{proc_name}' because it is not running.\n")

		processes = multiprocessing.active_children()
		for p in processes:
			if (p.name == proc_name):
				try:
					p.terminate()
				except:
					raise Exception(f"\nYikes - Failed to terminate `multiprocessing.Process` '{proc_name}.'\n")
				else:
					print(f"\nKilled `multiprocessing.Process` '{proc_name}' spawned from aiqc.Queue <id:{queue.id}>\n")
	"""

	"""
	# This is related to background processing. After I decoupled Jobs, I never reenabled background processing.
	def execute_jobs(job_statuses:list, verbose:bool=False):  
		# - This needs to be a top level function, otherwise you get pickle attribute error.
		# - Alternatively, you can put this is a separate submodule file, and call it via
		#   `import aiqc.execute_jobs.execute_jobs`
		# - Tried `mp.Manager` and `mp.Value` for shared variable for progress, but gave up after
		#   a full day of troubleshooting.
		# - Also you have to get a separate database connection for the separate process.
		BaseModel._meta.database.close()
		BaseModel._meta.database = get_db()
		for j in tqdm(
			job_statuses
			, desc = "🔮 Training Models 🔮"
			, ncols = 100
		):
			if (j['predictor_id'] is None):
				Job.run(id=j['job_id'], verbose=verbose, repeat_index=j['repeat_index'])
	"""