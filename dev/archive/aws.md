Defaults at the top of config.py
```
# API URL format 'https://{api_id}.execute-api.{region}.amazonaws.com/{stage}/'.
aws_api_root = "https://qzdvq719pa.execute-api.us-east-1.amazonaws.com/Stage_AIQC/"
```

Config json
```
"aws_api_key": None
```

Queue attribute
```
aws_uid = CharField(null=True)
```

Queue methods
```

	def _aws_get_upload_url(id:int):
		"""
		- Fetch a temporary, presigned URL that can be used to upload db to S3.
		"""
		queue = Queue.get_by_id(id)
		api_key = _aws_get_api_key()
		
		upload_url_endpoint = f"{aws_api_root}queues/upload-url"

		response = requests_get(
			url = upload_url_endpoint
			, headers = {"x-api-key":api_key}
		)
		status_code = response.status_code
		if (status_code!=200):
			raise Exception(f"\n=> Yikes - failed to obtain upload url for AWS S3.\nHTTP Error Code = {status_code}.\n")

		# The uid can be used later for polling.
		uid = response.json()['uid']
		queue.aws_uid = uid
		queue.save()
		# The upload url is temporary so it doesn't make sense to save it.
		upload_url = response.json()['upload_url']
		s3_url = response.json()['s3_url']
		return upload_url, s3_url
	

	def _aws_upload(id:int, presigned_url:str):
		"""
		- Fetch a temporary, presigned URL that can be used to upload db to S3.
		"""
		queue = Queue.get_by_id(id)
		uid = queue.uid
		if (queue.aws_uid is None):
			raise Exception("\nYikes - This Queue has not been assigned an UID for AWS yet. Run `Queue._aws_get_upload_url()`.\n")

		# Regular utf-8 encoding of sqlite file does not parse. S3 picks up on "type=sqlite3"
		db_path = aiqc.get_path_db()
		data = open(db_path, encoding='latin-1').read()
		response = requests.put(presigned_url, data=data)
		status_code = response.status_code
		if (status_code==200):
			print("\n=> Success - project database was successfully uploaded to AWS S3.\n")
		else:
			raise Exception(f"\n=> Yikes - failed upload project databse to AWS S3.\nHTTP Error Code = {status_code}.\n")

```
