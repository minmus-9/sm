all:	

clean:	
	find . -type f -name '*.py[co]'    -print0 | xargs -0 -n 25 rm -f  || true
	find . -type d -name '__pycache__' -print0 | xargs -0 -n 25 rm -rf || true
