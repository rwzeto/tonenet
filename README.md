Minimal fork/wrapper around https://github.com/saber5433/ToneNet to get it running.

To install: 

`git clone git@github.com:rwzeto/tonenet.git`

Then in the project directory:

`sh setup.sh`

Then activate the environment:

`source .venv/bin/activate`

Then run the example:

`python example.py`

Example output (user speaks "你"): 

```
❯ python example.py
Press ⏎ to start recording …
Recording — press ⏎ again to stop.

Captured 1.30 s
Detected tone → 3
```