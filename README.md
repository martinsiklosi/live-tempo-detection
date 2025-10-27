# Live Tempo Detection

Detect tempo and plot live. Intended to give non intrusive tempo feedback for drummers.

## Install

Requires Python $\ge$ 3.11.

To install the necessary python dependencies, run 

```bash
pip install -r requirements_core.txt
```

## Run

To start the live tempo detection, run the following command with your initial tempo.

```bash
python detect.py --initial 100
```

To see all availible options, run 

```bash
python detect.py --help
```

## Credit

Amazing tempo estimation algorithm developed by Robert Olsson Kihlborg.
