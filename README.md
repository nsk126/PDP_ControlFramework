# PDP Learning

## Virtual Environment (Optional)
To Run in a virual env, follow the instructions below. (_You need to have the python venv package installed_)

```bash
python -m venv .PDPvenv
.PDPvenv/Scripts/Activate
pip install -r requiments.txt
```

## Steps to run Pendulum model using PDP

```bash
python src/main.py -l <demos_file>
```

If no demo file is given, then demos are generated in the begining. _E.g:_ `python src/main.py`