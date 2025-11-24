[uv](https://docs.astral.sh/uv/)

```bash
uv init # nie jestem pewny ale to pewnie wystarcza do instalacji libek a jak nie to
uv add -r requirements.txt
# test czy pytorch działa
uv run main.py
```

jak dodajecie jakieś zależności to
```bash
uv pip freeze -r > requirements.txt
```

dodatkowo formatter
```bash
uv run ruff check
uv run ruff format
```