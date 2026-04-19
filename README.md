# oral-mytho

`oral-mytho` is a local app for:

- importing a story CSV,
- building pattern and keyword search artefacts,
- managing entries,
- exploring stories on a similarity map.

The app runs in your web browser, but it is launched from your computer with a few simple commands.

## Before You Start

You need 2 things installed on your computer:

1. `Git`
2. `Python 3`

If you do not have them yet:

- Git: [https://git-scm.com/downloads](https://git-scm.com/downloads)
- Python: [https://www.python.org/downloads/](https://www.python.org/downloads/)

Windows users: during Python installation, make sure the installer option `Add Python to PATH` is checked.

An internet connection is recommended the first time you launch the app, because the embedding model may need to download once.

Repository URL:

```text
https://github.com/alterfero/oral-mytho.git
```

## 1. Download The Project

Open a terminal:

- Mac: open `Terminal`
- Linux: open `Terminal`
- Windows: open `PowerShell`

Then copy and run:

```bash
git clone https://github.com/alterfero/oral-mytho.git
cd oral-mytho
```

## 2. Install And Launch

### Mac

Run these commands in `Terminal`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

### Linux

Run these commands in `Terminal`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

### Windows

Run these commands in `PowerShell`:

```powershell
py -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

## 3. Open The App

After the last command, Streamlit starts a local server and prints a local address in the terminal.

In most cases, the app opens automatically in your browser.

If it does not, open this address manually:

```text
http://localhost:8501
```

## 4. Use The App

The app has 3 main sections:

1. `Data processing`
   Import a CSV and build the JSONL dataset plus semantic search artefacts.
2. `Data management`
   Add, edit, and delete entries.
3. `Exploration`
   Search for a pattern and inspect related stories on the map.

If you want to erase all generated artefacts from your machine, use the `Clean session` button on the `Data processing` page.

## 5. Next Time You Want To Open It

You do not need to clone the repository again.

Just open a terminal, go into the project folder, and launch the app again.

### Mac / Linux

```bash
cd oral-mytho
.venv/bin/streamlit run app.py
```

### Windows

```powershell
cd oral-mytho
.\.venv\Scripts\streamlit.exe run app.py
```

## 6. Stop The App

To stop the software, go back to the terminal window and press:

```text
Ctrl + C
```

## Troubleshooting

### `git` is not recognized

Git is not installed, or the computer needs to be restarted after installation.

### `python3` or `py` is not recognized

Python is not installed correctly, or it was not added to your system path.

### The browser does not open automatically

Open [http://localhost:8501](http://localhost:8501) manually.

### The first installation takes a long time

This is normal. Python packages and the embedding model can take a few minutes to install/download the first time.
