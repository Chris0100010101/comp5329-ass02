import importlib
pkgs = {
    'langchain_openai': 'langchain-openai',
    'langchain_community': 'langchain-community',
    'langchain_core': 'langchain-core',
    'faiss': 'faiss-cpu',
    'requests': 'requests',
    'dotenv': 'python-dotenv',
    'pandas': 'pandas',
    'openai': 'openai',
}
missing = []
for mod, pkg in pkgs.items():
    try:
        importlib.import_module(mod)
        print('OK      ' + pkg)
    except ImportError:
        print('MISSING ' + pkg)
        missing.append(pkg)
if missing:
    print()
    print('pip install ' + ' '.join(missing))
else:
    print()
    print('All dependencies installed.')
