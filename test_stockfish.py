import subprocess

try:
    # Utilisez subprocess pour lancer Stockfish
    process = subprocess.run(["C:/Users/C_VANZETTA/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe"], check=True)
    print("Stockfish lancé avec succès.")
except Exception as e:
    print(f"Erreur lors du lancement de Stockfish : {e}")
