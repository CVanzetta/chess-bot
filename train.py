import numpy as np
from chess_bot_model import create_model

# Exemple de données : X sont les états de l'échiquier et y sont les coups à jouer
X_train = np.random.rand(1000, 64)  # 1000 positions d'échiquier aléatoires
y_train = np.random.rand(1000, 64)  # 1000 coups cibles aléatoires (pour l'exemple)

model = create_model()
model.fit(X_train, y_train, epochs=10)  # Entraîne le modèle
model.save('chess_bot_model.h5')  # Sauvegarde du modèle
