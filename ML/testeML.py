import numpy as np
import pickle
import torch
from PIL import Image

# =======================
# Charger le modèle pré-entraîné
# =======================
class MazeSolverNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MazeSolverNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialisation du modèle
input_size = 100
hidden_size = 256
output_size = 100
model = MazeSolverNet(input_size, hidden_size, output_size)





# Charger les poids du modèle à partir d'un fichier

model.load_state_dict(  # `load_state_dict` est une méthode de PyTorch pour charger les paramètres appris (poids) dans le modèle.
    
    torch.load("maze_model.pth")  # `torch.load` charge un fichier contenant les poids sauvegardés dans un dictionnaire.
    # "maze_model.pth" est le nom du fichier contenant les poids sauvegardés du modèle.
    # Ce fichier doit avoir été créé précédemment à l'aide de torch.save().
)

# Passer le modèle en mode évaluation

model.eval()  # `eval()` est une méthode de PyTorch qui change le mode du modèle en mode "évaluation".
# Cela désactive certains comportements spécifiques à l'entraînement :

# - Le Dropout : désactivé pour garantir des prédictions cohérentes.
# - La Normalisation par Batch : utilise les moyennes et variances calculées pendant l'entraînement.
# En mode évaluation, le modèle est prêt à effectuer des prédictions.

print("Modèle pré-entraîné chargé avec succès.")  # `print` est une fonction Python qui affiche du texte dans la console.




#	•	Dropout:
#	•	Dropout est une technique utilisée pendant l’entraînement d’un modèle pour éviter qu’il “mémorise trop bien” les données d’entraînement (ce qu’on appelle l’overfitting).
#	•	Elle fonctionne en “désactivant aléatoirement” une partie des neurones du réseau pendant chaque itération d’entraînement. 
#           Cela force le modèle à s’appuyer sur plusieurs chemins pour apprendre.
#	•	Pourquoi le désactiver en évaluation ?
#	•	Pendant l’évaluation (ou les prédictions), on veut des résultats stables et cohérents.
#	•	Si Dropout restait activé, une partie des neurones serait encore désactivée aléatoirement, ce qui rendrait les prédictions instables (elles changeraient à chaque exécution).
#	•	En désactivant Dropout, le modèle utilise tous ses neurones pour faire des prédictions, comme il est conçu pour le faire après l’entraînement.

# - La Normalisation par Batch :
#	•	Pendant l’entraînement, les valeurs qui passent dans le réseau de neurones (les activations) peuvent varier énormément.
#	•	La Normalisation par Batch (Batch Normalization ou BatchNorm) “stabilise” ces valeurs en les centrant autour de la moyenne et en ajustant leur écart-type (variance).
#	•	En gros, elle s’assure que les données sont “normalisées” pour chaque lot (batch) d’entraînement, ce qui aide le modèle à converger plus vite et à mieux apprendre.





# =======================
# Fonction pour prédire un chemin 
# =======================
def predict_path(model, maze):
    """
    Prédit le chemin pour un labyrinthe donné en utilisant un modèle pré-entraîné.

    Args:
        model: Modèle de réseau de neurones.
        maze: Labyrinthe 2D.

    Returns:
        predicted_path: Matrice 2D avec le chemin prédit (1 pour les cases du chemin, 0 ailleurs).
    """
    maze_np = np.array(maze)
    maze_flat = maze_np.flatten()
    maze_tensor = torch.tensor(maze_flat, dtype=torch.float32).unsqueeze(0)
    output = torch.sigmoid(model(maze_tensor)).detach().numpy().reshape(maze_np.shape)
    predicted_path = np.where(output > 0.5, 1, 0)
    return predicted_path

# =======================
# Fonction pour générer une image
# =======================
def generate_maze_image(maze, path=None, filename="maze_test.png"):
    """
    Génère une image du labyrinthe et du chemin prédit.

    Args:
        maze: Labyrinthe 2D (0 pour passage libre, 1 pour mur).
        path: Chemin prédit (liste de tuples (x, y)).
        filename: Nom du fichier de l'image générée.
    """
    h, w = len(maze), len(maze[0])
    img = Image.new("RGB", (w, h), "white")
    pixels = img.load()

    for i in range(h):
        for j in range(w):
            if maze[i][j] == 1:
                pixels[j, i] = (0, 0, 0)  # Mur en noir
            else:
                pixels[j, i] = (255, 255, 255)  # Passage libre en blanc

    if path:
        for px, py in path:
            pixels[py, px] = (255, 0, 0)  # Chemin en rouge

    img.save(filename)
    print(f"Image générée et sauvegardée sous '{filename}'.")

# =======================
# Charger un labyrinthe et tester
# =======================
# Charger le jeu de données
with open('maze_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Utiliser le premier labyrinthe pour tester
test_maze = dataset[0][0]

# Prédire le chemin
predicted_path = predict_path(model, test_maze)

# Convertir le chemin prédit en coordonnées pour l'affichage
path_coordinates = [(i, j) for i in range(len(predicted_path)) for j in range(len(predicted_path[0])) if predicted_path[i][j] == 1]

# Générer une image avec le chemin prédit
generate_maze_image(test_maze, path_coordinates, "maze_test.png")
print("Test terminé. Vérifiez l'image 'maze_test.png'.")