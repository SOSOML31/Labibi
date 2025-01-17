from PIL import Image
import random

# Conversion binary_map → graph_map

def bin_to_graph(binary_map, w):
    """
    Convertit une carte binaire en représentation graphe.
    Chaque case est représentée par une liste de 4 valeurs (murs haut, droite, bas, gauche).
    1 = mur, 0 = passage libre.
    """
    h = len(binary_map) // w
    graph_map = []
    for i in range(h):
        row = []
        for j in range(w):
            cell = [1, 1, 1, 1]  
            if binary_map[i * w + j] == 0:  #  libre
                if i > 0 and binary_map[(i - 1) * w + j] == 0:  # Mur haut
                    cell[0] = 0
                if j < w - 1 and binary_map[i * w + (j + 1)] == 0:  # Mur droite
                    cell[1] = 0
                if i < h - 1 and binary_map[(i + 1) * w + j] == 0:  # Mur bas
                    cell[2] = 0
                if j > 0 and binary_map[i * w + (j - 1)] == 0:  # Mur gauche
                    cell[3] = 0
            row.append(cell)
        graph_map.append(row)
    return graph_map

# Conversion graph_map → binary_map

def graph_to_bin(graph_map):
    """
    Convertit une carte graphe en carte binaire.
    1 = mur, 0 = passage libre.
    """
    binary_map = []
    for row in graph_map:
        for cell in row:
            binary_map.append(0 if sum(cell) < 4 else 1)
    return binary_map


# Conversion flat → 2D

def flat_to_2d(flat_lab, w):
    """
    Transforme un labyrinthe plat (liste 1D) en format 2D (liste de listes).
    """
    return [flat_lab[i:i + w] for i in range(0, len(flat_lab), w)]


# Conversion 2D → plat

def flat_it(lab_2d):
    """
    Transforme un labyrinthe en 2D (liste de listes) en format plat (liste 1D).
    """
    return [cell for row in lab_2d for cell in row]


# Affichage du labyrinthe

def print_maze(lab, in_out=None, path=None):
    """
    Affiche le labyrinthe dans la console.
    '█' représente les murs, '░' représente les cases libres.
    Si in_out est fourni, 'i' représente l'entrée et 'o' la sortie.
    Si path est fourni, les cases du chemin sont marquées par '.'.
    """
    for i, row in enumerate(lab):
        line = ""
        for j, cell in enumerate(row):
            if in_out and (i, j) == in_out[0]:  # Entrée
                line += "i"
            elif in_out and (i, j) == in_out[1]:  # Sortie
                line += "o"
            elif path and (i, j) in path:  # Chemin
                line += "."
            else:
                line += "░" if cell == 0 else "█"
        print(line)


# Génération d'un labyrinthe avec Prim

def generate_prim(w, h):
    """
    Génère un labyrinthe en utilisant l'algorithme de Prim.
    Retourne un labyrinthe sous forme de liste 2D avec des passages (0) et des murs (1).
    """
    maze = [[1] * w for _ in range(h)]  # Initialisation avec des murs partout
    start_x, start_y = random.randint(0, h - 1), random.randint(0, w - 1)
    maze[start_x][start_y] = 0  # Point de départ
    walls = [(start_x + dx, start_y + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)] if 0 <= start_x + dx < h and 0 <= start_y + dy < w]

    while walls:
        wx, wy = walls.pop(random.randint(0, len(walls) - 1))
        if maze[wx][wy] == 1:  # Si c'est un mur, on le transforme en passage
            neighbors = [(wx + dx, wy + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)] if 0 <= wx + dx < h and 0 <= wy + dy < w and maze[wx + dx][wy + dy] == 0]
            if len(neighbors) == 1:  # On s'assure qu'il ne connecte qu'à un seul passage
                maze[wx][wy] = 0
                walls.extend([(wx + dx, wy + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)] if 0 <= wx + dx < h and 0 <= wy + dy < w and maze[wx + dx][wy + dy] == 1])

    return maze

# =======================
# Génération d'une image PNG
# =======================
def fancy_maze(bin_lab_2d, name):
    """
    Génère une image PNG représentant le labyrinthe.
    Les murs sont noirs (0), les cases libres blanches (1).
    """
    h, w = len(bin_lab_2d), len(bin_lab_2d[0])
    img = Image.new("1", (w, h))
    pixels = img.load()
    for i in range(h):
        for j in range(w):
            pixels[j, i] = bin_lab_2d[i][j]
    img.save(f"{name}.png")

# =======================
# Lecture d'une image PNG
# =======================
def unmaze(file_name):
    """
    Lit une image PNG représentant un labyrinthe et retourne une carte binaire.
    """
    img = Image.open(file_name).convert("1")
    binary_map = []
    for y in range(img.height):
        for x in range(img.width):
            binary_map.append(img.getpixel((x, y)))
    return {"maze": binary_map}

# =======================
# Génération de l'entrée et de la sortie
# =======================
def get_random_in_out(lab_bin_2d):
    """
    Génère des coordonnées aléatoires pour l'entrée et la sortie.
    Les deux points doivent être différents et sur des cases libres.
    """
    h, w = len(lab_bin_2d), len(lab_bin_2d[0])
    free_cells = [(i, j) for i in range(h) for j in range(w) if lab_bin_2d[i][j] == 0]
    in_out = random.sample(free_cells, 2)  # Choisit 2 cases libres au hasard
    return in_out

# =======================
# Tests pour valider les fonctions
# =======================
def test_maze_utils():
    """
    Tests pour valider les conversions et affichages.
    """
    # Exemple de labyrinthe plat binaire
    flat_lab = [1, 1, 0, 1, 
                0, 0, 0, 1, 
                1, 0, 1, 1, 
                1, 0, 0, 0]
    w = 4
    
    # Conversion plat → 2D
    lab_2d = flat_to_2d(flat_lab, w)
    print("Labyrinthe 2D :")
    print_maze(lab_2d)
    
    # Conversion 2D → plat
    flat_back = flat_it(lab_2d)
    assert flat_back == flat_lab, "Erreur dans la conversion 2D → plat"
    
    # Conversion binaire → graphe
    graph_map = bin_to_graph(flat_lab, w)
    print("\nGraphe :")
    for row in graph_map:
        print(row)
    
    # Conversion graphe → binaire
    binary_back = graph_to_bin(graph_map)
    assert binary_back == flat_lab, "Erreur dans la conversion graphe → binaire"

    # Génération d'un labyrinthe
    print("\nLabyrinthe généré avec Prim :")
    prim_maze = generate_prim(10, 10)
    print_maze(prim_maze)

    # Sauvegarde en image
    fancy_maze(prim_maze, "generated_maze")
    print("\nImage sauvegardée sous 'generated_maze.png'")

# =======================
# Test avec entrée et sortie
# =======================
def test_with_in_out():
    """
    Test pour afficher le labyrinthe avec une entrée et une sortie.
    """
    # Génération d'un labyrinthe
    prim_maze = generate_prim(10, 10)
    in_out = get_random_in_out(prim_maze)  # Entrée et sortie
    print("\nLabyrinthe avec entrée et sortie :")
    print_maze(prim_maze, in_out)
    return prim_maze, in_out

# Lancer les tests
if __name__ == "__main__":
    test_maze_utils()
    test_with_in_out()