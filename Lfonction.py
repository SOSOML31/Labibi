def bin_to_graph(binary_map, width):
    """
    Convertit une carte binaire en une représentation de graphe.

    :param binary_map: Liste 1D représentant le labyrinthe (0 = libre, 1 = bloqué)
    :param width: Largeur du labyrinthe (pour déterminer la position 2D)
    :return: Liste 2D où chaque case est une liste de murs [haut, droite, bas, gauche]
    """
    height = len(binary_map) // width
    graph_map = []

    for row in range(height):
        for col in range(width):
            if binary_map[row * width + col] == 1:  # Case bloquée
                graph_map.append([1, 1, 1, 1])
            else:  # Case libre
                walls = [
                    1 if row == 0 or binary_map[(row - 1) * width + col] == 1 else 0,  # Mur haut
                    1 if col == width - 1 or binary_map[row * width + (col + 1)] == 1 else 0,  # Mur droit
                    1 if row == height - 1 or binary_map[(row + 1) * width + col] == 1 else 0,  # Mur bas
                    1 if col == 0 or binary_map[row * width + (col - 1)] == 1 else 0   # Mur gauche
                ]
                graph_map.append(walls)

    return graph_map


def graph_to_bin(graph_map, width):
    """
    Convertit une carte de graphe en une carte binaire.

    :param graph_map: Liste 2D où chaque case est une liste de murs [haut, droite, bas, gauche]
    :param width: Largeur du labyrinthe
    :return: Liste 1D représentant le labyrinthe (0 = libre, 1 = bloqué)
    """
    binary_map = []
    for i, cell in enumerate(graph_map):
        # Une case est bloquée si tous ses murs sont fermés
        is_blocked = all(cell)
        binary_map.append(1 if is_blocked else 0)

    return binary_map

def flat_to_2d(flat_lab, width):
    """
    Convertit un labyrinthe plat (1D) en une représentation 2D.

    :param flat_lab: Liste 1D représentant le labyrinthe
    :param width: Largeur du labyrinthe
    :return: Liste 2D représentant le labyrinthe
    """
    return [flat_lab[i:i + width] for i in range(0, len(flat_lab), width)]

def flat_it(lab_2d):
    """
    Convertit un labyrinthe 2D en une représentation plate (1D).

    :param lab_2d: Liste 2D représentant le labyrinthe
    :return: Liste 1D représentant le labyrinthe
    """
    return [cell for row in lab_2d for cell in row]



# Tests
if __name__ == "__main__":
    binary_map = [
        1, 1, 0, 1,
        0, 0, 0, 1,
        1, 0, 1, 1,
        1, 0, 0, 0
    ]
    width = 4

    expected_graph_map = [
        [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1],
        [1, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1],
        [1, 1, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1],
        [1, 1, 1, 1], [0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 1, 0]
    ]

    # Test bin_to_graph
    graph_map = bin_to_graph(binary_map, width)
    assert graph_map == expected_graph_map, f"Erreur dans bin_to_graph. Reçu: {graph_map}"

    # Test graph_to_bin
    converted_binary_map = graph_to_bin(graph_map, width)
    assert converted_binary_map == binary_map, f"Erreur dans graph_to_bin. Reçu: {converted_binary_map}"

    # Test flat_to_2d
    lab_2d = flat_to_2d(binary_map, width)
    expected_2d = [
        [1, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0]
    ]
    assert lab_2d == expected_2d, f"Erreur dans flat_to_2d. Reçu: {lab_2d}"

    # Test flat_it
    flat_lab = flat_it(lab_2d)
    assert flat_lab == binary_map, f"Erreur dans flat_it. Reçu: {flat_lab}"

    print("Tous les tests sont passés avec succès!")
