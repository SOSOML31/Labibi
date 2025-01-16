from pprint import pprint
from random import randint, choice, shuffle
from time import sleep


def make_mat(n, m, v=0):
    """Pour créer une matrice n*m avec des v (par défaut 0) dans toutes les cases"""
    return [[v for _ in range(n)] for __ in range(m)]


def print_maze(m, in_out=None, path=None):
    """Pour l'affichage dans la console"""
    s = ""
    for x, l in enumerate(m):
        for y, c in enumerate(l):
            if in_out:
                if in_out[0] == (x, y):
                    s += "✅"
                    continue
                if in_out[1] == (x, y):
                    s += "❎"
                    continue
            if path:
                if (x, y) in path:
                    s += "⭕"
                    continue
            if c == 1:
                s += "⬛"
            elif c == 0:
                s += "⬜"
        s += "\n"
    print(s)


def get_voisines(lab, cell):
    """Une fonction qui récupère les voisines (qui ne sortent pas du lab)"""
    voisines = [
        (cell[0], cell[1] - 1),  # top
        (cell[0] + 1, cell[1]),  # left
        (cell[0], cell[1] + 1),  # bot
        (cell[0] - 1, cell[1]),  # right
    ]
    # virer celles qui sont hors matrice
    valid_voisines = []
    for c in voisines:
        # si elle est hors tableau je l'enlève
        if (
                0 > c[0] or
                c[0] > (len(lab) - 1) or
                0 > c[1] or
                c[1] > (len(lab[0]) - 1)
        ):
            continue
        valid_voisines.append(c)
    return valid_voisines


def valid_cells(lab, cells):
    """Validation pour la génération"""
    # Filtrer les voisines
    ## pour chaque voisine
    valides = []
    for c in cells:
        # si elle est déjà à 0 je l'enlève (je passe à la suivante)
        if lab[c[0]][c[1]] == 0:
            continue
        else:
            # si elle à déjà exactement 1 "entrée" je la valide
            ## je récupère ses propres voisines
            voisines_de_la_voisine = get_voisines(lab, c)
            ## je fais la somme de leur valeur
            somme = sum([lab[vv[0]][vv[1]] for vv in voisines_de_la_voisine])
            ## si c'est exactement le nombre de voisine - 1 (il y a 1 entrée)
            if somme == len(voisines_de_la_voisine) - 1:
                valides.append(c)
    return valides


def gen_maze(m, n, slowmo=False, poper=False):  # genre prim's
    # lab avec que des 1
    # une list avec un 1 pour chaque n... pour chaque m... dans une list
    lab = make_mat(m, n, 1)
    # une case au hasard (un tuple (x, y))
    # (comme les indices commencent à 0 il faut aller jusu'a n - 1 parce que la limite haute est incluse)
    cell = (randint(0, n - 1), randint(0, m - 1))
    # je fais un trou
    lab[cell[0]][cell[1]] = 0
    # liste des "frontières"
    walls = get_voisines(lab, cell)
    # tant qu'il y a des "frontières" à visiter
    i = 0
    while len(walls):
        i += 1
        if poper:
            cell = walls.pop()
        else:
            # j'en choisi une au hasard
            cell = choice(walls)
            # je l'enlève des frontières à visiter
            walls.remove(cell)
        # je chope ses voisines
        nb = get_voisines(lab, cell)
        # je fais la somme pour pouvoir vérifier qu'il n'y a bien qu'un seul "trou" à coté
        k = sum([lab[o[0]][o[1]] for o in nb])
        if k == len(nb) - 1:
            # je fais un trou dedans
            lab[cell[0]][cell[1]] = 0
            if poper:
                shuffle(nnb := list(filter(lambda c: lab[c[0]][c[1]], nb)))
                walls += nnb
            else:
                # j'ajoute ses voisines à la "frontières"
                walls += list(filter(lambda c: lab[c[0]][c[1]], nb))
            if slowmo:
                print("\033c")
                print("Génération")
                print_maze(lab)
                print("Iterations : ", i)
                sleep(0.02)
    return lab


def chemin(maze, in_out):
    """Trouver le chemin"""
    # je mets l'entrée dans ma liste de case à explorer
    ###  Format (case, precedente, distance)
    explore = [(in_out[0], None, 0)]
    '''
        Ici c'est pour les lab ou il n'y a qu'un seul chemin
        donc je n'ai pas besoin de la distance
        si j'ai un lab avec plusieurs chemin je vais en trouver un, mais pas
        forcement le plus court...
        Comment je pourrais faire pour être SUR d'avoir le plus court ?
    '''
    # liste pour mon chemin (pour les cases avec les distances)
    path_list = []
    # liste des cases déjà visitées par ce que c'est pratique de l'avoir même si
    # elles sont dans path_list
    visited = []
    # au début, je n'ai pas de case precedente (celle d'où je viens)
    prev_cell = None
    while True:
        # je prends une case à explorer (et sa précédente)
        cell, prev, dist = explore.pop()
        # je récupère ses voisines
        voisines = get_voisines(maze, cell)
        # j'enlève les murs des voisines (voir la doc de filter !)
        voisines = list(filter(lambda c: maze[c[0]][c[1]] == 0, voisines))
        # j'enlève les cases que j'ai déjà explorées
        voisines = list(filter(lambda c: c not in visited, voisines))
        # j'ajoute les voisines (avec la précédente, c'est-à-dire cell)
        # dans la liste des cases à explorer
        explore += [(c, cell, dist + 1) for c in voisines]
        # j'ajoute ma cellule à la liste des cases visitées
        visited.append(cell)
        # J'ajoute les infos de la cell actuelle à path_list
        path_list.append((cell, prev, dist + 1))
        # si j'ai trouvé la sortie j'arrête
        if cell == in_out[1]:
            sortie = (cell, prev, dist + 1)
            break
        # si je n'ai plus rien à explorer c'est que la sortie n'est pas atteignable
        if len(explore) == 0:
            print(cell, "→", in_out[1])
            pprint([c[0] for c in path_list])
            print_maze(maze)
            print_maze(maze, in_out, [c[0] for c in path_list])
            raise Exception("Impossible de trouver la sortie !!")

    # magie magie !
    # je trouve le plus court chemin
    # je mets la sortie dans le plus court chemin
    shortest = [sortie]
    # tant que je n'ai pas rejoin l'entrée
    while True:
        # je trouve la case précédente
        precedente = list(filter(lambda c: c[0] == shortest[-1][1], path_list))[0]
        # je l'ajoute à mon chemin le plus court
        shortest.append(precedente)
        # quand je suis au bout j'arrête
        if precedente[0] == in_out[0]:
            break
    # je retourne mon chemin (juste les cases et remis dans l'ordre)
    rep = [c[0] for c in shortest]
    rep.reverse()
    return rep


# TEST #
if __name__ == "__main__":
    H = 20
    L = 25
    m = gen_maze(L, H, slowmo=True, poper=True)
    in_out = (0, 0), (H-1, L-1)
    # pour être sûr que in et out ne sont pas des murs, je fais des trous
    # mais c'est quand même possible que j'ai la sortie ou l'entrée
    # entourée de murs  !!!
    m[in_out[0][0]][in_out[0][1]] = 0
    m[in_out[1][0]][in_out[1][1]] = 0
    print("In et out")
    print_maze(m, in_out)
    path = chemin(m, in_out)
    print("Chemin")
    print_maze(m, in_out, path)