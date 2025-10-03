

"""
Exercice : Calcul de distances euclidiennes et k-plus proches voisins
======================================================================
Cet exercice démontre l'utilisation du broadcasting NumPy pour calculer
efficacement une matrice de distances entre points.

"""
#%%
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("ÉTAPE 1 : Création de la matrice X (10 lignes × 2 colonnes)")
print("="*70)

# Créer une matrice 10x2 avec des nombres aléatoires
rng = np.random.default_rng(seed=12345)
X = rng.uniform(size=(10, 2))

print(f"\nMatrice X de dimension {X.shape}:")
print(X)
print(f"\nX contient {X.shape[0]} points en dimension {X.shape[1]}")










#%%
print("\n" + "="*70)
print("ÉTAPE 2 : Visualisation avec matplotlib")
print("="*70)

# Créer le nuage de points
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=100, c='blue', alpha=0.6, edgecolors='black')

# Annoter chaque point avec son indice
for i in range(len(X)):
    plt.annotate(f'P{i}', (X[i, 0], X[i, 1]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.xlabel('Coordonnée x')
plt.ylabel('Coordonnée y')
plt.title('Nuage de points (10 points en 2D)')
plt.grid(True, alpha=0.3)
plt.savefig('nuage_points.png', dpi=150, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : nuage_points.png")








#%%
print("\n" + "="*70)
print("ÉTAPE 3 : Calcul de la matrice de distances euclidiennes")
print("="*70)

print("\n--- 3.1 : Création de X1 avec np.newaxis ---")
# X1 aura la forme (10, 1, 2) : chaque point est maintenant dans sa propre "tranche"
X1 = X[:, np.newaxis, :]
print(f"\nDimension de X : {X.shape}")
print(f"Dimension de X1 : {X1.shape}")
print("\nExplication :")
print("  - X[:, np.newaxis, :] insère une nouvelle dimension à l'indice 1")
print("  - Chaque ligne de X devient une 'tranche' séparée")
print(f"\nPremier élément de X1 (point 0):")
print(X1[0])
#%%
print("\n--- 3.2 : Création de X2 avec dimension (1, 10, 2) ---")
# X2 aura la forme (1, 10, 2) : tous les points dans une seule "tranche"
X2 = X[np.newaxis, :, :]
print(f"\nDimension de X2 : {X2.shape}")
print("\nExplication :")
print("  - X[np.newaxis, :, :] insère une nouvelle dimension à l'indice 0")
print("  - Tous les points sont dans une seule 'tranche'")
#%%
print("\n--- 3.3 : Calcul des différences et élévation au carré ---")
print("\nMaintenant, grâce au broadcasting NumPy :")
print(f"  - X1 de shape {X1.shape} ")
print(f"  - X2 de shape {X2.shape} ")
print(f"  - X1 - X2 donnera un tableau de shape (10, 10, 2)")
#%%
# Calcul des différences pour chaque coordonnée
differences = X1 - X2
print(f"\nShape de (X1 - X2) : {differences.shape}")
print("\nExplication du broadcasting :")
print("  - Pour chaque point i dans X1 (dimension 0)")
print("  - On calcule la différence avec chaque point j dans X2 (dimension 1)")
print("  - Pour chaque coordonnée (dimension 2)")
print(f"  - Résultat : différences[i, j, k] = X[i, k] - X[j, k]")
#%%
# Élever au carré
differences_squared = differences ** 2
print(f"\nShape de (X1 - X2)² : {differences_squared.shape}")

print("\n--- 3.4 : Sommation sur le dernier axe ---")
print("\nOn somme les carrés des différences pour chaque paire de points :")
print("  distance²[i, j] = (X[i, 0] - X[j, 0])² + (X[i, 1] - X[j, 1])²")

# Somme sur le dernier axe (axis=2 ou axis=-1)
sum_squared = np.sum(differences_squared, axis=-1)
print(f"\nShape après sommation sur axis=-1 : {sum_squared.shape}")
print("\nC'est maintenant une matrice 10×10 !")
#%%
print("\n--- 3.5 : Application de la racine carrée ---")
# Distance euclidienne finale
distance_matrix = np.sqrt(sum_squared)
print(f"\nMatrice de distances (shape: {distance_matrix.shape}):")
print(distance_matrix)
#%%
print("\n" + "="*70)
print("ÉTAPE 4 : Vérification de la diagonale")
print("="*70)

diagonal = np.diag(distance_matrix)
print(f"\nValeurs diagonales (distance d'un point à lui-même):")
print(diagonal)
print(f"\nMaximum sur la diagonale : {np.max(np.abs(diagonal)):.2e}")

if np.allclose(diagonal, 0):
    print("✓ Tous les termes diagonaux sont bien nuls (ou très proches de 0)")
else:
    print("✗ Attention : certains termes diagonaux ne sont pas nuls")


#%%
print("\n" + "="*70)
print("ÉTAPE 5 : Classement des points par distance (np.argsort)")
print("="*70)

# Pour chaque ligne, trier les indices par distance croissante
sorted_indices = np.argsort(distance_matrix, axis=1)

print("\nIndices des points triés par distance croissante :")
print("(pour chaque point, liste des autres points du plus proche au plus loin)\n")

for i in range(len(sorted_indices)):
    print(f"Point {i}: {sorted_indices[i]}")
    print(f"  → Distances correspondantes: {distance_matrix[i, sorted_indices[i]]}")

print("\nExplication :")
print("  - La première colonne contient toujours le point lui-même (distance 0)")
print("  - Les colonnes suivantes sont les voisins par ordre de proximité")

#%%
print("\n" + "="*70)
print("ÉTAPE 6 : K plus proches voisins avec argpartition (k=2)")
print("="*70)

k = 2
print(f"\nRecherche des {k} plus proches voisins pour chaque point")
print("(en excluant le point lui-même)\n")

# argpartition réorganise pour avoir les k+1 plus petits éléments au début
# (k+1 car on inclut le point lui-même dans le comptage)
partitioned_indices = np.argpartition(distance_matrix, k+1, axis=1)

print(f"Indices après argpartition (k={k}) :")
print(partitioned_indices)

print("\n" + "-"*70)
print("Analyse détaillée des k plus proches voisins :")
print("-"*70)

for i in range(len(X)):
    # Les k+1 premiers éléments contiennent le point lui-même + ses k voisins
    nearest_k = partitioned_indices[i, :k+1]
    # On retire le point lui-même (distance 0)
    nearest_k = nearest_k[nearest_k != i][:k]
    
    print(f"\nPoint {i} en position {X[i]}:")
    for j, neighbor_idx in enumerate(nearest_k, 1):
        dist = distance_matrix[i, neighbor_idx]
        print(f"  {j}. Voisin {neighbor_idx} à distance {dist:.4f} - position {X[neighbor_idx]}")

print("\n" + "="*70)
print("ÉTAPE 7 : Visualisation des connexions avec les plus proches voisins")
print("="*70)

plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], s=150, c='blue', alpha=0.6, 
           edgecolors='black', linewidths=2, zorder=3)

# Annoter les points
for i in range(len(X)):
    plt.annotate(f'P{i}', (X[i, 0], X[i, 1]), 
                xytext=(8, 8), textcoords='offset points', 
                fontsize=11, fontweight='bold')

# Tracer les connexions vers les k plus proches voisins
for i in range(len(X)):
    nearest_k = partitioned_indices[i, :k+1]
    nearest_k = nearest_k[nearest_k != i][:k]
    
    for neighbor_idx in nearest_k:
        plt.plot([X[i, 0], X[neighbor_idx, 0]], 
                [X[i, 1], X[neighbor_idx, 1]], 
                'r-', alpha=0.3, linewidth=1, zorder=1)

plt.xlabel('Coordonnée x', fontsize=12)
plt.ylabel('Coordonnée y', fontsize=12)
plt.title(f'K plus proches voisins (k={k})', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('knn_connexions.png', dpi=150, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : knn_connexions.png")

print("\n" + "="*70)
print("COMPARAISON : argsort vs argpartition")
print("="*70)

print("\nDifférence entre argsort et argpartition :")
print("\n1. np.argsort :")
print("   - Trie TOUS les éléments")
print("   - Plus lent pour grandes matrices")
print("   - Résultat complètement trié")

print("\n2. np.argpartition :")
print("   - Partitionne autour de k")
print("   - Plus rapide quand on veut seulement les k premiers")
print("   - Les k premiers sont garantis, mais pas triés entre eux")
print("   - Le reste n'est pas trié non plus")

print("\nExemple pour le point 0 :")
print(f"  argsort:      {sorted_indices[0]}")
print(f"  argpartition: {partitioned_indices[0]}")
print("\nNotez que les k+1 premiers de argpartition contiennent les bons voisins,")
print("mais pas nécessairement dans l'ordre exact.")

print("\n" + "="*70)
print("RÉCAPITULATIF DU BROADCASTING")
print("="*70)

print("""
Le broadcasting est la clé de cet exercice. Voici comment ça fonctionne :

1. X shape (10, 2) : 10 points en 2D

2. X1 = X[:, np.newaxis, :] → shape (10, 1, 2)
   Chaque point est "isolé" dans sa propre tranche

3. X2 = X[np.newaxis, :, :] → shape (1, 10, 2)
   Tous les points sont dans une seule tranche

4. X1 - X2 → shape (10, 10, 2)
   NumPy broadcast automatiquement :
   - Dimension 0 : 10 répété 1 fois = 10 (de X1)
   - Dimension 1 : 1 répété 10 fois = 10 (de X2)
   - Dimension 2 : 2 inchangé = 2
   
   Résultat : pour chaque paire (i,j), on a les 2 différences de coordonnées

5. Somme sur axis=-1 et racine carrée → matrice de distances 10×10

Avantage : Calcul vectorisé ultra-rapide, pas de boucles Python !
""")

print("\n" + "="*70)
print("FIN DE L'EXERCICE")
print("="*70)
print("\n✓ Fichiers générés :")
print("  - nuage_points.png : visualisation des points")
print("  - knn_connexions.png : connexions entre voisins proches")
# %%
