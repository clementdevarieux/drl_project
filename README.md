# Projet DRL 5IABD2

## Objectif
L'objectif de ce projet est de mettre en place un environnement de simulation pour un agent intelligent capable de jouer à un jeu de type "Farkle". L'agent devra apprendre à jouer à ce jeu en utilisant des techniques de Deep Reinforcement Learning.

## Jeu de Farkle
Le jeu de Farkle est un jeu de dés qui se joue avec 6 dés. Le but du jeu est de marquer le plus de points possible en lançant les dés. Les règles du jeu sont les suivantes:
- Un joueur lance les 6 dés
- Le joueur peut choisir de garder certains dés et de relancer les autres
- Le joueur peut relancer les dés autant de fois qu'il le souhaite
- Si le joueur ne peut pas garder de dés, il perd tous les points accumulés lors de ce tour
- Si le joueur garde tous les dés, il peut relancer les 6 dés
- Le joueur peut marquer des points en gardant des dés selon les combinaisons suivantes:
  - 1: 100 points
  - 5: 50 points
  - 3 dés identiques: 100 * valeur du dé, ou 1000 si ce sont 3 dés de 1
  - 4 dés identiques: 200 * valeur du dé, ou 2000 si ce sont 4 dés de 1
  - 5 dés identiques: 400 * valeur du dé, ou 4000 si ce sont 5 dés de 1
  - 6 dés identiques: 800 * valeur du dé, ou 8000 si ce sont 6 dés de 1
  - Suite de 6 dés: 1500 points
  - 3 paires: 1500 points
  - 2 triplets: 2500 points
  - 4 identiques et une paire: 1500 points
  - Rien en un lancé: 500 points

## Environnement de simulation
