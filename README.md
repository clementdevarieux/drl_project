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
### Lancement du GUI
Pour lancer le GUI, il suffit de lancer le script `run_game_gui.py` suivant:
```py
from environnements.Farkle_GUI_v4 import Farkle_GUI_v4

env = Farkle_GUI_v4()
env.run_game_GUI()
```
Vous obtiendrez alors l'inteface graphique suivante:

![Farkle GUI_1](images/gui_1.png)

Il vous faudra sélectionner un des id d'actions possibles.
Si nous prenons par exemple l'action 10, nous sauvegardons le dé 2 et le dé 4, augmentant ainsi notre score potentiel de 150, puis relançons les dés.

Nous voyons que cela a sauvegardé les deux dés, a relancé automatiquement les autres dés et a mis à jour le score potentiel.

![Farkle GUI_2](images/gui_2.png)

Le jeu continue ainsi jusqu'à ce qu'un des joueurs atteigne 10000 points.

### Choix qui ont été fait sur le fonctionnement des actions
Nous avons codé une fonction available_actions, qui retourne un vecteur de taille 6.

Par exemple, si nous avons les dés suivants : [1, 3, 4, 5, 6, 6], la fonction available_actions retournera le vecteur suivant : [1, 0, 0, 1, 0, 0].
Cela nous indique quels dés peuvent être sauvegardés (cela prend bien entendu en compte les dés déjà sauvegardés lors des jets précédents).

Nous avons aussi rajouté une convention mettant des 2 dans la sortie d'available actions lorsque les dés en question doivent obligatoirement être tous sélectionnés ensemble.
Par exemple, si nous avons les dés: [3, 1, 3, 3, 4, 6], available_actions retournera [2, 1, 2, 2, 0, 0].

Nous avons ensuite mis en place dans l'init de l'environnement 'available_action_keys_from_action', qui prend donc en entrée la sortie de self.available_actions et retourne les clés de TOUTES les combinaisons possibles d'actions réalisables à partir de notre vecteur d'available actions.

Ensuite, lorsq'un id d'action est selectionné, cela retourne vers self.actions_dict, qui renvoie un vecteur de taille 7 correspondant à l'action réalisée.
Les 6 premières valeurs correspondent aux dés sauvegardés, et la dernière valeur vaut 0 si nous relançons les dés, et 1 si nous décidons d'arrêter et de valider le score potentiel en cours.

### State description
Notre self.state_description fonctionne de la manière suivante:
- les 36 premières valeurs sont un one-hot de la valeur des dés
- les 12 valeurs suivantes sont un one-hot de si les dés en questions sont déjà sauvegardés ou non
- la valeur suivante est notre score potentiel en cours
- la suivante notre score réel déjà validé
- et la derniere est le score réel de l'adversaire

à notre qu'il a été choisi de mettre entre 0 et 1 le score afin de pouvoir donner cela directement à nos modèles faisant appel à des réseaux de neurones. Ainsi, lorsque nous scorons 150 points, nous ajoutons 0.015 à notre score.

### Récompenses
Lorsque notre agent atteint en premier un score de 1 (soit 10000 points), nous lui donnons un reward de 1. Si c'est l'adversaire jouant en random qui gagne, nous avons un reward de -1, sinon le reward est toujours à 0.

### Environnements de tests
Nous avons aussi utilisé des environnements de tests pour vérifier nos résultats, et pouvoir comparer l'efficacité des modèles en fonction de l'environnement.
Les environnements utilisés sont donc les suivants:
- Farkle_v4
- GridWorld
- TicTacToe
- LineWorld

## Modèles

Les algorithmes codés sont disponibles dans le folder algorithmes, et sont les suivants:
- Random
- Tabular Q-learning
- DQN
- Double DQN
- MCRR
- Reinforce
- MCTS

### Les résultats des modèles sauvegardés
Vous pourrez retrouver dans le folder model les entrainements sauvegardés des modèles, et pourrez ainsi faire appel à ces derniers pour tester le score obtenu sur un nouvelle partie.

Nous avons également sauvegardé les résultats de parties réalisées toutes les x parties pendant l'entrainement (sur une nouvelle partie faisant appel au modèle qui est en train de s'entrainer, en faisant une 'pause' durant l'entrainement). Ces résultats sont obtenables dans le folder results.