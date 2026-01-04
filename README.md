Dataset 3 – Aerial / satellite faciles (urbain ou champs)
Plusieurs jeux de données de segmentation aérienne sont pensés pour être “propres” : bâtiments, routes, végétation, etc. (par exemple Vaihingen, jeux simplifiés type “buildings/roads/vegetation”).​

Avantages

Grandes formes homogènes (bâtiments, routes, champs), donc les méthodes classiques + CNN marchent bien.

Application réelle claire (urbanisme, agriculture, cartographie).

Pour ton projet

Prétraitement : normalisation, éventuellement renforcement des contrastes entre classes (par ex. mettre en avant la végétation).

Segmentation classique (threshold ou K-means sur la couleur) pour dégrossir “végétation/bâtiment/route”, puis petit CNN encodeur‑décodeur pour nettoyer les frontières.


https://www.kaggle.com/datasets/aletbm/land-cover-from-aerial-imagery-landcover-ai



Environement virutelle 
IMG_SEG
Python 3.10.19 (Version)