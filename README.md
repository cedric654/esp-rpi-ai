# Épreuve synthèse de programme - Technique de l'informatique (420.BO) - Intelligence artificielle

## Description

Mon projet est un système de caméra de sécurité intelligente. À l’aide de l’intelligence artificielle, la caméra de sécurité reconnait toutes les armes qu’une personne pourrait posséder, telle qu’un pistolet ou un couteau. Également, il est possible d’accéder à une application mobile et web qui permet de visualiser la caméra en direct.

## Installation

### Section 1 - Configuration de l'environnement

#### Étape 1 : Mettre à jour le Raspberry Pi

Tout d'abord, le Raspberry Pi doit être entièrement mis à jour. Ouvrez un terminal et lancer :

```bash
sudo apt-get update
sudo apt-get dist-upgrade
```

#### Étape 2 : Télécharger le référentiel GitHub

Ensuite, clonez ce référentiel GitHub en utilisant la commande suivante. Le référentiel contient les scripts que nous utiliserons pour exécuter TensorFlow Lite, ainsi qu'un script shell qui facilitera l'installation de tout. Lancer :

```bash
git clone https://github.com/cedric654/esp-rpi-ai.git
```

#### Étape 3 : Installer virtualenv

Changer de répertoire en lançant :

```bash
cd esp-rpi-ai
```

Installer virtualenv en lançant :

```bash
sudo pip3 install virtualenv
```

Créer ensuite l'environnement virtuel "esp-rpi-ai-env" en lançant :

```bash
python3 -m venv esp-rpi-ai-env
```

Maintenant, vous devez émettre cette commande pour activer l'environnement chaque fois que vous ouvrez une nouvelle fenêtre de terminal. Vous pouvez savoir quand l'environnement est actif en vérifiant si (esp-ai-env) apparaît avant le chemin dans votre invite de commande. Lancer :

```bash
source esp-rpi-ai-env/bin/activate
```

#### Étape 4 : Installer les dépendances TensorFlow Lite, OpenCV et Flask

Pour faciliter les choses, un script shell téléchargera et installera automatiquement tous les packages et dépendances. Exécutez-le en lançant :

```bash
bash get_pi_requirements.sh
```

### Section 2 - Exécuter le modèle Tensorflow Lite et le serveur Web

#### Étape 1 : Démarrer le serveur

Il est temps de voir le modèle de détection d'objets TFLite en action ! Tout d'abord, libérez de la mémoire et de la puissance de traitement en fermant toutes les applications que vous n'utilisez pas. Assurez-vous également que votre webcam ou Picamera est branchée.

Si vous suivez toutes les étapes correctement, il est censé fonctionner sans aucune erreur

Il est maintenant temps d'exécuter cette commande pour que cela fonctionne :

```bash
python main.py
```
