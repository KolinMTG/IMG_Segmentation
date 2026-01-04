
"""Renvoie un logger configuré pour l'application."""
#! /!\ Pour utiliser les logs, faire dans votre script :
# from log_config import get_logger
# logger = get_logger("nom_du_fichier_log.log", log_level=logging.INFO, console=True)
#! Puis effectuer vos appels au logger normalement
# exemple : logger.info("Ceci est un message d'information")


import logging
from src.cste import GeneralPath
import os

# Assure que le répertoire de logs existe
if not os.path.exists(GeneralPath.LOG_PATH):
    os.makedirs(GeneralPath.LOG_PATH)


def get_logger(log_file_name: str, log_level=logging.INFO, console: bool = True) -> logging.Logger:
    """Configure and return a logger with file and optional console output.

    Args:
        log_file_name (str): Nom du fichier de log.
        log_level (int): Niveau de log (ex: logging.INFO).
        console (bool): True pour afficher aussi dans la console, False sinon.
    """
    logger = logging.getLogger(log_file_name)
    logger.setLevel(log_level)

    # Évite les doublons de handlers si le logger existe déjà
    if not logger.handlers:
        log_file = os.path.join(GeneralPath.LOG_PATH, log_file_name)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(file_handler)

        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H:%M:%S'))
            logger.addHandler(console_handler)

    return logger