import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD
from heapq import nlargest
import azure.functions as func
import logging
import json
import pickle
import requests

# Chargement des données au démarrage
logging.info("Chargement des données...")


def get_pickle_from_url(url):
    """
    Downloads a pickle file from the given URL and loads it into a Python object.

    Args:
      url: The URL of the pickle file.

    Returns:
      The loaded Python object.

    Raises:
      requests.exceptions.RequestException: If there's an issue with the network request.
      pickle.UnpicklingError: If there's an error while unpickling the data.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = pickle.loads(response.content)
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from URL: {e}")
        raise
    except pickle.UnpicklingError as e:
        print(f"Error unpickling data: {e}")
        raise

# Charger les fichiers Pickle
# embeddings_file = "azure_datas/articles_embeddings.pickle"
# clicks_file = "azure_datas/clicks.pickle"
# model_file = "azure_datas/model_SVD.pkl"
# articles_file = "azure_datas/articles.pickle"


embeddings_file = "https://ocaifunctionazure.blob.core.windows.net/ocai-p10-files/articles_embeddings.pickle"
clicks_file = "https://ocaifunctionazure.blob.core.windows.net/ocai-p10-files/clicks.pickle"
model_file = "https://ocaifunctionazure.blob.core.windows.net/ocai-p10-files/model_SVD.pkl"
articles_file = "https://ocaifunctionazure.blob.core.windows.net/ocai-p10-files/articles.pickle"

articles_embeddings = ""
df_clicks = ""
model_SVD = ""
df_articles = ""

try:
    articles_embeddings = get_pickle_from_url(embeddings_file)
    df_clicks = get_pickle_from_url(clicks_file)
    model_SVD = get_pickle_from_url(model_file)
    df_articles = get_pickle_from_url(articles_file)
except Exception as e:
    print(f"An error occurred: {e}")

# with open(embeddings_file, "rb") as f:
#    articles_embeddings = pickle.load(f)

# Charger les clics
# with open(clicks_file, "rb") as f:
#    df_clicks = pickle.load(f)

# Charger le modèle SVD entraîné
# with open(model_file, "rb") as f:
#    model_SVD = pickle.load(f)

# Charger les métadonnées des articles
# with open(articles_file, "rb") as f:
#    # DataFrame contenant les colonnes "article_id" et "category_id"
#    df_articles = pickle.load(f)

logging.info("Données et modèle SVD chargés avec succès.")

# Fonction de recommandation basée sur le contenu


def content_based_filtering_recommendation(user_id, clicks, embeddings, n=5):
    # Articles lus par l'utilisateur
    articles_read = clicks[clicks['user_id'] ==
                           user_id]['click_article_id'].tolist()
    logging.info(f"Articles lus par l'utilisateur {user_id} : {articles_read}")

    if not articles_read:
        logging.warning(f"L'utilisateur {user_id} n'a lu aucun article.")
        return []

    try:
        # Extraire les embeddings des articles lus
        articles_read_embedding = embeddings.loc[articles_read]
        logging.info(
            f"Embeddings des articles lus : {articles_read_embedding}")

        # Articles restants après avoir exclu les articles lus
        articles = embeddings.drop(index=articles_read)
        logging.info(
            f"Nombre d'articles disponibles pour recommandation : {len(articles)}")

        # Calculer la similarité cosinus
        matrix = cosine_similarity(articles_read_embedding, articles)
        logging.info(f"Matrice de similarité calculée avec succès.")

        # Générer les recommandations
        rec = []
        for _ in range(n):
            coord_x, coord_y = divmod(np.argmax(matrix), matrix.shape[1])
            recommended_article_id = int(articles.index[coord_y])
            rec.append(recommended_article_id)
            matrix[coord_x][coord_y] = 0

        rec.sort()
        logging.info(f"Recommandations générées : {rec}")
        return rec
    except KeyError as e:
        logging.error(f"Erreur lors de l'extraction des embeddings : {e}")
        return []

# Fonction de recommandation collaborative


def collaborative_filtering_recommendation(clicks, user_id, n=5):
    logging.info(
        f"Début de la recommandation collaborative pour l'utilisateur {user_id}.")

    # Liste des articles disponibles dans les embeddings
    available_articles = list(articles_embeddings.index)
    logging.info(
        f"Nombre total d'articles disponibles : {len(available_articles)}")

    # Articles déjà lus par l'utilisateur
    articles_read = clicks[clicks['user_id'] ==
                           user_id]['click_article_id'].tolist()
    logging.info(f"Articles lus par l'utilisateur {user_id} : {articles_read}")

    # Retirer les articles déjà lus des articles disponibles
    for article in articles_read:
        if article in available_articles:
            available_articles.remove(article)
    logging.info(
        f"Articles disponibles après exclusion des articles lus : {len(available_articles)}")

    # Si aucun article n'est disponible pour la recommandation
    if not available_articles:
        logging.warning(
            f"Aucun article disponible pour la recommandation collaborative pour l'utilisateur {user_id}.")
        return []

    # Calcul des prédictions
    predictions = {}
    for article_id in available_articles:
        pred = model_SVD.predict(user_id, article_id)
        predictions[article_id] = pred.est
    logging.info(f"Nombre de prédictions calculées : {len(predictions)}")

    # Sélectionner les meilleurs articles
    recommended_articles = nlargest(n, predictions, key=predictions.get)
    logging.info(
        f"Articles recommandés pour l'utilisateur {user_id} : {recommended_articles}")

    return recommended_articles


# Route pour les recommandations

app = func.FunctionApp()

@app.route(route="recommend", auth_level=func.AuthLevel.ANONYMOUS)
def HttpOcrFunc(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    """
    Route HTTP pour obtenir les recommandations.
    """
    logging.info("Requête reçue pour la recommandation.")

    try:
        # Récupérer l'ID utilisateur
        user_id_param = req.params.get("user_id")
        if user_id_param is None or user_id_param == "":
            raise ValueError("user_id est manquant.")

        # Convertir en entier (y compris l'utilisateur 0)
        user_id = int(user_id_param)
    except (TypeError, ValueError) as e:
        logging.error(f"Erreur de paramètre user_id : {str(e)}")
        return func.HttpResponse(
            "Veuillez fournir un 'user_id' valide.",
            status_code=400,
        )

    try:
        # Obtenir les recommandations pour les deux modèles
        recommendations_content_based = content_based_filtering_recommendation(
            user_id, df_clicks, articles_embeddings, n=5)
        recommendations_collaborative = collaborative_filtering_recommendation(
            df_clicks, user_id, n=5)

        # Ajouter les catégories aux recommandations
        content_based_results = [
            {
                "article_id": article_id,
                "category_id": int(df_articles.loc[df_articles["article_id"] == article_id, "category_id"].values[0])
            }
            for article_id in recommendations_content_based
        ]

        collaborative_results = [
            {
                "article_id": article_id,
                "category_id": int(df_articles.loc[df_articles["article_id"] == article_id, "category_id"].values[0])
            }
            for article_id in recommendations_collaborative
        ]

        # Construire la réponse
        response = {
            "content_based": content_based_results,
            "collaborative_filtering": collaborative_results
        }

        return func.HttpResponse(
            body=json.dumps(response),
            status_code=200,
            mimetype="application/json",
        )
    except Exception as e:
        logging.error(f"Erreur lors de la recommandation : {str(e)}")
        return func.HttpResponse(
            body="Une erreur s'est produite lors du traitement de la demande.",
            status_code=500,
        )