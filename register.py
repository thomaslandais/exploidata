import mysql.connector
from urllib.parse import urlparse
import csv
import os.path
import requests
import PyPDF2
import requests
from io import BytesIO
import numpy as np
import time


def getFileInfo(fileName):
    # Define the base URL for the API
    API_ROOT = "https://api.test.safe.a2display.fr"

    # Endpoint
    endpoint = ""

    # Check extension
    if fileName.endswith('.pdf'):
        # Retrieve between .pdf
        foldername = fileName.split('.pdf')[0]
        endpoint = "/uploads/objects/" + foldername + "/" + fileName
    elif fileName.endswith('.jpg'):
        endpoint = "/uploads/objects/" + fileName

    # Make a GET request to the API
    response = requests.get(API_ROOT + endpoint)
    start_time = time.time()
    number_of_pages = 1
    weight = np.nan

    # Check if the request was successful
    if response.status_code == 200:
        weight = len(response.content)
        if fileName.endswith('.pdf'):
            # Load response content as a file-like object
            file_object = BytesIO(response.content)

            # Read the PDF file
            pdf = PyPDF2.PdfReader(file_object)

            # Get the number of pages
            number_of_pages = len(pdf.pages)
            elapsed_time = time.time() - start_time
            print("Extact data took ", elapsed_time, "seconds => ", fileName.split('.')[1])
            #extrata text from 2 first pages of pdf
            content = ""
            for i in range(0,2):
                if i < len(pdf.pages):
                    content += pdf.pages[i].extract_text()
            # remove \n
            content = content.replace('\n', ' ')
            return {"number_of_pages": number_of_pages, "weight": weight, "content" : content}
        else:
            elapsed_time = time.time() - start_time
            print("Extact data took ", elapsed_time, "seconds => ", fileName.split('.')[1])
            return {"weight": weight}
    else:
        print("Request failed with status code", response.status_code , " => ", fileName.split('.')[1])
    




def registerDataInCsv(accountId) :
    # Informations de connexion à la base de données
    database_url = "mysql://thomas:display@51.15.10.163:51106/a2api.thomas2"

    # Analyser l'URL de la base de données pour obtenir les détails de connexion
    url_parts = urlparse(database_url)

    try:
        # Se connecter à la base de données
        conn = mysql.connector.connect(
            host=url_parts.hostname,
            port=url_parts.port,
            user=url_parts.username,
            password=url_parts.password,
            database=url_parts.path[1:]  # Supprimer le slash au début du chemin
        )

        # Créer un nouvel objet cursor
        cursor = conn.cursor()

        # Prépare la requête SQL
        query = """
        SELECT 
            o.name,
            o.file_extension,
            o.CREATION_DATETIME, 
            f.name as file_name,
            c.name as category_name
        FROM 
            objects o 
        JOIN 
            item_category ic ON o.id = ic.item_id 
        JOIN 
            categories c ON ic.category_id = c.id 
        JOIN
            files f ON o.file_id = f.id
        WHERE 
            o.account_id = """ + str(accountId) + """
        AND
            f.type = 'document'
        ORDER BY
            c.name ASC
        """

        # Exécutez la requête SQL
        cursor.execute(query)

        # Récupérer tous les résultats
        results = cursor.fetchall()
        # si il y a des résultats and if this file not already exist
        if(len(results) > 0):
            # enregistrer les résultats dans un csv 
            img_writer = csv.DictWriter(open('./data/img/data-'+ str(accountId) +'.csv', 'w', newline=''), fieldnames=['name', 'file_extension', 'CREATION_DATETIME', 'file_size', 'category_name'])
            pdf_writer = csv.DictWriter(open('./data/pdf/data-'+ str(accountId) +'.csv', 'w', newline=''), fieldnames=['name', 'file_extension', 'CREATION_DATETIME','Number_of_pages', 'file_size','content', 'category_name'])
            img_writer.writeheader()
            pdf_writer.writeheader()
            for result in results:
                fileInfos = getFileInfo(result[3])
                if fileInfos is not None and 'weight' in fileInfos:
                    if result[1] == 'jpg' or result[1] == 'png' or result[1] == 'jpeg' :
                        img_writer.writerow({'name': result[0], 'file_extension': result[1], 'CREATION_DATETIME' : result[2],'file_size':fileInfos['weight'], 'category_name': result[4]})
                    elif result[1] == 'pdf':
                        pdf_writer.writerow({'name': result[0], 'file_extension': result[1], 'CREATION_DATETIME' : result[2],'Number_of_pages' : fileInfos["number_of_pages"],'file_size':fileInfos['weight'],'content': fileInfos['content'], 'category_name': result[4]})
            print("Register data account " + str(accountId) + " in csv ")

    except mysql.connector.Error as err:
        print("Une erreur s'est produite lors de la connexion à la base de données:", err)

    finally:
        if conn is not None:
            conn.close()


def registerAllAccount() :
    if not os.path.exists('./data/img'):
        os.makedirs('./data/img')

    if not os.path.exists('./data/pdf'):
        os.makedirs('./data/pdf')
        # Informations de connexion à la base de données
    database_url = "mysql://thomas:display@51.15.10.163:51106/a2api.thomas2"

    # Analyser l'URL de la base de données pour obtenir les détails de connexion
    url_parts = urlparse(database_url)

    try:
        # Se connecter à la base de données
        conn = mysql.connector.connect(
            host=url_parts.hostname,
            port=url_parts.port,
            user=url_parts.username,
            password=url_parts.password,
            database=url_parts.path[1:]  # Supprimer le slash au début du chemin
        )

        # Créer un nouvel objet cursor
        cursor = conn.cursor()

        query = "Select id from accounts"

        # Exécutez la requête SQL
        cursor.execute(query)

        results = cursor.fetchall()

        for result in results:
            registerDataInCsv(result[0])

    except mysql.connector.Error as err:
        print("Une erreur s'est produite lors de la connexion à la base de données:", err)

    finally:
        if conn is not None:
            conn.close()

registerAllAccount()
        