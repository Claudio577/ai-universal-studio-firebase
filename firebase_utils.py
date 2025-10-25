import os, json, tempfile, requests
import firebase_admin
from firebase_admin import credentials, firestore, storage

def init_firebase():
    if not firebase_admin._apps:
        creds_json = json.loads(os.environ["FIREBASE_CREDENTIALS"])
        cred = credentials.Certificate(creds_json)
        firebase_admin.initialize_app(cred, {
            'storageBucket': os.environ["FIREBASE_BUCKET"]
        })
    return firestore.client(), storage.bucket()

def baixar_arquivo(url):
    r = requests.get(url)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(r.content)
    return tmp.name
