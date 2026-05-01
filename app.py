from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split   # ✅ AJOUT: évaluation ML
from sklearn.metrics import accuracy_score             # ✅ AJOUT: précision ML
from sklearn.model_selection import cross_val_score    # ✅ Cross-validation
from sklearn.cluster import KMeans                     # ✅ Service 3: clustering profils
from sklearn.metrics.pairwise import cosine_similarity # ✅ Service 3: similarité IA
import numpy as np                                     # ✅ Service 3: vecteurs
from datetime import datetime
from functools import wraps
import sqlite3                                         # ✅ AJOUT: base de données
import os
from werkzeug.security import generate_password_hash, check_password_hash  # ✅ AJOUT: bcrypt

# ============================================================
# CONFIGURATION
# ============================================================
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

app.secret_key = os.environ.get('SECRET_KEY', 'maroctour_secret_key_dev_only_change_in_prod')

# ============================================================
# BASE DE DONNÉES SQLite  ✅ AMÉLIORÉ
# ============================================================
DB_PATH = "maroctour.db"

def get_db():
    """Obtenir une connexion SQLite"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialiser la base de données"""
    conn = get_db()
    cursor = conn.cursor()

    # Table profils ML — matching compagnons de voyage
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS travel_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_name TEXT NOT NULL,
            age INTEGER,
            sexe TEXT,
            budget INTEGER,
            marie TEXT,
            region TEXT,
            predicted_dest TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')

    # Table utilisateurs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL,
            destinations_visited INTEGER DEFAULT 0,
            trips_planned INTEGER DEFAULT 0,
            groups_joined INTEGER DEFAULT 0
        )
    ''')

    # Table groupes de voyage (persistance SQLite)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dest TEXT NOT NULL,
            date TEXT NOT NULL,
            max INTEGER NOT NULL,
            current INTEGER DEFAULT 1,
            desc TEXT,
            creator_id INTEGER,
            created_at TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS group_members (
            group_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            PRIMARY KEY (group_id, user_id)
        )
    ''')

    # Table destinations (ajoutées par l'admin)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS destinations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            budget INTEGER DEFAULT 0,
            season TEXT,
            description TEXT,
            created_at TEXT NOT NULL
        )
    ''')

    # Table destinations sauvegardées (service 1)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_destinations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            nom TEXT NOT NULL,
            type TEXT NOT NULL,
            emoji TEXT,
            description TEXT,
            budget INTEGER,
            tags TEXT,
            created_at TEXT NOT NULL
        )
    ''')

    # Table groupes sauvegardés (service 3)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            group_id INTEGER NOT NULL,
            group_name TEXT NOT NULL,
            destination TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, group_id)
        )
    ''')

    # Table plans de voyage sauvegardés
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trip_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            destination TEXT NOT NULL,
            duree INTEGER NOT NULL,
            voyageurs INTEGER NOT NULL,
            budget_utilisateur INTEGER NOT NULL,
            plan_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')

    # Table commentaires utilisateurs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_name TEXT NOT NULL,
            destination TEXT NOT NULL,
            content TEXT NOT NULL,
            rating INTEGER DEFAULT 5,
            created_at TEXT NOT NULL
        )
    ''')

    # Migration: ajouter les colonnes manquantes si la DB existait avant
    migrations = [
        "ALTER TABLE users ADD COLUMN destinations_visited INTEGER DEFAULT 0",
        "ALTER TABLE users ADD COLUMN trips_planned INTEGER DEFAULT 0",
        "ALTER TABLE users ADD COLUMN groups_joined INTEGER DEFAULT 0",
        # Nouvelles colonnes groupes enrichis
        "ALTER TABLE groups ADD COLUMN type_activite TEXT DEFAULT 'Mixte'",
        "ALTER TABLE groups ADD COLUMN niveau TEXT DEFAULT 'Tous niveaux'",
        "ALTER TABLE groups ADD COLUMN interets TEXT DEFAULT '[]'",
        "ALTER TABLE groups ADD COLUMN age_min INTEGER DEFAULT 18",
        "ALTER TABLE groups ADD COLUMN age_max INTEGER DEFAULT 99",
        "ALTER TABLE groups ADD COLUMN budget_par_personne INTEGER DEFAULT 0",
        "ALTER TABLE groups ADD COLUMN langue TEXT DEFAULT 'Arabe/Français'",
    ]
    for sql in migrations:
        try:
            cursor.execute(sql)
        except Exception:
            pass  # colonne déjà existante

    conn.commit()
    conn.close()
    print("[OK] Base de données SQLite initialisée")

# Initialiser la DB au démarrage
init_db()

# Contenu admin
admin_content = {
    "news": "",
    "welcome_msg": ""
}

# Admin credentials (depuis variables d'environnement)
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'admin@maroctour.ma')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')

# ============================================================
# MACHINE LEARNING - Chargement et entraînement
# ============================================================
print("[*] Chargement du modele ML...")

try:
    df = pd.read_csv("data.csv", encoding='utf-8-sig')

    le_sexe = LabelEncoder()
    le_marie = LabelEncoder()
    le_region = LabelEncoder()
    le_destination = LabelEncoder()

    df["sexe"] = le_sexe.fit_transform(df["sexe"])
    df["marié"] = le_marie.fit_transform(df["marié"])
    df["région_origine"] = le_region.fit_transform(df["région_origine"])
    df["type_destination"] = le_destination.fit_transform(df["type_destination"])

    X = df.drop("type_destination", axis=1)
    y = df["type_destination"]

    # Train/Test Split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Normalisation des features numériques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Modèle SVM (rbf) — meilleure généralisation que Random Forest
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        random_state=42,
        probability=True
    )
    model.fit(X_train_scaled, y_train)

    # Accuracy sur split interne (30%)
    y_pred = model.predict(X_test_scaled)
    ML_ACCURACY = round(accuracy_score(y_test, y_pred) * 100, 2)

    # Cross-validation 5 folds — evaluation plus fiable
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')
    ML_CV_SCORE = round(cv_scores.mean() * 100, 2)
    ML_CV_STD   = round(cv_scores.std()  * 100, 2)

    print(f"[OK] SVM (rbf) entraine avec {len(X_train)} echantillons (70%)")
    print(f"[OK] Accuracy split interne ({len(X_test)} echantillons, 30%): {ML_ACCURACY}%")
    print(f"[OK] Cross-validation 5-fold: {ML_CV_SCORE}% (+/- {ML_CV_STD}%)")

    # Test sur test_model.csv (data externe jamais vue)
    ML_ACCURACY_EXT = None
    if os.path.exists("test_model.csv"):
        try:
            df_ext = pd.read_csv("test_model.csv", encoding='utf-8-sig')
            ext_cols = df_ext.columns.tolist()

            def safe_enc(le, val):
                return le.transform([val])[0] if val in le.classes_ else 0

            df_ext['sexe_e']   = df_ext[ext_cols[1]].apply(lambda v: safe_enc(le_sexe,   v))
            df_ext['marie_e']  = df_ext[ext_cols[3]].apply(lambda v: safe_enc(le_marie,  v))
            df_ext['region_e'] = df_ext[ext_cols[4]].apply(lambda v: safe_enc(le_region, v))

            X_ext = df_ext[[ext_cols[0], 'sexe_e', ext_cols[2], 'marie_e', 'region_e']].values
            y_ext_true = df_ext[ext_cols[5]].tolist()
            y_ext_pred = le_destination.inverse_transform(model.predict(scaler.transform(X_ext)))

            correct = sum(1 for t, p in zip(y_ext_true, y_ext_pred) if t == p)
            ML_ACCURACY_EXT = round(correct / len(y_ext_true) * 100, 2)
            print(f"[OK] Test externe test_model.csv ({len(y_ext_true)} cas): {ML_ACCURACY_EXT}%")

            erreurs = [(y_ext_true[i], y_ext_pred[i]) for i in range(len(y_ext_true)) if y_ext_true[i] != y_ext_pred[i]]
            if erreurs:
                print(f"[!]  {len(erreurs)} erreur(s) sur test externe")
            else:
                print("[OK] Aucune erreur sur test externe")
        except Exception as e_ext:
            print(f"[!]  test_model.csv erreur: {e_ext}")

    ML_READY = True

except Exception as e:
    print(f"[!]  Erreur ML: {e}")
    print("   Le service 1 utilisera la logique simplifiée")
    ML_READY = False
    ML_ACCURACY_EXT = None
    ML_ACCURACY = 0

# ============================================================
# ML SERVICE 3 - KMeans + Cosine Similarity (groupes IA)
# ============================================================

# Vocabulaire fixe pour l'encodage des vecteurs
_ACTIVITES  = ['Ville', 'Aventure', 'Culture', 'Plage', 'Montagne', 'Mixte']
_NIVEAUX    = ['Tous niveaux', 'Débutant', 'Intermédiaire', 'Avancé']
_INTERETS   = ['Randonnée', 'Photo', 'Gastronomie', 'Histoire', 'Surf',
               'Camping', 'Musique', 'Méditation', 'Escalade', 'Artisanat',
               'Désert', 'Plongée']

def encode_profile_vector(age, budget, type_activite, niveau, interets):
    """Convertit un profil (user ou groupe) en vecteur numérique pour cosine similarity."""
    vec = []
    vec.append(min(age, 100) / 100.0)          # âge normalisé
    vec.append(min(budget, 20000) / 20000.0)   # budget normalisé
    for act in _ACTIVITES:                      # type activité — one-hot
        vec.append(1.0 if type_activite == act else 0.0)
    for niv in _NIVEAUX:                        # niveau — one-hot
        vec.append(1.0 if niveau == niv else 0.0)
    for interet in _INTERETS:                   # intérêts — multi-hot
        vec.append(1.0 if interet in interets else 0.0)
    return np.array(vec, dtype=float)

print("[*] Entraînement KMeans (Service 3)...")
try:
    df_km = pd.read_csv("data.csv", encoding='utf-8-sig')

    # Encoder toutes les features (âge, budget, sexe, marié)
    le_km_sexe  = LabelEncoder()
    le_km_marie = LabelEncoder()
    df_km['sexe_enc']  = le_km_sexe.fit_transform(df_km['sexe'])
    df_km['marie_enc'] = le_km_marie.fit_transform(df_km['marié'])

    X_km = df_km[['âge', 'budget', 'sexe_enc', 'marie_enc']].values
    scaler_km = StandardScaler()
    X_km_scaled = scaler_km.fit_transform(X_km)

    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_km['cluster'] = kmeans_model.fit_predict(X_km_scaled)

    # Trouver le type_destination dominant par cluster
    cluster_type_map = (
        df_km.groupby('cluster')['type_destination']
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    KMEANS_READY = True
    for k, v in cluster_type_map.items():
        print(f"  Cluster {k} = {v}")
    print(f"[OK] KMeans entraine - 4 clusters avec type dominant")
except Exception as e:
    KMEANS_READY = False
    cluster_type_map = {}
    print(f"[!]  KMeans erreur: {e}")

# ============================================================
# ENRICHISSEMENT DES LIEUX (description, tags, highlights…)
# ============================================================
ENRICHISSEMENT_LIEUX = {
    # ── VILLES ──────────────────────────────────────────────
    "Rabat": {
        "emoji": "🏛️",
        "description": "Capitale royale classée UNESCO, Rabat marie une médina andalouse du XIIe siècle, des boulevards haussmanniens et une côte atlantique sauvage. Ville douce, sûre et accueillante.",
        "tags": ["Patrimoine", "Culture", "Famille", "Détente"],
        "highlights": ["Kasbah des Oudayas (UNESCO)", "Tour Hassan & Mausolée Mohammed V", "Chellah (nécropole romaine)", "Corniche & Plage de Oudayas"],
        "pour_qui": "Familles, couples, voyageurs culturels",
        "a_ne_pas_manquer": "Le coucher de soleil depuis la Kasbah des Oudayas sur l'estuaire du Bou Regreg.",
        "transport_interne": "Tramway (8 MAD), petit taxi (10-25 MAD), vélo en libre-service",
        "niveau_visite": "Facile",
        "securite": "Très sûre — capitale, présence policière permanente",
        "langue": "Darija, Français, Espagnol, Anglais",
        "monnaie": "MAD — nombreux distributeurs",
        "classification_raison": "Ville culturelle et patrimoniale idéale pour les budgets moyens à élevés et les familles"
    },
    "Marrakech": {
        "emoji": "🕌",
        "description": "La Ville Rouge hypnotise par ses souks labyrinthiques, ses riads secrets et la place Jemaa el-Fna classée UNESCO. Carrefour entre tradition millénaire et tourisme international.",
        "tags": ["Culture", "Shopping", "Gastronomie", "Luxe", "Nuit"],
        "highlights": ["Place Jemaa el-Fna (UNESCO)", "Jardins Majorelle", "Palais Bahia", "Hammam traditionnel", "Souks médinaux"],
        "pour_qui": "Couples, groupes, aventuriers urbains, foodies",
        "a_ne_pas_manquer": "La place Jemaa el-Fna à la tombée du jour : 100 stands de cuisine, musiciens Gnawa et jongleurs.",
        "transport_interne": "Calèche (50 MAD), petit taxi (15-30 MAD), marche dans la médina",
        "niveau_visite": "Modéré (médina dense)",
        "securite": "Bonne — restez vigilant dans les souks (arnaques fréquentes)",
        "langue": "Darija Marrakchia, Français, Anglais, Espagnol",
        "monnaie": "MAD — acceptation partielle des cartes dans les riads",
        "classification_raison": "Destination Ville premium pour profils avec budget élevé et attrait pour la culture"
    },
    "Fès": {
        "emoji": "🏺",
        "description": "La capitale spirituelle du Maroc est la plus ancienne cité médiévale du monde encore habitée. Ses 9 400 ruelles, ses madrasas et ses tanneries mérinides sont un voyage dans le temps.",
        "tags": ["Patrimoine", "Histoire", "Spiritualité", "Art", "Artisanat"],
        "highlights": ["Médina Fès el-Bali (UNESCO)", "Tanneries Chouara", "Medersa Bou Inania", "Université al-Qaraouiyyin (859 ap. J.-C.)", "Musée Nejjarine"],
        "pour_qui": "Historiens, amateurs d'art, couples, voyageurs solo",
        "a_ne_pas_manquer": "Les tanneries vues de la terrasse d'un magasin de cuir — palette de couleurs primaires dans les cuves médiévales.",
        "transport_interne": "Petit taxi (15-20 MAD), âne (médina), guide officiel conseillé",
        "niveau_visite": "Difficile sans guide (médina labyrinthique)",
        "securite": "Bonne — engagez un guide officiel ONMT pour éviter les arnaques",
        "langue": "Darija Fassi, Français, Anglais",
        "monnaie": "MAD — préférez le cash dans la médina",
        "classification_raison": "Destination Ville idéale pour profils culturels avec budget moyen et intérêt pour l'histoire"
    },
    "Chefchaouen": {
        "emoji": "💙",
        "description": "La Perle Bleue du Rif : chaque ruelle est peinte en nuances de bleu cobalt, indigo et turquoise. Village de montagne fondé en 1471, réputé pour son artisanat rifain et son fromage de chèvre.",
        "tags": ["Nature", "Photographie", "Détente", "Artisanat", "Randonnée"],
        "highlights": ["Médina bleue (photogénique)", "Cascade d'Akchour (40 min)", "Ras El Ma (25 min à pied)", "Kasbah Ethnographique", "Mosquée Espagnole (vue panoramique)"],
        "pour_qui": "Photographes, couples, voyageurs solo, backpackers",
        "a_ne_pas_manquer": "Le golden hour dans la médina bleue — lumière dorée sur les murs bleus, ruelles désertes avant 8h.",
        "transport_interne": "Marche (médina compacte), taxi pour Akchour (30 MAD)",
        "niveau_visite": "Facile",
        "securite": "Très sûre — village tranquille",
        "langue": "Darija Rifaine, Espagnol, Français",
        "monnaie": "MAD — peu de distributeurs, emportez du cash",
        "classification_raison": "Destination Ville romantique pour profils jeunes, célibataires ou couples avec budget modéré"
    },
    "Casablanca": {
        "emoji": "🌊",
        "description": "La métropole cosmopolite du Maroc : 4 millions d'habitants, gratte-ciels, Corniche atlantique et la 3ème plus grande mosquée du monde. Capitale économique et culturelle moderne.",
        "tags": ["Moderne", "Affaires", "Shopping", "Gastronomie", "Nuit"],
        "highlights": ["Mosquée Hassan II (3ème mondiale)", "Corniche Ain Diab", "Morocco Mall (Afrique)", "Quartier Habous", "Boulevard Art Déco"],
        "pour_qui": "Professionnels, familles, shoppers, amateurs d'architecture",
        "a_ne_pas_manquer": "La Mosquée Hassan II au coucher du soleil — sol en verre sur l'Atlantique et minaret de 210 m.",
        "transport_interne": "Train Casa-Port/Voyageurs (55 MAD), bus, taxi (20-50 MAD)",
        "niveau_visite": "Facile — ville moderne bien organisée",
        "securite": "Bonne dans les quartiers touristiques — normale dans la médina",
        "langue": "Darija, Français, Anglais, Espagnol",
        "monnaie": "MAD — acceptation large des cartes",
        "classification_raison": "Destination Ville pour profils urbains, budgets élevés et voyageurs d'affaires"
    },
    "Agadir": {
        "emoji": "🏖️",
        "description": "Station balnéaire moderne à la plage de sable fin de 10 km. Rebâtie après le tremblement de terre de 1960, Agadir offre un tourisme de soleil, de surf et de gastronomie de mer.",
        "tags": ["Plage", "Sport", "Famille", "Détente", "Soleil"],
        "highlights": ["Plage d'Agadir (10 km)", "Kasbah d'Agadir Oufla (vue)", "Souk El Had (marché)", "Marina", "Surf & Kitesurf"],
        "pour_qui": "Familles, sportifs, amateurs de plage, couples",
        "a_ne_pas_manquer": "Le coucher de soleil depuis la terrasse de la Kasbah sur la baie d'Agadir et l'Atlantique.",
        "transport_interne": "Taxi (15-30 MAD), bus, location de vélo",
        "niveau_visite": "Facile",
        "securite": "Très sûre — ville touristique organisée",
        "langue": "Amazigh Chleuh, Darija, Français, Anglais",
        "monnaie": "MAD — large acceptation des cartes",
        "classification_raison": "Destination Ville-plage pour tous profils avec budget moyen à élevé"
    },
    "Tanger": {
        "emoji": "⚓",
        "description": "Ville-carrefour à la croisée de l'Atlantique et de la Méditerranée, de l'Europe et de l'Afrique. Port mythique aux allures cosmopolites, chargée d'histoire et de personnages légendaires.",
        "tags": ["Histoire", "Culture", "Gastronomie", "Vue mer", "Carrefour"],
        "highlights": ["Cap Spartel (pointe Afrique-Atlantique)", "Grotte d'Hercule", "Médina & Kasbah", "Villa Perdicaris", "Port et corniche"],
        "pour_qui": "Voyageurs historiques, couples, aventuriers, Européens en escale",
        "a_ne_pas_manquer": "Cap Spartel à l'aube — point de rencontre de l'Atlantique et de la Méditerranée, deux eaux de couleurs différentes.",
        "transport_interne": "Petit taxi (15-25 MAD), bus, marche dans la médina",
        "niveau_visite": "Modéré",
        "securite": "Bonne — restez attentif au port et dans les souks",
        "langue": "Darija Tanjaouia, Espagnol, Français, Anglais",
        "monnaie": "MAD — acceptation des cartes dans les hôtels",
        "classification_raison": "Destination Ville pour profils ouverts sur l'Europe, budget modéré, voyageurs solo ou en couple"
    },
    "Essaouira": {
        "emoji": "🌬️",
        "description": "La Cité des Vents : médina fortifiée portugaise du XVIIe siècle classée UNESCO, réputée pour ses galeries d'art, ses concerts Gnawa et son spot de windsurf de renommée mondiale.",
        "tags": ["Art", "Musique", "Surf", "Patrimoine", "Détente"],
        "highlights": ["Remparts & Scala (UNESCO)", "Port de pêche artisanal", "Windsurf & Kitesurf", "Galeries Gnawa", "Arganiers & chèvres grimpantes"],
        "pour_qui": "Artistes, musiciens, surfeurs, couples romantiques",
        "a_ne_pas_manquer": "Le Festival Gnawa (juin) — musique mystique sous les étoiles dans les remparts de la médina.",
        "transport_interne": "Marche (médina compacte), vélo, taxi (10-20 MAD)",
        "niveau_visite": "Facile",
        "securite": "Très sûre — ambiance détendue, village d'artistes",
        "langue": "Darija, Français, Anglais",
        "monnaie": "MAD — peu de distributeurs, cash conseillé",
        "classification_raison": "Destination Ville-nature pour profils créatifs, couples ou solo avec budget modéré"
    },
    # ── NATURE ──────────────────────────────────────────────
    "Oukaïmeden": {
        "emoji": "⛷️",
        "description": "Station de ski la plus haute d'Afrique (2 650 m) et paradis de la randonnée printanière. Gravures rupestres néolithiques, télésièges et panorama sur le Haut Atlas.",
        "tags": ["Ski", "Randonnée", "Altitude", "Nature", "Sport"],
        "highlights": ["Pistes de ski (débutant à confirmé)", "Gravures rupestres néolithiques (UNESCO)", "Randonnée au Jbel Oukaïmeden (3 273 m)", "Lac d'altitude", "Village berbère"],
        "pour_qui": "Sportifs, familles, amateurs de montagne",
        "a_ne_pas_manquer": "En hiver : ski avec vue sur les sommets enneigés. Au printemps : tapis de fleurs sauvages à 2 600 m.",
        "transport_interne": "Télésiège, randonnée à pied",
        "niveau_visite": "Modéré (altitude)",
        "securite": "Bonne — station organisée, secours montagne disponible",
        "langue": "Amazigh Tachelhit, Darija, Français",
        "monnaie": "MAD — peu de commerces, emportez du cash",
        "classification_raison": "Destination Nature-montagne idéale pour profils sportifs avec budget modéré"
    },
    "Vallée d'Imlil": {
        "emoji": "🏔️",
        "description": "Porte du Toubkal (4 167 m, plus haut sommet d'Afrique du Nord). Vallée berbère encaissée à 1 740 m d'altitude : noyers, villages de pisé rouge et cascades d'altitude.",
        "tags": ["Trek", "Montagne", "Berbère", "Aventure", "Nature"],
        "highlights": ["Trek Toubkal (2 jours, 4 167 m)", "Villages Aremd & Achayn", "Cascade d'Imlil", "Vergers & apiculture", "Kasbah du Toubkal (vue)"],
        "pour_qui": "Trekkeurs, aventuriers, couples, voyageurs solo sportifs",
        "a_ne_pas_manquer": "L'ascension du Toubkal (4 167 m) — toit de l'Afrique du Nord avec vue sur l'Atlantique et le Sahara.",
        "transport_interne": "Taxi collectif depuis Marrakech, randonnée à pied, mulets",
        "niveau_visite": "Difficile (trek haute altitude)",
        "securite": "Bonne — guide obligatoire au-dessus de 3 000 m",
        "langue": "Amazigh Tachelhit, Darija",
        "monnaie": "MAD — cash uniquement dans la vallée",
        "classification_raison": "Destination Nature-aventure pour profils jeunes, célibataires ou couples sportifs"
    },
    "Parc national de Toubkal": {
        "emoji": "🦅",
        "description": "Premier parc national du Maroc (1942), 380 km² de paysages alpins : forêts de genévriers, cascades, lacs de montagne et faune endémique (aigle royal, mouflon, lynx caracal).",
        "tags": ["Faune", "Randonnée", "Biodiversité", "Altitude", "Trek"],
        "highlights": ["Sommet Toubkal (4 167 m)", "Lac Ifni (2 310 m)", "Forêts de genévriers millénaires", "Aigle royal & mouflon du Haut Atlas", "Refuges CAF"],
        "pour_qui": "Naturalistes, trekkeurs, photographes animaliers",
        "a_ne_pas_manquer": "Le Lac Ifni à 2 310 m — miroir d'altitude encerclé de sommets, accessible en 6h de trek depuis Imlil.",
        "transport_interne": "Randonnée à pied, mulets, guide local",
        "niveau_visite": "Très difficile (randonnée multi-jours)",
        "securite": "Bonne avec guide — conditions météo changeantes",
        "langue": "Amazigh Tachelhit",
        "monnaie": "MAD — cash uniquement",
        "classification_raison": "Destination Nature-aventure extrême pour profils très sportifs avec équipement adéquat"
    },
    "Gorges du Todra": {
        "emoji": "🏞️",
        "description": "Canyon spectaculaire de 300 m de hauteur taillé dans le calcaire rouge par l'oued Todra. L'un des plus beaux sites d'escalade d'Afrique, entouré de palmeraies et de kasbahs.",
        "tags": ["Escalade", "Randonnée", "Paysage", "Desert", "Nature"],
        "highlights": ["Canyon principal (300 m)", "Escalade (40+ voies 3-7b)", "Gorges supérieures (Tamtattouchte)", "Palmeraie de Tinghir (14 km)", "Kasbahs du Draa"],
        "pour_qui": "Grimpeurs, randonneurs, photographes, couples aventuriers",
        "a_ne_pas_manquer": "Les gorges à 11h — la lumière zénithale illumine les parois rouges en or. Baignade dans l'oued à 12°C.",
        "transport_interne": "Taxi depuis Tinghir (30 MAD), marche dans les gorges",
        "niveau_visite": "Modéré (canyon accessible à pied)",
        "securite": "Très bonne — site touristique balisé",
        "langue": "Amazigh Tamazight, Darija, Français",
        "monnaie": "MAD — distributeurs à Tinghir",
        "classification_raison": "Destination Nature pour profils actifs avec budget modéré, célibataires ou couples"
    },
    "Dunes de Merzouga (Sahara)": {
        "emoji": "🏜️",
        "description": "Erg Chebbi : dunes de sable fin pouvant atteindre 160 m de hauteur. L'expérience saharienne ultime : lever de soleil à dromadaire, nuit en bivouac et ciel étoilé à 0% de pollution lumineuse.",
        "tags": ["Désert", "Aventure", "Étoiles", "Dromadaire", "Bivouac"],
        "highlights": ["Lever de soleil sur les dunes", "Balade à dromadaire", "Bivouac de luxe", "Village Khamlia (musique Gnawa)", "Sandboarding"],
        "pour_qui": "Aventuriers, couples romantiques, familles, photographes",
        "a_ne_pas_manquer": "La nuit en bivouac sous la Voie Lactée à l'œil nu — Sahara = 0% pollution lumineuse.",
        "transport_interne": "4x4, dromadaire, marche dans les dunes",
        "niveau_visite": "Facile (dunes accessibles)",
        "securite": "Très bonne — guides sahariens expérimentés",
        "langue": "Tamazight, Arabe Hassanya, Darija",
        "monnaie": "MAD — cash uniquement dans le désert",
        "classification_raison": "Destination Nature-désert pour tous profils avec budget moyen à élevé (bivouac de luxe)"
    },
    "Ifrane": {
        "emoji": "🌲",
        "description": "La «Suisse du Maroc» à 1 665 m d'altitude : maisons à toits pointus rouges, forêts de cèdres millénaires et neige abondante en hiver. Ville la plus propre d'Afrique selon l'ONU.",
        "tags": ["Nature", "Ski", "Famille", "Detente", "Forêt"],
        "highlights": ["Forêt de cèdres (singes Magot)", "Station de ski Michlifen", "Lac Dayet Aoua", "Lion d'Ifrane (sculpture)", "Campus Université Al Akhawayn"],
        "pour_qui": "Familles, amants de la nature, skieur débutants",
        "a_ne_pas_manquer": "Les singes Magot en liberté dans la cédraie d'Azrou — ils viennent manger dans la main des visiteurs.",
        "transport_interne": "Marche, taxi, location de ski sur place",
        "niveau_visite": "Facile",
        "securite": "Excellente — ville universitaire très sûre",
        "langue": "Darija, Français, Amazigh",
        "monnaie": "MAD — distributeurs disponibles",
        "classification_raison": "Destination Nature-famille accessible pour tous budgets, idéale hiver comme été"
    },
    "Vallée du Draa": {
        "emoji": "🌴",
        "description": "La plus longue oasis du Maroc (200 km) : palmeraies de 10 millions de dattiers, kasbahs médiévales en pisé rouge et villages ksar. Route des kasbahs entre Ouarzazate et Zagora.",
        "tags": ["Oasis", "Kasbahs", "Desert", "Histoire", "Palmiers"],
        "highlights": ["Palmeraie Agdez-Zagora (200 km)", "Kasbah Tamnougalt", "Ksar d'Aït Benhaddou (UNESCO)", "Zagora & désert de l'Iriqui", "Dunes de Tinfou"],
        "pour_qui": "Amateurs d'histoire, photographes, aventuriers en voiture",
        "a_ne_pas_manquer": "Le lever du soleil depuis le sommet de la dune de Tinfou — mer de palmeraies à l'infini et kasbahs rouges.",
        "transport_interne": "Voiture de location (route des kasbahs), taxi collectif, chameau",
        "niveau_visite": "Modéré (distances importantes)",
        "securite": "Très bonne",
        "langue": "Tamazight Amazigh, Arabe, Darija",
        "monnaie": "MAD — cash uniquement dans les villages",
        "classification_raison": "Destination Nature-désert pour profils culturels et aventuriers avec budget modéré"
    },
    # ── NOUVELLES VILLES ────────────────────────────────────
    "Meknès": {
        "emoji": "🏰",
        "description": "Surnommée l'Ismaïlia, Meknès fut la capitale impériale du Sultan Moulay Ismail au XVIIe siècle. Ses remparts monumentaux, ses portes triomphales et son écurie royale de 12 000 chevaux témoignent d'une grandeur passée.",
        "tags": ["Patrimoine", "Histoire", "Culture", "Famille"],
        "highlights": ["Bab Mansour (plus belle porte du Maroc)", "Mausolée Moulay Ismail", "Heri es-Souani (greniers royaux)", "Médina (UNESCO)", "Volubilis (cité romaine, 5 km)"],
        "pour_qui": "Historiens, familles, voyageurs culturels",
        "a_ne_pas_manquer": "Le site romain de Volubilis à 5 km — mosaïques, arcs de triomphe et colonnes à la lumière du coucher de soleil.",
        "transport_interne": "Calèche (30 MAD), petit taxi (10-20 MAD), marche",
        "niveau_visite": "Facile à modéré",
        "securite": "Très bonne — ville tranquille",
        "langue": "Darija Mekhnassia, Français",
        "monnaie": "MAD — distributeurs disponibles",
        "classification_raison": "Destination Ville impériale pour profils culturels avec budget modéré"
    },
    "Tétouan": {
        "emoji": "🕊️",
        "description": "Joyau andalou du Nord : les Maures expulsés d'Espagne rebâtirent ici une Grenade marocaine en 1492. Sa médina hispano-mauresque classée UNESCO est la mieux conservée du Maroc.",
        "tags": ["Patrimoine", "Architecture", "Culture", "UNESCO"],
        "highlights": ["Médina hispano-mauresque (UNESCO)", "Musée Archéologique", "Quartier Español", "Plage Martil (5 km)", "École d'Art et d'Artisanat"],
        "pour_qui": "Amateurs d'histoire andalouse, familles, voyageurs culturels",
        "a_ne_pas_manquer": "La médina au petit matin — ses maisons blanches à encadrements verts révèlent une architecture unique en Afrique.",
        "transport_interne": "Marche (médina compacte), taxi (10-20 MAD), bus",
        "niveau_visite": "Facile",
        "securite": "Bonne",
        "langue": "Darija, Espagnol, Français",
        "monnaie": "MAD — distributeurs disponibles",
        "classification_raison": "Destination Ville culturelle pour profils patrimoniaux avec budget modéré"
    },
    "El Jadida": {
        "emoji": "🏰",
        "description": "Ancienne cité portugaise Mazagan classée UNESCO, El Jadida offre une citerne souterraine mystérieuse du XVIe siècle, des remparts atlantiques et des plages de sable fin à deux heures de Casablanca.",
        "tags": ["Patrimoine", "Plage", "UNESCO", "Histoire"],
        "highlights": ["Citerne Portugaise (UNESCO)", "Remparts de Mazagan", "Plage El Jadida", "Sidi Bouzid (coucher de soleil)", "Médina fortifiée"],
        "pour_qui": "Familles, amateurs d'histoire, voyageurs côtiers",
        "a_ne_pas_manquer": "La citerne portugaise — voûtes gothiques reflétées dans quelques centimètres d'eau, photo d'Orson Welles pour Othello.",
        "transport_interne": "Marche (médina), taxi (15-25 MAD), bus",
        "niveau_visite": "Facile",
        "securite": "Très bonne",
        "langue": "Darija, Français",
        "monnaie": "MAD — distributeurs disponibles",
        "classification_raison": "Destination Ville-plage pour profils culturels et familles avec budget modéré"
    },
    "Safi": {
        "emoji": "🏺",
        "description": "Capitale mondiale de la poterie artisanale, Safi est une ville de pêcheurs aux falaises atlantiques spectaculaires. Son quartier des potiers, actif depuis des siècles, produit les céramiques les plus renommées du Maroc.",
        "tags": ["Artisanat", "Culture", "Mer", "Gastronomie"],
        "highlights": ["Quartier des potiers (Colline des Potiers)", "Musée National de la Céramique", "Kechla (citadelle portugaise)", "Port de pêche artisanal", "Plage Sidi Bouzid"],
        "pour_qui": "Artisans, curieux, familles, amateurs de gastronomie",
        "a_ne_pas_manquer": "Les fours des potiers en activité — voir les artisans tourner et peindre les pièces comme leurs ancêtres au XIVe siècle.",
        "transport_interne": "Marche (médina), taxi (10-20 MAD)",
        "niveau_visite": "Facile",
        "securite": "Bonne",
        "langue": "Darija, Français",
        "monnaie": "MAD — distributeurs disponibles",
        "classification_raison": "Destination Ville artisanale pour profils culturels avec budget modéré"
    },
    "Oujda": {
        "emoji": "🌟",
        "description": "Carrefour de civilisations à la frontière algérienne, Oujda est la capitale du Raï marocain. Sa médina animée, ses influences franco-maghrébines et son accueil chaleureux en font une étape authentique de l'Oriental.",
        "tags": ["Musique", "Culture", "Authentique", "Oriental"],
        "highlights": ["Médina & Bab Sidi Abdelwahab", "Festival Rai (juillet)", "Parc Al Wifaq", "Sidi Yahya (pèlerinage)", "Cuisine oujdie (merguez, briouates)"],
        "pour_qui": "Amateurs de musique Raï, aventuriers hors des sentiers battus",
        "a_ne_pas_manquer": "Le Festival Raï en juillet — nuits de musique en plein air dans la capitale historique du genre.",
        "transport_interne": "Taxi (10-25 MAD), bus, marche",
        "niveau_visite": "Facile",
        "securite": "Bonne",
        "langue": "Darija Oujdia, Français",
        "monnaie": "MAD — distributeurs disponibles",
        "classification_raison": "Destination Ville authentique pour profils curieux avec petit budget"
    },
    "Nador": {
        "emoji": "🌊",
        "description": "Ville rifaine sur la côte méditerranéenne, Nador possède la lagune de Marchica, un plan d'eau quasi-fermé de 25 km sur 7 km. Ville en plein développement touristique avec plages et gastronomie de mer.",
        "tags": ["Mer", "Plage", "Lagune", "Gastronomie"],
        "highlights": ["Lagune de Marchica (25 km)", "Cap des Trois Fourches", "Plages méditerranéennes", "Nador Marina", "Poisson frais du port"],
        "pour_qui": "Familles, amateurs de mer, voyageurs authentiques",
        "a_ne_pas_manquer": "Une excursion en bateau sur la lagune de Marchica — eau turquoise et montagnes du Rif en toile de fond.",
        "transport_interne": "Taxi (10-25 MAD), bus, bateau sur la lagune",
        "niveau_visite": "Facile",
        "securite": "Bonne",
        "langue": "Tarifit (Rifain), Darija, Espagnol",
        "monnaie": "MAD — distributeurs disponibles",
        "classification_raison": "Destination Ville côtière pour familles et amateurs de mer avec budget modéré"
    },
    "Laâyoune": {
        "emoji": "🌅",
        "description": "Capitale du Sahara marocain, Laâyoune surprend par sa modernité, ses larges boulevards ensoleillés et son authenticité saharienne. Porte du grand sud, à mi-chemin entre le Maroc et la Mauritanie.",
        "tags": ["Désert", "Authentique", "Culture Hassanie", "Grand Sud"],
        "highlights": ["Place Mechouar (centre névralgique)", "Mosquée Moulay Abdel Aziz", "Plage Foum El Oued (25 km)", "Musée Régional du Sahara", "Artisanat hassani"],
        "pour_qui": "Aventuriers, explorateurs, amateurs de cultures sahariennes",
        "a_ne_pas_manquer": "Une soirée de musique hassanie autour du thé saharien — trois verres rituels, poésie du désert.",
        "transport_interne": "Taxi (15-30 MAD), bus",
        "niveau_visite": "Facile",
        "securite": "Bonne — ville tranquille",
        "langue": "Hassania, Darija, Français",
        "monnaie": "MAD — distributeurs disponibles",
        "classification_raison": "Destination Ville saharienne pour explorateurs et aventuriers avec budget modéré"
    },
    "Dakhla": {
        "emoji": "🏄",
        "description": "Paradis mondial du kitesurf et du windsurf, Dakhla est une péninsule encerclée par une lagune aux eaux turquoise sur 40 km. Poissons ultra-frais, dunes de sable et couchers de soleil mémorables.",
        "tags": ["Kitesurf", "Plage", "Aventure", "Gastronomie", "Lagune"],
        "highlights": ["Lagune Dakhla (40 km, spot mondial kitesurf)", "Baie des Dunes (quad et sandboard)", "Poissons & homards frais", "Péninsule de Dakhla", "Couchers de soleil sur l'Atlantique"],
        "pour_qui": "Kitesurfeurs, amateurs de mer, aventuriers, couples",
        "a_ne_pas_manquer": "Le coucher de soleil sur la lagune avec des kitesurfs en vol — spectacle naturel à couper le souffle.",
        "transport_interne": "4x4, taxi, quad, bateau",
        "niveau_visite": "Facile (kitesurf = modéré)",
        "securite": "Excellente",
        "langue": "Hassania, Darija, Français",
        "monnaie": "MAD — distributeurs disponibles",
        "classification_raison": "Destination Ville-plage premium pour sportifs et aventuriers avec budget élevé"
    },
    "Béni Mellal": {
        "emoji": "🌿",
        "description": "Porte du Moyen Atlas et base idéale pour les Cascades d'Ouzoud, Béni Mellal est une ville verdoyante au pied du Jbel Tassemit. Son marché hebdomadaire et ses sources naturelles attirent les visiteurs en quête d'authenticité.",
        "tags": ["Nature", "Randonnée", "Authenticité", "Détente"],
        "highlights": ["Cascades d'Ouzoud (60 km)", "Source Ain Asserdoun", "Kasbah Ras el-Ain", "Marché lundi (souks)", "Forêt de la Montagne"],
        "pour_qui": "Familles, randonneurs, voyageurs authentiques",
        "a_ne_pas_manquer": "Les Cascades d'Ouzoud à 60 km — trois chutes de 110 m, singes magots et moulins berbères.",
        "transport_interne": "Taxi (15-30 MAD), bus, location voiture",
        "niveau_visite": "Facile",
        "securite": "Très bonne",
        "langue": "Tamazight, Darija, Français",
        "monnaie": "MAD — distributeurs disponibles",
        "classification_raison": "Destination Ville-nature pour familles avec budget modéré"
    },
    "Moulay Idriss": {
        "emoji": "✨",
        "description": "La ville sacrée du Maroc, berceau de l'Islam marocain. Fondée par Moulay Idriss 1er, arrière-petit-fils du Prophète, cette cité blanche sur deux collines est un lieu de pèlerinage intense et de sérénité mystique.",
        "tags": ["Spiritualité", "Pèlerinage", "Histoire", "Mystique"],
        "highlights": ["Mausolée de Moulay Idriss I", "Vue panoramique sur les deux collines", "Médina circulaire unique", "Moussem estival (festival)", "Volubilis (5 km)"],
        "pour_qui": "Voyageurs spirituels, historiens, pèlerins, curieux",
        "a_ne_pas_manquer": "Le Moussem de Moulay Idriss en été — pèlerinage national avec fantasias, musique et prières collectives.",
        "transport_interne": "Marche (médina compacte), taxi",
        "niveau_visite": "Facile",
        "securite": "Excellente — ville sacrée respectée",
        "langue": "Darija, Français",
        "monnaie": "MAD",
        "classification_raison": "Destination Ville spirituelle pour tous profils avec petit budget"
    },
    # ── NOUVEAUX SITES NATURELS ──────────────────────────────
    "Gorges du Dadès": {
        "emoji": "🏔️",
        "description": "Frère des Gorges du Todra, le Dadès est encore plus sauvage et coloré. Ses falaises roses et rouges sculptées par l'oued, ses villages accrochés aux parois et ses doigts de singe (formations rocheuses) composent un tableau époustouflant.",
        "tags": ["Canyon", "4x4", "Randonnée", "Paysage", "Villages Berbères"],
        "highlights": ["Doigts de Singe (formations rocheuses)", "Village de Aït Arbi", "Gorges étroites du haut Dadès", "Palmeraie de Boumalne", "Route des 1000 Kasbahs"],
        "pour_qui": "Aventuriers, photographes, amateurs de 4x4, randonneurs",
        "a_ne_pas_manquer": "Les 'Doigts de Singe' — formations rocheuses érodées en formes anthropomorphes, uniques au monde.",
        "transport_interne": "4x4, taxi depuis Boumalne (40 MAD), randonnée",
        "niveau_visite": "Modéré",
        "securite": "Bonne — guide local recommandé pour les gorges supérieures",
        "langue": "Tamazight, Darija",
        "monnaie": "MAD — cash uniquement",
        "classification_raison": "Destination Nature-aventure pour profils actifs avec budget modéré"
    },
    "Cascades d'Ouzoud": {
        "emoji": "💧",
        "description": "Les plus belles cascades du Maroc et d'Afrique du Nord : trois chutes d'eau de 110 m tombant dans un bassin d'émeraude. Singes magots joueurs, moulins à eau berbères et arc-en-ciel permanent composent un décor de conte.",
        "tags": ["Cascades", "Singes", "Nature", "Randonnée", "Baignade"],
        "highlights": ["Chutes principales (110 m, 3 cascades)", "Singes magots (en liberté)", "Moulins à eau berbères du XVIe", "Baignade dans le bassin", "Vue panoramique depuis les falaises"],
        "pour_qui": "Familles, photographes, randonneurs, amateurs de nature",
        "a_ne_pas_manquer": "L'arc-en-ciel permanent au-dessus des chutes à 11h — les embruns créent un double arc en lumière dorée.",
        "transport_interne": "Taxi depuis Béni Mellal ou Azilal, marche sur site",
        "niveau_visite": "Facile",
        "securite": "Bonne — attention rochers glissants",
        "langue": "Tamazight, Darija",
        "monnaie": "MAD — cash",
        "classification_raison": "Destination Nature-famille accessible pour tous profils avec budget modéré"
    },
    "Plage de Legzira": {
        "emoji": "🌊",
        "description": "Plage secrète de l'Anti-Atlas aux arches de grès rouge sculptées par l'Atlantique. Avant qu'une arche ne s'effondre en 2016, c'était l'une des plages les plus photographiées du monde. L'arche restante reste un spectacle rare.",
        "tags": ["Plage", "Arche Naturelle", "Photographie", "Solitude"],
        "highlights": ["Arche naturelle de grès rouge", "Plage sauvage et préservée", "Coucher de soleil sur l'Atlantique", "Accès à pied depuis Sidi Ifni", "Pêche artisanale locale"],
        "pour_qui": "Photographes, couples, solitaires en quête d'authenticité",
        "a_ne_pas_manquer": "Le coucher de soleil sous l'arche rouge — lumière orange sur grès rouge et mer turquoise, image iconique du Maroc.",
        "transport_interne": "Taxi depuis Sidi Ifni (20 MAD), marche sur la plage",
        "niveau_visite": "Facile — attention aux marées",
        "securite": "Bonne — consulter les horaires de marée obligatoirement",
        "langue": "Tamazight Chleuh, Darija",
        "monnaie": "MAD — cash",
        "classification_raison": "Destination Nature-plage pour profils contemplatifs avec budget modéré"
    },
    "Cap Spartel": {
        "emoji": "⚡",
        "description": "Pointe nord-ouest de l'Afrique, là où l'Atlantique rencontre la Méditerranée. Le phare historique de 1864 surplombe une falaise spectaculaire, et par temps clair, on aperçoit les côtes espagnoles à 14 km.",
        "tags": ["Vue Panoramique", "Phare", "Histoire", "Atlantique"],
        "highlights": ["Phare de Cap Spartel (1864)", "Vue sur l'Espagne (14 km)", "Rencontre Atlantique-Méditerranée", "Plage Robinson", "Grottes d'Hercule (3 km)"],
        "pour_qui": "Voyageurs en quête de points de vue uniques, photographes, curieux",
        "a_ne_pas_manquer": "L'aube au phare — regarder les deux mers de couleurs différentes se rejoindre au premier soleil.",
        "transport_interne": "Taxi depuis Tanger (50 MAD), voiture recommandée",
        "niveau_visite": "Facile",
        "securite": "Très bonne",
        "langue": "Darija, Espagnol, Français",
        "monnaie": "MAD",
        "classification_raison": "Destination Nature-panoramique pour tous profils avec budget modéré"
    },
    "Grottes d'Hercule": {
        "emoji": "🗿",
        "description": "Cavités marines naturelles habitées depuis le Néolithique, où la légende place les travaux d'Hercule. La fenêtre côté mer, vue de l'intérieur, dessine parfaitement la silhouette du continent africain.",
        "tags": ["Mythologie", "Géologie", "Histoire", "Photographie"],
        "highlights": ["Fenêtre 'Carte de l'Afrique'", "Salles préhistoriques (outils néolithiques)", "Extraction des meules millénaires", "Vue sur l'Atlantique", "Plage adjacente"],
        "pour_qui": "Curieux, familles, amateurs de mythologie, photographes",
        "a_ne_pas_manquer": "La fenêtre sur mer vue de l'intérieur — la lumière du soleil découpe parfaitement la carte de l'Afrique dans la roche.",
        "transport_interne": "Taxi depuis Tanger (50 MAD), voiture",
        "niveau_visite": "Facile",
        "securite": "Bonne — sols glissants, chaussures fermées",
        "langue": "Darija, Espagnol, Français",
        "monnaie": "MAD — entrée payante (15 MAD)",
        "classification_raison": "Destination Nature-mythologie pour tous profils avec petit budget"
    },
    "Vallée du Ziz": {
        "emoji": "🌴",
        "description": "La plus grande palmeraie au monde traversée par un fleuve intermittent, entre les gorges du Ziz au nord et Erfoud au sud. 800 000 palmiers dattiers, kasbahs de terre rouge et ciel immaculé.",
        "tags": ["Oasis", "Palmeraie", "4x4", "Désert", "Kasbahs"],
        "highlights": ["Tunnel du Légionnaire (vue panoramique)", "Palmeraie de Erfoud-Rich (800 km²)", "Gorges du Ziz", "Ksar El Fida (village fortifié)", "Fossiles de Erfoud"],
        "pour_qui": "Aventuriers, photographes, amateurs de désert",
        "a_ne_pas_manquer": "La vue depuis le Tunnel du Légionnaire — mer de palmiers à perte de vue, kasbahs et montagnes de l'Atlas en fond.",
        "transport_interne": "Voiture ou 4x4, taxi collectif",
        "niveau_visite": "Modéré",
        "securite": "Bonne — chaleur extrême en été",
        "langue": "Tamazight, Darija",
        "monnaie": "MAD — cash",
        "classification_raison": "Destination Nature-désert pour aventuriers avec budget modéré"
    },
    "Forêt de Cèdres d'Azrou": {
        "emoji": "🌲",
        "description": "Cédraie millénaire du Moyen Atlas à 1 250 m d'altitude : cèdres de l'Atlas de 40 m et 600 ans d'âge, singes magots en totale liberté et air pur de montagne. À 20 km d'Ifrane.",
        "tags": ["Forêt", "Singes", "Montagne", "Randonnée", "Nature"],
        "highlights": ["Singes magots en liberté (contact direct)", "Cèdres millénaires de 40 m", "Sentiers forestiers balisés", "Village d'Azrou (poterie berbère)", "Lac Afourgagh (proche)"],
        "pour_qui": "Familles, amants des animaux, randonneurs",
        "a_ne_pas_manquer": "Nourrir les singes magots à la main — ils viennent directement chercher les noix et les fruits.",
        "transport_interne": "Voiture ou taxi depuis Azrou (20 MAD), marche en forêt",
        "niveau_visite": "Facile",
        "securite": "Bonne — singes inoffensifs si on ne les provoque pas",
        "langue": "Tamazight, Darija",
        "monnaie": "MAD — cash",
        "classification_raison": "Destination Nature-famille accessible pour tous profils avec budget modéré"
    },
    "Cirque de Jaffar": {
        "emoji": "🏔️",
        "description": "Amphithéâtre naturel volcanique à 2 000 m d'altitude dans le Moyen Atlas : paysage lunaire rouge et ocre, forêts de genévriers et silence absolu. L'un des secrets les mieux gardés du Maroc.",
        "tags": ["Aventure", "4x4", "Isolement", "Paysage Unique", "Camping"],
        "highlights": ["Cirque volcanique (5 km de diamètre)", "Falaises de basalte rouge", "Forêts de genévriers centenaires", "Bivouac sous les étoiles", "Randonnée sportive"],
        "pour_qui": "Aventuriers aguerris, campeurs, photographes de paysages",
        "a_ne_pas_manquer": "Le bivouac la nuit dans le cirque — silence total, ciel étoilé et lever de soleil sur les falaises rouges.",
        "transport_interne": "4x4 indispensable, guide local obligatoire",
        "niveau_visite": "Très difficile — zone isolée",
        "securite": "Bonne avec guide — ne jamais y aller seul",
        "langue": "Tamazight, Darija",
        "monnaie": "MAD — cash uniquement",
        "classification_raison": "Destination Nature-aventure extrême pour profils expérimentés avec budget modéré"
    },
    "Iles Purpuraires": {
        "emoji": "🏝️",
        "description": "Archipel de 6 îlots au large d'Essaouira, habitat mondial du rarissime Faucon d'Éléonore. Site romain de Mogador, ruines phéniciennes et eaux cristallines. Site protégé accessible en excursion.",
        "tags": ["Ornithologie", "Histoire", "Mer", "Nature Protégée"],
        "highlights": ["Faucon d'Éléonore (espèce rare, nicheur)", "Ruines romaines de Mogador", "Plongée et snorkeling", "Phoques moines (rare)", "Vue d'Essaouira depuis la mer"],
        "pour_qui": "Ornithologues, plongeurs, amateurs d'histoire, naturalistes",
        "a_ne_pas_manquer": "Observer le Faucon d'Éléonore en vol — espèce rare qui ne niche que dans quelques sites au monde.",
        "transport_interne": "Excursion bateau depuis le port d'Essaouira",
        "niveau_visite": "Facile (en bateau)",
        "securite": "Bonne — site protégé, respecter les règles",
        "langue": "Darija, Français",
        "monnaie": "MAD — excursion inclus repas",
        "classification_raison": "Destination Nature-maritime pour naturalistes et aventuriers avec budget modéré"
    },
}


# ============================================================
# CHARGEMENT DONNÉES VILLES / NATURE
# ============================================================
try:
    villes_df = pd.read_csv("villes.csv", encoding='utf-8-sig')
    nature_df = pd.read_csv("nature.csv", encoding='utf-8-sig')
    print(f"[OK] Données chargées: {len(villes_df)} villes, {len(nature_df)} sites nature")
    DATA_READY = True
except Exception as e:
    print(f"[!]  Erreur données: {e}")
    DATA_READY = False


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def nettoyer_info(info_dict):
    """Nettoyer les données d'une destination"""
    colonnes_multi = ["activites", "conseils", "duree_visite", "meilleure_saison",
                      "equipement", "securite", "animaux", "attractions",
                      "restaurants", "specialites", "dialecte", "fetes",
                      "artisanat", "musique", "habitudes"]

    for col in colonnes_multi:
        if col in info_dict and pd.notna(info_dict[col]) and str(info_dict[col]).strip():
            info_dict[col] = str(info_dict[col]).replace(";", ", ")
        else:
            info_dict[col] = ""

    if "images" in info_dict and pd.notna(info_dict["images"]):
        info_dict["images"] = [img.strip() for img in str(info_dict["images"]).split(";") if img.strip()]
    else:
        info_dict["images"] = []

    # Convertir les NaN en chaînes vides
    for key in info_dict:
        if isinstance(info_dict[key], float) and pd.isna(info_dict[key]):
            info_dict[key] = ""

    return info_dict


def predire_destination(age, sexe, budget, marie, region):
    """Prédire le type de destination avec le modèle ML"""
    if ML_READY:
        try:
            valeurs = {
                "âge": age,
                "sexe": le_sexe.transform([sexe])[0],
                "budget": budget,
                "marié": le_marie.transform([marie])[0],
                "région_origine": le_region.transform([region])[0]
            }
            df_nouveau = pd.DataFrame([valeurs])
            df_scaled = scaler.transform(df_nouveau)
            prediction = model.predict(df_scaled)[0]
            return le_destination.inverse_transform([prediction])[0].strip()
        except Exception as e:
            print(f"Erreur prédiction ML: {e}")

    # Fallback: logique simplifiée
    if budget >= 5000 and age > 30:
        return "Ville"
    elif budget < 3500:
        return "Nature"
    else:
        return "Nature" if marie == "Non" else "Ville"


def obtenir_lieux(type_dest, budget):
    """Obtenir les lieux filtrés par type et budget"""
    if not DATA_READY:
        return []

    lieux = []
    df_source = villes_df if type_dest == "Ville" else nature_df

    for _, row in df_source.iterrows():
        try:
            # Nettoyage robuste du budget (supprimer espaces, virgules)
            budget_str = str(row["budget"]).replace(',', '').strip()
            row_budget = int(budget_str) if budget_str else 0
        except:
            continue
        if budget >= row_budget:
            info = nettoyer_info(row.to_dict())
            lieux.append(info)

    return lieux


def login_required(f):
    """Décorateur pour protéger les routes"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth_page'))
        return f(*args, **kwargs)
    return decorated


# ============================================================
# ROUTES - PAGES HTML
# ============================================================

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')


@app.route('/auth')
def auth_page():
    """Page de connexion/inscription"""
    return render_template('auth.html')


@app.route('/user')
def user_page():
    """Dashboard utilisateur"""
    if 'user_id' not in session:
        return redirect(url_for('auth_page'))
    uid = session['user_id']
    conn = get_db()
    user  = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    dests = conn.execute(
        "SELECT id, nom, type, emoji, description, budget, tags, created_at "
        "FROM saved_destinations WHERE user_id = ? ORDER BY created_at DESC", (uid,)
    ).fetchall()
    groups = conn.execute(
        "SELECT id, group_id, group_name, destination, created_at "
        "FROM saved_groups WHERE user_id = ? ORDER BY created_at DESC", (uid,)
    ).fetchall()
    plans = conn.execute(
        "SELECT id, destination, duree, voyageurs, budget_utilisateur, created_at "
        "FROM trip_plans WHERE user_id = ? ORDER BY created_at DESC", (uid,)
    ).fetchall()
    comments = conn.execute(
        "SELECT * FROM comments ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    conn.close()
    import json as _json
    dests_list  = []
    for r in dests:
        d = dict(r)
        try: d['tags'] = _json.loads(d.get('tags') or '[]')
        except: d['tags'] = []
        dests_list.append(d)
    return render_template('user.html',
        user     = dict(user) if user else None,
        saved_dests  = dests_list,
        saved_groups = [dict(r) for r in groups],
        saved_plans  = [dict(r) for r in plans],
        comments     = [dict(r) for r in comments],
    )


@app.route('/service1')
def service1_page():
    """Service 1 - Destination Intelligente"""
    return render_template('service1.html')


@app.route('/service2')
def service2_page():
    """Service 2 - Planificateur de Voyage"""
    return render_template('service2.html')


@app.route('/service3')
def service3_page():
    """Service 3 - Voyages en Groupe"""
    return render_template('service3.html')


@app.route('/admin')
def admin_page():
    """Page admin"""
    return render_template('admin.html')


# ============================================================
# API - AUTHENTIFICATION
# ============================================================

@app.route('/api/register', methods=['POST'])
def api_register():
    """Inscription d'un nouvel utilisateur"""
    data = request.get_json()

    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not name or not email or not password:
        return jsonify({"success": False, "message": "Tous les champs sont obligatoires"}), 400

    if '@' not in email or '.' not in email.split('@')[-1]:
        return jsonify({"success": False, "message": "Adresse email invalide"}), 400

    if len(password) < 6:
        return jsonify({"success": False, "message": "Le mot de passe doit contenir au moins 6 caractères"}), 400

    if len(name) < 2:
        return jsonify({"success": False, "message": "Le nom doit contenir au moins 2 caractères"}), 400

    hashed_password = generate_password_hash(password)

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (name, email, password, created_at) VALUES (?, ?, ?, ?)",
            (name, email, hashed_password, datetime.now().isoformat())
        )
        conn.commit()
        print(f"[OK] Nouvel utilisateur: {name} ({email})")
        return jsonify({"success": True, "message": "Inscription réussie!"})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Cet email est déjà utilisé"}), 400
    finally:
        conn.close()


@app.route('/api/login', methods=['POST'])
def api_login():
    """Connexion utilisateur"""
    data = request.get_json()
    email = data.get('email', '').strip()
    password = data.get('password', '')

    # ✅ AMÉLIORÉ: Vérification du hash avec bcrypt
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if user and check_password_hash(user['password'], password):
        session.clear()  # Prévention session fixation
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        session['user_email'] = user['email']
        return jsonify({
            "success": True,
            "message": f"Bienvenue {user['name']}!",
            "user": {"id": user['id'], "name": user['name'], "email": user['email']}
        })

    return jsonify({"success": False, "message": "Email ou mot de passe incorrect"}), 401


@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Déconnexion"""
    session.clear()
    return jsonify({"success": True, "message": "Déconnexion réussie"})


@app.route('/api/current-user')
def api_current_user():
    """Obtenir l'utilisateur connecté"""
    if 'user_id' in session:
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE id = ?", (session['user_id'],)).fetchone()
        conn.close()
        if user:
            return jsonify({
                "logged_in": True,
                "user": {
                    "id": user['id'],
                    "name": user['name'],
                    "email": user['email'],
                    "destinations_visited": user['destinations_visited'],
                    "trips_planned": user['trips_planned'],
                    "groups_joined": user['groups_joined']
                }
            })
    return jsonify({"logged_in": False})


@app.route('/api/forgot-password', methods=['POST'])
def api_forgot_password():
    """Récupération mot de passe"""
    data = request.get_json()
    email = data.get('email', '').strip()

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    # ✅ AMÉLIORÉ: Ne jamais exposer le mot de passe haché
    if user:
        return jsonify({
            "success": True,
            "message": "Un email de réinitialisation a été envoyé. (Demo)"
        })
    return jsonify({"success": False, "message": "Aucun compte trouvé avec cet email"}), 404


# ============================================================
# API - SERVICE 1: PRÉDICTION ML
# ============================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Prédire la destination et retourner les lieux"""
    data = request.get_json()

    try:
        age = int(data['age'])
        sexe = data['sexe']
        budget = int(data['budget'])
        marie = data['marie']
        region = data['region']
    except (KeyError, ValueError) as e:
        return jsonify({"success": False, "message": f"Données invalides: {e}"}), 400

    if not (10 <= age <= 110):
        return jsonify({"success": False, "message": "Âge invalide (10–110)"}), 400
    if budget < 500:
        return jsonify({"success": False, "message": "Budget minimum: 500 MAD"}), 400
    if sexe not in ('Homme', 'Femme'):
        return jsonify({"success": False, "message": "Sexe invalide"}), 400
    if marie not in ('Oui', 'Non'):
        return jsonify({"success": False, "message": "Situation matrimoniale invalide"}), 400

    # Prédiction
    type_dest = predire_destination(age, sexe, budget, marie, region)

    # Obtenir lieux + enrichir avec ENRICHISSEMENT_LIEUX
    lieux = obtenir_lieux(type_dest, budget)
    for lieu in lieux:
        extra = ENRICHISSEMENT_LIEUX.get(lieu.get('nom', ''), {})
        lieu.update(extra)

    # Raisons de classification basées sur le profil
    raisons = []
    if type_dest == "Ville":
        if budget >= 6000:
            raisons.append("💰 Budget élevé → destinations urbaines premium accessibles")
        elif budget >= 3500:
            raisons.append("💰 Budget moyen → villes culturelles bien desservies")
        if age >= 35:
            raisons.append("🎂 Profil adulte → préférence confort urbain et gastronomie")
        if marie == "Oui":
            raisons.append("💑 En couple/famille → activités culturelles variées en ville")
        if region in ("France", "Belgique", "Suisse", "Royaume-Uni", "USA", "Canada"):
            raisons.append("✈️ Voyageur occidental → infrastructure touristique urbaine adaptée")
    else:
        if budget < 4000:
            raisons.append("💰 Budget accessible → nature & aventure, peu d'infrastructure coûteuse")
        if age < 35:
            raisons.append("🎂 Profil jeune → préférence aventure, sport et découverte")
        if marie == "Non":
            raisons.append("🎒 Célibataire → liberté de mouvement pour randonnées et treks")
        if sexe == "Homme":
            raisons.append("🏃 Profil masculin → activités physiques et aventure privilégiées")
        if region in ("Maroc", "Tunisie", "Égypte"):
            raisons.append("🌍 Voyageur régional → destinations nature du Maghreb bien connues")
    if not raisons:
        raisons.append(f"🤖 Modèle Decision Tree ({ML_ACCURACY}% précision) → profil global analysé")

    # Incrémenter stats utilisateur
    if 'user_id' in session:
        conn = get_db()
        try:
            conn.execute(
                "UPDATE users SET destinations_visited = destinations_visited + 1 WHERE id = ?",
                (session['user_id'],)
            )
            conn.commit()
        finally:
            conn.close()

    return jsonify({
        "success": True,
        "type_destination": type_dest,
        "ml_used": ML_READY,
        "ml_accuracy": ML_ACCURACY,
        "raisons_classification": raisons,
        "profil": {"age": age, "sexe": sexe, "budget": budget, "marie": marie, "region": region},
        "lieux": lieux,
        "count": len(lieux)
    })


# ============================================================
# API - SERVICE 2: PLANIFICATEUR DE VOYAGE
# ============================================================

# ─────────────────────────────────────────────────────────────
#  ALIAS — normalise les valeurs du <select> vers les clés DB
# ─────────────────────────────────────────────────────────────
DEST_ALIASES = {
    "Dades":       "Gorges du Dadès",
    "Ouzoud":      "Cascades d'Ouzoud",
    "Legzira":     "Plage de Legzira",
    "Spartel":     "Cap Spartel",
    "Hercule":     "Grottes d'Hercule",
    "Ziz":         "Vallée du Ziz",
    "Azrou":       "Forêt de Cèdres d'Azrou",
    "Cirque":      "Cirque de Jaffar",
    "IlesP":       "Iles Purpuraires",
    "Toubkal":     "Parc national de Toubkal",
    "Draa":        "Vallée du Draa",
    "Oukaimeden":  "Oukaïmeden",
}


def generate_generic_plan(dest_name):
    """Génère un plan 3 jours depuis ENRICHISSEMENT_LIEUX quand PLANS_DEST ne couvre pas la destination"""
    enr = ENRICHISSEMENT_LIEUX.get(dest_name, {})
    highlights = enr.get("highlights", [f"Découverte de {dest_name}"])
    emoji      = enr.get("emoji", "📍")
    tags       = enr.get("tags", ["Culture"])
    desc       = enr.get("description", f"Explorez {dest_name}.")
    a_ne_pas   = enr.get("a_ne_pas_manquer", "")
    transport  = enr.get("transport_interne", "Taxi local")
    langue     = enr.get("langue", "Darija, Français")
    niveau     = enr.get("niveau_visite", "Modéré")

    _icons  = ["fa-map-marker-alt","fa-camera","fa-hiking","fa-binoculars",
               "fa-mountain","fa-utensils","fa-tree","fa-sun","fa-moon","fa-compass"]
    _heures = ["08:30","10:00","12:00","14:00","16:00","18:00","09:30","11:30","15:00","17:30"]

    # Construire des activités depuis les highlights
    acts = []
    for i, hl in enumerate(highlights[:8]):
        acts.append({
            "heure":       _heures[i % len(_heures)],
            "icon":        _icons[i % len(_icons)],
            "titre":       hl,
            "description": (desc[:120] if i == 0 else f"{hl} — incontournable de {dest_name}."),
            "cout":        60 + i * 35,
            "tag":         tags[i % len(tags)]
        })

    # Ajouter activité repas si absente
    if not any("utensils" in a["icon"] for a in acts):
        acts.append({
            "heure":"13:00","icon":"fa-utensils",
            "titre":f"Déjeuner traditionnel — {dest_name}",
            "description":"Cuisine locale authentique, tajine et thé à la menthe.",
            "cout":80,"tag":"Gastronomie"
        })

    # Découper en journées de 3-4 activités
    jours = []
    titres_jours = ["Découverte & Arrivée", "Exploration approfondie", "Nature & Culture", "Plein air & Traditions"]
    chunk = 3
    for j, start in enumerate(range(0, len(acts), chunk)):
        jours.append({
            "titre":     titres_jours[j % len(titres_jours)],
            "activites": acts[start:start+chunk]
        })
    if not jours:
        jours = [{"titre": f"Découverte de {dest_name}", "activites": acts}]

    return {
        "emoji":             emoji,
        "slogan":            enr.get("classification_raison", f"Destination incontournable — {dest_name}")[:90],
        "hotel":             {"nom": f"Maison d'hôtes — {dest_name}", "etoiles": 3,
                              "prix_nuit": 320, "quartier": "Centre"},
        "transport_arrivee": transport,
        "budget_jour":       {"hebergement": 320, "repas": 150, "activites": 130, "transport": 70},
        "conseils": [
            a_ne_pas or f"À ne pas manquer: {highlights[0] if highlights else dest_name}",
            f"Transport sur place: {transport}",
            f"Langues: {langue}",
            f"Niveau de visite: {niveau}",
        ],
        "jours": jours,
    }


# ─────────────────────────────────────────────────────────────
#  DONNÉES RICHES PAR DESTINATION  (plans jour par jour)
# ─────────────────────────────────────────────────────────────
PLANS_DEST = {
    "Marrakech": {
        "emoji": "🕌", "type": "Ville",
        "slogan": "La Ville Rouge — Joyau du Sud",
        "hotel": {"nom": "Riad Yasmine", "etoiles": 4, "prix_nuit": 450, "quartier": "Médina"},
        "transport_arrivee": "Taxi aéroport ↔ médina : 150 MAD",
        "budget_jour": {"hebergement": 450, "repas": 220, "activites": 180, "transport": 80},
        "conseils": [
            "Négociez toujours les prix dans les souks (divisez par 3 le prix annoncé)",
            "Visitez la médina avant 9h pour éviter la foule",
            "Portez de la crème solaire — il fait souvent plus de 35°C en été",
            "Habillez-vous modestement lors des visites de mosquées"
        ],
        "jours": [
            {
                "titre": "Arrivée & Cœur de la Médina",
                "activites": [
                    {"heure":"09:00","icon":"fa-sun","titre":"Place Jemaa el-Fna","description":"Plongez dans l'effervescence de la place mythique classée UNESCO : acrobates, charmeurs de serpents, calèches et vendeurs de jus d'orange frais à 4 MAD.","cout":20,"tag":"Culture"},
                    {"heure":"10:30","icon":"fa-mosque","titre":"Medersa Ben Youssef","description":"Chef-d'œuvre mérinide du XIVe siècle. Zellige bleu-turquoise, stucs sculptés, bois de cèdre gravé — l'un des plus beaux édifices d'Afrique du Nord.","cout":70,"tag":"Patrimoine"},
                    {"heure":"13:00","icon":"fa-utensils","titre":"Déjeuner au Café des Épices","description":"Savourez un tajine d'agneau aux pruneaux (85 MAD) sur la terrasse avec vue sur la Place Rahba Kedima, au milieu des sacs d'épices colorées.","cout":110,"tag":"Gastronomie"},
                    {"heure":"15:00","icon":"fa-store","titre":"Souks de la Médina","description":"Labyrinthe sensoriel : souk des ferronniers, des tanneurs, des tisserands. Rapportez de l'argan, des babouches en cuir et des lanternes en cuivre ciselé.","cout":350,"tag":"Shopping"},
                    {"heure":"19:30","icon":"fa-moon","titre":"Dîner Riad — Dar Yacout","description":"Festin royal dans un palais du XVIIe s. : pastilla au pigeon, tajine de kefta, cornes de gazelle et musique andalouse. Réservez à l'avance.","cout":400,"tag":"Gastronomie"},
                ]
            },
            {
                "titre": "Palais, Jardins & Art",
                "activites": [
                    {"heure":"09:00","icon":"fa-tree","titre":"Jardins Majorelle","description":"Oasis créée par Jacques Majorelle puis rachetée par Yves Saint-Laurent. Le bleu cobalt intense, les cactus géants et le musée berbère en font un lieu hors du temps.","cout":150,"tag":"Nature"},
                    {"heure":"11:00","icon":"fa-landmark","titre":"Palais Bahia","description":"8 hectares de jardins, cours pavées de zellige et salles ornées de plafonds en cèdre sculpté. Construit pour la favorite du Grand Vizir au XIXe siècle.","cout":70,"tag":"Patrimoine"},
                    {"heure":"13:00","icon":"fa-utensils","titre":"Déjeuner chez Nomad","description":"Cuisine marocaine contemporaine sur un rooftop avec vue à 360° sur la médina. Couscous au beurre smen, salade mechouia et limonade au gingembre.","cout":130,"tag":"Gastronomie"},
                    {"heure":"15:30","icon":"fa-spa","titre":"Hammam de la Rose","description":"Rituel purificateur en 4 étapes : savon beldi, gommage au kessa, masque au ghassoul, enveloppement au lait de rose. 90 minutes de bien-être absolu.","cout":280,"tag":"Bien-être"},
                    {"heure":"19:30","icon":"fa-music","titre":"Spectacle Jemaa el-Fna","description":"La place s'embrase à la nuit : 100 stands de cuisine de rue (harira, escargots, brochettes), musiciens Gnawa, fakirs et conteurs Halqa.","cout":60,"tag":"Culture"},
                ]
            },
            {
                "titre": "Histoire, Ruines & Coucher de Soleil",
                "activites": [
                    {"heure":"09:00","icon":"fa-palette","titre":"Musée de Marrakech","description":"Installé dans le splendide Palais Mnebhi, il réunit bronzes mérinides, calligraphies coraniques, céramiques fassis et bijoux berbères en argent.","cout":50,"tag":"Art"},
                    {"heure":"10:30","icon":"fa-tomb","titre":"Tombeaux Saadiens","description":"Nécropole royale redécouverte en 1917 : 66 tombeaux des sultans saadiens du XVIe s., ornés de mosaïques polychromes et colonnes de marbre de Carrare.","cout":70,"tag":"Patrimoine"},
                    {"heure":"13:00","icon":"fa-utensils","titre":"Cours de cuisine — Souk Cuisine","description":"Marché, achat des ingrédients avec le chef, puis préparation de la pastilla, du tagine et des briouates. Vous mangez ce que vous cuisinez !","cout":450,"tag":"Gastronomie"},
                    {"heure":"16:30","icon":"fa-fort-awesome","titre":"Palais El Badi","description":"Ruines grandioses du « palais des incomparables » (1578). Montez sur les remparts pour un panorama saisissant sur la médina et les montagnes de l'Atlas.","cout":70,"tag":"Patrimoine"},
                    {"heure":"19:30","icon":"fa-wine-glass","titre":"Dîner au Comptoir Darna","description":"L'adresse la plus festive de Marrakech : danseuses orientales, cocktails au jasmin et cuisine fusion sous les palmiers illuminés.","cout":500,"tag":"Gastronomie"},
                ]
            },
            {
                "titre": "Excursion Palmeraie & Vallée de l'Ourika",
                "activites": [
                    {"heure":"08:00","icon":"fa-horse","titre":"Palmeraie — Balade à dromadaire","description":"30 000 palmiers à 6 km du centre. Traversée en dromadaire ou en calèche à l'aube quand la lumière est dorée et la chaleur encore douce.","cout":200,"tag":"Nature"},
                    {"heure":"10:30","icon":"fa-mountain","titre":"Cascade d'Ourika — Vallée de l'Atlas","description":"45 min de route. Vallée verdoyante encaissée entre les sommets enneigés. Randonnée facile (2h) jusqu'aux cascades d'altitude à 1 800 m.","cout":150,"tag":"Nature"},
                    {"heure":"13:30","icon":"fa-utensils","titre":"Déjeuner berbère à Ourika","description":"Tajine de poulet fermier au citron confit et olives sur une terrasse suspendue au-dessus de la rivière. Vue imprenable sur les crêtes de l'Atlas.","cout":100,"tag":"Gastronomie"},
                    {"heure":"16:00","icon":"fa-seedling","titre":"Coopérative féminine Argane","description":"Rencontrez les femmes berbères qui extraient à la main l'huile d'argan. Dégustation d'amlou (beurre d'amande et argan) sur pain khobz.","cout":200,"tag":"Culture"},
                    {"heure":"19:00","icon":"fa-moon","titre":"Dîner sous les étoiles — Riad","description":"Retour en médina. Dîner intime dans le patio fleuri de votre riad aux chandelles : soupe harira, tajine de légumes, thé à la menthe glacé.","cout":300,"tag":"Gastronomie"},
                ]
            },
        ]
    },
    "Fès": {
        "emoji": "🏛️", "type": "Ville",
        "slogan": "La Capitale Spirituelle — Médina Vivante",
        "hotel": {"nom": "Riad Rcif", "etoiles": 4, "prix_nuit": 400, "quartier": "Fès el-Bali"},
        "transport_arrivee": "Taxi gare / aéroport ↔ médina : 80 MAD",
        "budget_jour": {"hebergement": 400, "repas": 180, "activites": 150, "transport": 60},
        "conseils": [
            "Engagez un guide officiel (agréé ONMT) — la médina a 9 400 ruelles !",
            "Les tanneries sentent fort — le vendeur vous offrira de la menthe",
            "Le Palais Royal ne se visite pas à l'intérieur, mais la façade vaut le détour",
            "Marché de Rcif le matin : épices, henné, argan, saffran et poterie fassi"
        ],
        "jours": [
            {
                "titre": "Fès el-Bali — La Plus Grande Médina du Monde",
                "activites": [
                    {"heure":"09:00","icon":"fa-archway","titre":"Bab Boujloud — Porte Bleue","description":"Porte d'entrée monumentale de la médina (1913) : zellige bleu côté Fès, vert côté Moulay Idriss. Point de départ idéal de votre exploration.","cout":0,"tag":"Patrimoine"},
                    {"heure":"10:00","icon":"fa-mosque","titre":"Medersa Bou Inania","description":"La seule madrasa de Fès ouverte au public pour la prière. Construite par les Mérinides (1351), elle est considérée comme la plus belle du Maghreb.","cout":50,"tag":"Patrimoine"},
                    {"heure":"13:00","icon":"fa-utensils","titre":"Déjeuner Dar Roumana","description":"Dans un riad du XIVe s. : bastilla aux fruits de mer, tajine de kefta et salade de bettrave à l'orange. Cuisine gastronomique fassi haut de gamme.","cout":200,"tag":"Gastronomie"},
                    {"heure":"15:00","icon":"fa-tint","titre":"Tanneries Chouara","description":"Plus grandes tanneries du monde médiéval : cuves de couleurs primaires (safran, pavot, menthe, indigo). Vue panoramique depuis les terrasses des magasins de cuir.","cout":80,"tag":"Culture"},
                    {"heure":"19:00","icon":"fa-moon","titre":"Dîner Fassi — Spécialités Ancestrales","description":"Maison Blanche : rfissa au poulet (vermicelles et lentilles), couscous au lait de brebis et pastilla sucrée. Musique andalouse en fond sonore.","cout":250,"tag":"Gastronomie"},
                ]
            },
            {
                "titre": "Art, Musées & Jardins",
                "activites": [
                    {"heure":"09:00","icon":"fa-palette","titre":"Musée Nejjarine — Arts du Bois","description":"Fondouk du XVIIIe siècle transformé en musée. Collections de menuiserie fassi, instruments de musique, coffrets en cèdre et portes sculptées.","cout":20,"tag":"Art"},
                    {"heure":"10:30","icon":"fa-university","titre":"Université al-Qaraouiyyin","description":"Fondée en 859 par Fatima al-Fihriya, c'est la plus ancienne université du monde encore en activité. La façade et la cour intérieure sont accessibles aux non-musulmans.","cout":0,"tag":"Patrimoine"},
                    {"heure":"13:00","icon":"fa-utensils","titre":"Déjeuner au Riad Rcif","description":"Vue spectaculaire sur le minaret de la Qaraouiyyin. Harira avec dates et chebakia, tajine de poulet aux citrons confits et couscous aux sept légumes.","cout":150,"tag":"Gastronomie"},
                    {"heure":"15:00","icon":"fa-tree","titre":"Jardin Jnan Sbil — Havre de Paix","description":"20 hectares de jardins andalous au cœur de la médina. Bassins, palmiers, cyprès et roses Damas. Idéal pour une pause après la foule des souks.","cout":10,"tag":"Nature"},
                    {"heure":"17:00","icon":"fa-store","titre":"Souk des Potiers — Poterie Fassi","description":"La poterie bleue de Fès est mondialement reconnue. Visitez les ateliers rue Tarik Ibn Ziad : tournage, cuisson au bois et peinture à la main.","cout":200,"tag":"Art"},
                    {"heure":"19:30","icon":"fa-music","titre":"Soirée Musique Andalouse","description":"Institut de Musique al-Basyir : concerts de musique classique andalouse fassi (malhoun, samâ) dans une salle du palais Dar Bennis.","cout":80,"tag":"Culture"},
                ]
            },
            {
                "titre": "Mérinides, Palais & Artisanat",
                "activites": [
                    {"heure":"09:00","icon":"fa-mountain","titre":"Tombeaux Mérinides & Vue sur Fès","description":"Ruines du XVe siècle sur la colline. La vue panoramique sur la médina depuis ce promontoire est la plus photographiée du Maroc au lever du soleil.","cout":0,"tag":"Patrimoine"},
                    {"heure":"10:30","icon":"fa-landmark","titre":"Palais Royal de Fès","description":"Façade monumentale de 10 000 m² ornée de 7 portails en bronze doré ciselé à la main. Symbole de la majesté royale alaouite.","cout":0,"tag":"Patrimoine"},
                    {"heure":"13:00","icon":"fa-utensils","titre":"Sandwich Msemen au marché","description":"Déjeuner populaire fassi : msemen (crêpe feuilletée) au kefta et œuf, jus d'orange et thé à la menthe — repas complet pour moins de 30 MAD.","cout":30,"tag":"Gastronomie"},
                    {"heure":"14:30","icon":"fa-hammer","titre":"Quartier des Forgeron — Souk Ain Allou","description":"Artisans qui forgent à la main : cuivres repoussés, candélabres, théières en maillechort. Spectacle fascinant de l'artisanat vivant.","cout":300,"tag":"Art"},
                    {"heure":"17:00","icon":"fa-mosque","titre":"Zawiya de Moulay Idriss II","description":"Mausolée du fondateur de Fès (IXe s.), cœur spirituel de la cité. Accessible aux non-musulmans dans le couloir extérieur — une expérience émouvante.","cout":0,"tag":"Culture"},
                    {"heure":"19:30","icon":"fa-moon","titre":"Dîner Maison Arabe","description":"Dans le palais de Dar Mnebhi : tanjia de bœuf (plat mythique de Fès), bastilla à la volaille et brochettes de kefta aux épices. Soirée de gala.","cout":350,"tag":"Gastronomie"},
                ]
            },
        ]
    },
    "Casablanca": {
        "emoji": "🌊", "type": "Ville",
        "slogan": "La Métropole Cosmopolite — Capitale Économique",
        "hotel": {"nom": "Hôtel Kenzi Tower", "etoiles": 5, "prix_nuit": 700, "quartier": "Quartier des Affaires"},
        "transport_arrivee": "Train aéroport ↔ Gare Centrale : 55 MAD (30 min)",
        "budget_jour": {"hebergement": 700, "repas": 300, "activites": 150, "transport": 100},
        "conseils": [
            "Le train Casa-Port / Casa-Voyageurs est le moyen le plus rapide (55 MAD)",
            "La Mosquée Hassan II : réservez la visite guidée à l'avance (130 MAD)",
            "Quartier Habous : shopping d'artisanat local moins cher qu'en médina",
            "Corniche Ain Diab le soir : restaurants de fruits de mer, bars et clubs"
        ],
        "jours": [
            {
                "titre": "Hassan II & Patrimoine Colonial",
                "activites": [
                    {"heure":"09:00","icon":"fa-mosque","titre":"Mosquée Hassan II","description":"3ème plus grande mosquée du monde (200 000 fidèles). Minaret de 210 m, sol en verre sur l'océan, hammam et fontaines monumentales. Visite guidée obligatoire.","cout":130,"tag":"Patrimoine"},
                    {"heure":"11:30","icon":"fa-building","titre":"Quartier des Habous — Nouvelle Médina","description":"Construit par les Français en 1936, ce quartier mêle architecture arabo-andalouse et coloniale. Librairies coraniques, bijoutiers et pâtisseries traditionnelles.","cout":200,"tag":"Culture"},
                    {"heure":"13:30","icon":"fa-utensils","titre":"Déjeuner à La Sqala","description":"Dans un bastion portugais du XVIIIe siècle : briouates au fromage, tajine de poisson et salade marocaine dans un jardin d'orangers et de roses.","cout":180,"tag":"Gastronomie"},
                    {"heure":"15:30","icon":"fa-water","titre":"Corniche Ain Diab","description":"6 km de promenade longeant l'Atlantique. Clubs de plage, aquaparcs, piscines d'eau de mer. La marée montante crée des vagues spectaculaires.","cout":50,"tag":"Nature"},
                    {"heure":"19:30","icon":"fa-fish","titre":"Dîner au Le Cabestan","description":"Restaurant gastronomique perché sur les rochers face à l'Atlantique : homard, bar grillé, daurade royale. Coucher de soleil inoubliable sur l'océan.","cout":600,"tag":"Gastronomie"},
                ]
            },
            {
                "titre": "Art Déco, Mode & Ocean",
                "activites": [
                    {"heure":"09:00","icon":"fa-landmark","titre":"Villa des Arts — Musée Contemporain","description":"Chef-d'œuvre Art Déco des années 30 : expositions d'art contemporain marocain, sculptures en bronze et peintures modernes dans 3 000 m² de salles.","cout":30,"tag":"Art"},
                    {"heure":"10:30","icon":"fa-city","titre":"Boulevard Mohammed V — Art Déco","description":"Flânez sur le boulevard historique et admirez les façades Art Déco des années 30 : Immeuble La Réunion, Banque d'État, Grand Hôtel. Architecture coloniale remarquable.","cout":0,"tag":"Art"},
                    {"heure":"13:00","icon":"fa-utensils","titre":"Déjeuner au Rick's Café","description":"Réplique du café de Humphrey Bogart dans Casablanca (1942). Ambiance jazz live, cocktails sans alcool et cuisine méditerranéenne dans un décor hollywoodien.","cout":350,"tag":"Gastronomie"},
                    {"heure":"15:00","icon":"fa-shopping-bag","titre":"Morocco Mall","description":"Le plus grand centre commercial d'Afrique (250 000 m²) : marques internationales, aquarium géant de 1 200 m², piste de patinage et foodcourt panoramique.","cout":500,"tag":"Shopping"},
                    {"heure":"19:30","icon":"fa-moon","titre":"Soirée Ain Diab — Bars & Restaurants","description":"La corniche s'illumine : choix entre boîtes de nuit, restaurants de fruits de mer et terrasses avec vue mer. Essayez les crevettes grillées au beurre d'ail.","cout":400,"tag":"Gastronomie"},
                ]
            },
            {
                "titre": "Médina, Port & Marché Populaire",
                "activites": [
                    {"heure":"09:00","icon":"fa-archway","titre":"Ancienne Médina de Casablanca","description":"Médina compacte du XVIIIe s. nichée entre le port et la ville moderne. Épices, henné, babouches et tissu caftan à prix imbattables.","cout":150,"tag":"Culture"},
                    {"heure":"10:30","icon":"fa-fish","titre":"Port de Pêche — Criée du Matin","description":"Spectacle vivant de la vente aux enchères du poisson frais : thon, espadon, sardines et céphalopodes débarqués à l'aube par la flottille casablancaise.","cout":0,"tag":"Culture"},
                    {"heure":"13:00","icon":"fa-utensils","titre":"Poisson Frais au Port","description":"À 30 m de la criée, des gargotes grillent le poisson à la commande : sardines grillées, sole meunière et brochettes de calamars pour moins de 80 MAD.","cout":80,"tag":"Gastronomie"},
                    {"heure":"15:00","icon":"fa-tree","titre":"Parc de la Ligue Arabe","description":"Poumon vert du centre-ville (5 ha) : allées bordées de palmiers et d'acacias, fontaines et kiosques. Lieu de promenade des Casablancais depuis 1918.","cout":0,"tag":"Nature"},
                    {"heure":"17:00","icon":"fa-palette","titre":"Galerie 38 — Art Contemporain","description":"Espace Violette ou Galerie 38 : expositions des artistes marocains emergents. Vernissages le jeudi soir — accès libre sur invitation ou spontanément.","cout":0,"tag":"Art"},
                    {"heure":"19:30","icon":"fa-moon","titre":"Dîner Chez Paul — Cuisine Française","description":"Brasserie classique au cœur du Maarif : entrecôte, foie gras, bouillabaisse et cave à vins avec 200 références. Une institution depuis 1947.","cout":400,"tag":"Gastronomie"},
                ]
            },
        ]
    },
    "Chefchaouen": {
        "emoji": "💙", "type": "Ville",
        "slogan": "La Perle Bleue du Rif — Village des Mille Nuances",
        "hotel": {"nom": "Riad Cherifa", "etoiles": 3, "prix_nuit": 320, "quartier": "Médina Bleue"},
        "transport_arrivee": "CTM depuis Fès (3h) : 90 MAD | depuis Tanger (3h) : 70 MAD",
        "budget_jour": {"hebergement": 320, "repas": 120, "activites": 80, "transport": 50},
        "conseils": [
            "Prenez une chambre avec vue sur la médina — le coucher de soleil est magique",
            "Le fromage de chèvre local vendu au marché du dimanche est délicieux",
            "Randonnée vers Ras El Ma (2h) : cascade et piscine naturelle accessible",
            "Les ruelles bleues sont moins fréquentées à l'aube — idéal pour la photo"
        ],
        "jours": [
            {
                "titre": "Découverte de la Médina Bleue",
                "activites": [
                    {"heure":"08:00","icon":"fa-camera","titre":"Ruelles Bleues à l'Aube","description":"Le bleu de Chefchaouen est unique au monde : bleu ciel, cobalt, turquoise, indigo. Avant 8h, les ruelles sont désertes — la lumière rasante crée des ombres irréelles.","cout":0,"tag":"Culture"},
                    {"heure":"10:00","icon":"fa-fort-awesome","titre":"Kasbah et Musée Ethnographique","description":"Forteresse du XVe s. construite par Moulay Ali Ben Rachid. Le musée présente les costumes rifains, les instruments de musique et les bijoux berbères.","cout":10,"tag":"Patrimoine"},
                    {"heure":"12:30","icon":"fa-utensils","titre":"Déjeuner sur la Plaza Uta el-Hammam","description":"Asseyez-vous en terrasse sous les ormes centenaires : harira, msemen au miel et tajine de chevreau (spécialité locale) pour 60-90 MAD.","cout":90,"tag":"Gastronomie"},
                    {"heure":"14:30","icon":"fa-store","titre":"Artisanat Rifain & Boutiques Bleues","description":"Laine tissée Rif (jellabas rayées), savon beldi local, huile d'argan, poteries turquoise et cactus en pot — tout est authentiquement local ici.","cout":250,"tag":"Shopping"},
                    {"heure":"17:00","icon":"fa-mosque","titre":"Mosquée Espagnole — Coucher de Soleil","description":"20 minutes de montée depuis la médina. Panorama absolu sur Chefchaouen dans sa cuvette de montagne, les minarets et le Rif enneigé en fond.","cout":0,"tag":"Nature"},
                    {"heure":"19:30","icon":"fa-moon","titre":"Dîner au Lala Mesouda","description":"Le meilleur restaurant de Chefchaouen : tajine de chevreau au safran, couscous aux 7 légumes et thé à la menthe poivrée sur terrasse illuminée.","cout":150,"tag":"Gastronomie"},
                ]
            },
            {
                "titre": "Nature, Cascade & Randonnée",
                "activites": [
                    {"heure":"08:30","icon":"fa-water","titre":"Ras El Ma — Source & Cascade","description":"25 min à pied depuis la médina. La rivière El Jaouna jaillit de la roche sous un pont romain. Lavandières au travail, canards et truite d'eau froide.","cout":0,"tag":"Nature"},
                    {"heure":"10:30","icon":"fa-hiking","titre":"Randonnée Gorges d'Akchour","description":"40 min en taxi (30 MAD). Trek de 4h aller-retour : forêt de pins et de cèdres, cascades successives culminant au Pont de Dieu — arche naturelle de 25 m.","cout":150,"tag":"Nature"},
                    {"heure":"14:00","icon":"fa-utensils","titre":"Pique-nique en Forêt","description":"Les guides locaux préparent un pique-nique berbère (khoubz, olives, fromage local, harissa et fruits de saison) au bord de la rivière Laou.","cout":80,"tag":"Gastronomie"},
                    {"heure":"16:30","icon":"fa-spa","titre":"Hammam Traditionnel","description":"Hammam public de la médina (15 MAD) ou hammam privé du riad (100 MAD). Détente musculaire après la randonnée avec savon beldi et henné.","cout":100,"tag":"Bien-être"},
                    {"heure":"19:30","icon":"fa-moon","titre":"Soirée Musicale & Dîner","description":"Restaurant Bab Ssour : musique traditionnelle rifaine, pastilla aux pigeons et brochettes de kefta au romarin. Vue sur le Rif illuminé la nuit.","cout":130,"tag":"Gastronomie"},
                ]
            },
            {
                "titre": "Marché, Fromage & Artisans",
                "activites": [
                    {"heure":"08:00","icon":"fa-shopping-basket","titre":"Marché Hebdomadaire Lundi/Jeudi","description":"Marché rifain authentique : femmes en jupe à rayures rouges, fromages de chèvre fumés, miel de thym, noix du Rif et piments séchés à prix paysans.","cout":100,"tag":"Culture"},
                    {"heure":"10:00","icon":"fa-hands","titre":"Coopérative de Tissage Féminin","description":"15 femmes rifaines tissent à la main les célèbres kilims de Chefchaouen. Démonstration sur métier à tisser vertical — achetez directement à la source.","cout":300,"tag":"Culture"},
                    {"heure":"13:00","icon":"fa-utensils","titre":"Déjeuner Chez Casa Hassan","description":"Tajine de poulet aux amandes et abricots, salade marocaine et pain khobz sorti du four. Vue directe sur la Plaza Uta depuis la terrasse.","cout":100,"tag":"Gastronomie"},
                    {"heure":"15:00","icon":"fa-camera","titre":"Street Photography — Ruelles Bleues","description":"Parcourez les quartiers el-Andalus et el-Onsar : chaque recoin est une composition. Les habitants sont accueillants — demandez la permission pour les portraits.","cout":0,"tag":"Art"},
                    {"heure":"18:00","icon":"fa-tree","titre":"Promenade au Pin Solitaire","description":"Sentier de 30 min derrière la médina menant au célèbre pin isolé avec vue à 360°. C'est l'heure magique du golden hour pour la photographie.","cout":0,"tag":"Nature"},
                    {"heure":"20:00","icon":"fa-moon","titre":"Dîner Rooftop — Vue Nocturne","description":"Restaurant Bab Al-Ain : terrasse panoramique, brochettes de chevreau marinées, salade de tomates au cumin et fondant au chocolat-argan.","cout":130,"tag":"Gastronomie"},
                ]
            },
        ]
    },
    "Essaouira": {
        "emoji": "🌊", "type": "Ville",
        "slogan": "La Cité des Vents — Perle de l'Atlantique",
        "hotel": {"nom": "Riad Mimouna", "etoiles": 3, "prix_nuit": 350, "quartier": "Médina UNESCO"},
        "transport_arrivee": "Bus Supratours depuis Marrakech (3h) : 80 MAD",
        "budget_jour": {"hebergement": 350, "repas": 160, "activites": 120, "transport": 60},
        "conseils": [
            "Le vent de l'Atlantique (Alizé) est fort — emportez un coupe-vent",
            "Le port de pêche ferme à 14h — visitez-le le matin",
            "La saison Gnawa Festival (juin) : musique mystique et concerts gratuits",
            "Spécialité locale : l'huile d'argan grillée (amlou) sur pain khobz"
        ],
        "jours": [
            {
                "titre": "Remparts, Port & Médina UNESCO",
                "activites": [
                    {"heure":"09:00","icon":"fa-anchor","titre":"Port de Pêche Artisanal","description":"50 barques bleues débarquent sardines, céphalopodes et dorades. Les goélands plongent en piqué sur les caisses de poisson. Achetez du poisson pour le griller vous-même.","cout":0,"tag":"Culture"},
                    {"heure":"10:30","icon":"fa-fort-awesome","titre":"Remparts Nord — Scala de la Ville","description":"Bastions portugais du XVIIIe s. armés de 24 canons en bronze. Vue imprenable sur l'Atlantique et les îles Purpuraires (site romain classé UNESCO).","cout":10,"tag":"Patrimoine"},
                    {"heure":"13:00","icon":"fa-fish","titre":"Déjeuner — Grillades du Port","description":"Au-delà des portes du port : stands de poisson grillé sur braises. Gambas, sardines, seiche — le tout pour 60-80 MAD avec pain et thé.","cout":80,"tag":"Gastronomie"},
                    {"heure":"14:30","icon":"fa-store","titre":"Médina & Galeries d'Art","description":"Essaouira est un carrefour artistique : galeries de peinture gnawa, sculpteurs sur thuya, bijoutiers berbères. La rue de la Skala regorge d'ateliers d'artistes.","cout":200,"tag":"Art"},
                    {"heure":"17:00","icon":"fa-sun","titre":"Plage — Coucher de Soleil Atlantique","description":"3 km de plage sauvage face à l'Atlantique : le soleil plonge dans l'océan en créant des couleurs d'or et de pourpre. Spectacle gratuit et inoubliable.","cout":0,"tag":"Nature"},
                    {"heure":"19:30","icon":"fa-music","titre":"Concert Gnawa — Bar Taros","description":"Musique mystique des esclaves sub-sahariens : guembri (basse à 3 cordes), crotales en métal et chants de transe. Le Taros accueille des concerts tous les soirs.","cout":50,"tag":"Culture"},
                ]
            },
            {
                "titre": "Surf, Windsurf & Vie Locale",
                "activites": [
                    {"heure":"08:30","icon":"fa-wind","titre":"Cours de Windsurf — École Ocean Vagabond","description":"Essaouira est classée 4ème spot de windsurf mondial. Cours débutant (2h) avec moniteur certifié : planche, combinaison et harnais fournis.","cout":350,"tag":"Sport"},
                    {"heure":"11:00","icon":"fa-tree","titre":"Forêt d'Arganiers — Chèvres Grimpantes","description":"À 5 km de la ville : l'unique forêt d'arganiers endémique au Maroc, classée Réserve de Biosphère UNESCO. Les chèvres grimpent dans les branches pour manger les fruits.","cout":0,"tag":"Nature"},
                    {"heure":"13:30","icon":"fa-utensils","titre":"Déjeuner Chez Sam — Cuisine de Mer","description":"Restaurant historique face au port (depuis 1968) : tagine de lotte aux légumes, fruits de mer et vinaigre d'argan sur les tables en bois blanc.","cout":160,"tag":"Gastronomie"},
                    {"heure":"15:30","icon":"fa-hammer","titre":"Atelier de Thuya — Artisanat Unique","description":"Le thuya d'Essaouira (conifère endémique) est sculpté en objets d'art : échiquier marquetés, boîtes à bijoux, tables basses. Visitez l'atelier Moulay Idriss.","cout":250,"tag":"Art"},
                    {"heure":"18:00","icon":"fa-spa","titre":"Hammam Riad & Soin Argan","description":"Hammam du riad avec huile d'argan pressée à froid : après le gommage, massage huile chaude sur tout le corps. 90 minutes de régénération profonde.","cout":300,"tag":"Bien-être"},
                    {"heure":"20:00","icon":"fa-moon","titre":"Dîner Dar Liouba — Cuisine Gnawa","description":"Maison de la cuisine gnawa : couscous au poisson et lait caillé, tagine de poulet aux olives violettes et msemen au miel de thym.","cout":180,"tag":"Gastronomie"},
                ]
            },
        ]
    },
    "Merzouga": {
        "emoji": "🏜️", "type": "Nature",
        "slogan": "Erg Chebbi — Les Dunes d'Or du Sahara",
        "hotel": {"nom": "Kasbah Mohayut", "etoiles": 4, "prix_nuit": 500, "quartier": "Bord des dunes"},
        "transport_arrivee": "Route depuis Errachidia (1h) ou Ouarzazate (4h). Location voiture conseillée.",
        "budget_jour": {"hebergement": 500, "repas": 180, "activites": 250, "transport": 120},
        "conseils": [
            "Réservez le camp de luxe dans les dunes (bivouac) au moins 2 semaines à l'avance",
            "Emportez une écharpe/chèche pour protéger le visage du vent de sable",
            "Le lever du soleil sur les dunes vaut le réveil à 5h30 — chapeau, lunettes et crème",
            "Température : 45°C le jour, 5°C la nuit en hiver — prévoyez des couches"
        ],
        "jours": [
            {
                "titre": "Arrivée au Désert & Erg Chebbi",
                "activites": [
                    {"heure":"06:00","icon":"fa-sun","titre":"Lever de Soleil sur les Dunes","description":"Réveil avant l'aube. Montez à pied ou à dromadaire le sommet de la grande dune (160 m). Les dunes virent de l'orange au rouge sang — un spectacle cosmique gratuit.","cout":0,"tag":"Nature"},
                    {"heure":"09:00","icon":"fa-horse","titre":"Balade à Dromadaire — 2h","description":"Caravane de dromadaires encadrée par un guide touareg. Traversée des couloirs de sable entre les crêtes éolienne. Les dromadaires marchent à 6 km/h — idéal pour photographier.","cout":250,"tag":"Nature"},
                    {"heure":"12:00","icon":"fa-utensils","titre":"Déjeuner Bivouac Berbère","description":"Sous la khaïma (tente berbère) tendue face aux dunes : tajine de dromadaire, couscous au beurre de chamelle, thé touareg sucré et dattes Medjool du Sahara.","cout":120,"tag":"Gastronomie"},
                    {"heure":"14:30","icon":"fa-snowboarding","titre":"Sandboarding sur l'Erg","description":"Planchez sur les pentes de sable fin avec des surfs adaptés. Dévalées de 50 m de dénivelé à 40 km/h. Sensations fortes garanties sans risque — le sable amortit les chutes !","cout":100,"tag":"Sport"},
                    {"heure":"17:00","icon":"fa-music","titre":"Coucher de Soleil & Musique Gnawa","description":"Les musiciens nomades jouent au sommet des dunes pendant que le soleil disparaît à l'horizon. Guembri, crotales et chant en harmattan. Ce moment ne s'oublie pas.","cout":80,"tag":"Culture"},
                    {"heure":"20:00","icon":"fa-star","titre":"Nuit en Camp de Luxe — Ciel Étoilé","description":"Bivouac de luxe au cœur des dunes : tentes berbères chaufées, lit queen-size, douche chaude. Observation des étoiles (Sahara = 0% pollution lumineuse). Voie Lactée à l'œil nu.","cout":500,"tag":"Nature"},
                ]
            },
            {
                "titre": "Villages Nomades & Sources Bleues",
                "activites": [
                    {"heure":"07:00","icon":"fa-sun","titre":"Aube Photographique","description":"Sortez seul à l'aube avec votre appareil photo. Les dunes désertées créent des compositions abstraites. Les empreintes du vent (ridules) disparaîtront à 8h avec le soleil.","cout":0,"tag":"Art"},
                    {"heure":"09:30","icon":"fa-users","titre":"Village Khamlia — Musique Gnawa","description":"Village de descendants d'esclaves sub-sahariens. Les femmes en robes colorées dansent la Lila (cérémonie de transe). Offrandes de thé et musique rituelle spontanée.","cout":50,"tag":"Culture"},
                    {"heure":"12:00","icon":"fa-utensils","titre":"Déjeuner Chez Famille Nomade","description":"Accueil dans une famille touareg sédentarisée : méchoui (agneau rôti entier), pain cuit sous la braise et lait caillé de chamelle. Hospitalité saharienne authentique.","cout":150,"tag":"Gastronomie"},
                    {"heure":"14:30","icon":"fa-car","titre":"4x4 Erg — Pistes Sahariennes","description":"Rally dans le désert en 4x4 Toyota conduit par un guide rifain : dunes vives, oasis cachées, lits de rivières fossiles (oueds asséchés) et fossiles de trilobites.","cout":300,"tag":"Sport"},
                    {"heure":"17:30","icon":"fa-water","titre":"Source Bleue de Merzouga","description":"Source d'eau douce qui surgit mystérieusement dans le désert à -5°C. Les Touaregs y viennent abreuver leurs dromadaires depuis des millénaires.","cout":0,"tag":"Nature"},
                    {"heure":"20:00","icon":"fa-star","titre":"Veillée Saharienne — Contes & Feu","description":"Autour d'un feu de bois de tamaris : le guide raconte les légendes des caravanes transsahariennes, des Touaregs et des tempêtes de sable. Thé au poivre noir.","cout":60,"tag":"Culture"},
                ]
            },
            {
                "titre": "Oasis, Palmeraies & Gorges",
                "activites": [
                    {"heure":"07:30","icon":"fa-mountain","titre":"Excursion Gorges du Todra — 1h de route","description":"Canon spectaculaire de 300 m de hauteur pour 10 m de largeur. Les falaises roses se colorent au lever du soleil. Rivière froide (12°C) entre les parois.","cout":150,"tag":"Nature"},
                    {"heure":"10:00","icon":"fa-tree","titre":"Palmeraie de Skoura — Oasis millénaire","description":"3 000 hectares de palmiers-dattiers irrigués par les khettaras (canaux souterrains). Kasbahs en pisé rouge, ânes et champs de henné dans cette oasis médiévale.","cout":50,"tag":"Nature"},
                    {"heure":"13:00","icon":"fa-utensils","titre":"Déjeuner Kasbah Ait Benhaddou","description":"Kasbah classée UNESCO (tournage de Game of Thrones). Restaurant panoramique : tajine de pigeon aux figues et amandes, couscous berbère, pastilla sucrée.","cout":180,"tag":"Gastronomie"},
                    {"heure":"15:30","icon":"fa-landmark","titre":"Kasbah Amerhidil — Skoura","description":"L'une des plus belles kasbahs du Maroc (XVIe s.) : tours d'angle en pisé rouge, motifs géométriques et frises de briques cuites. Musée des arts du désert à l'intérieur.","cout":30,"tag":"Patrimoine"},
                    {"heure":"18:00","icon":"fa-sun","titre":"Retour & Coucher de Soleil Erg","description":"Retour à Merzouga pour le coucher de soleil. Les dunes prennent des couleurs de brique, d'or et de cinabre selon l'angle de la lumière rasante.","cout":0,"tag":"Nature"},
                    {"heure":"20:00","icon":"fa-moon","titre":"Dîner & Hammam de Sable","description":"Spécialité saharienne unique : bain de sable chaud naturellement à 50°C (traitement des rhumatismes). Puis dîner à la kasbah : méchoui et thé noir aux clous de girofle.","cout":300,"tag":"Bien-être"},
                ]
            },
        ]
    },
    "Imlil": {
        "emoji": "🏔️", "type": "Nature",
        "slogan": "Vallée d'Imlil — Porte du Toubkal (4 167 m)",
        "hotel": {"nom": "Kasbah du Toubkal", "etoiles": 4, "prix_nuit": 550, "quartier": "Imlil village"},
        "transport_arrivee": "Taxi collectif depuis Marrakech (1h30) : 30 MAD/pers | Taxi privé : 300 MAD",
        "budget_jour": {"hebergement": 550, "repas": 150, "activites": 200, "transport": 80},
        "conseils": [
            "Trek Toubkal (2 jours) : guide obligatoire (300 MAD/j) — réservez à Imlil",
            "Altitude 4 167 m : acclimatation nécessaire — montez en 2 jours minimum",
            "Équipement : chaussures de randonnée, veste duvet, gants, lampe frontale",
            "Le bivouac au Refuge Toubkal (3 207 m) est une expérience inoubliable"
        ],
        "jours": [
            {
                "titre": "Arrivée & Vallée d'Imlil",
                "activites": [
                    {"heure":"09:00","icon":"fa-hiking","titre":"Trek Vallée — Villages Berbères","description":"Randonnée douce (3h, 400 m D+) à travers noyers, pommiers et terrasses de culture. Villages d'Aremd, Sidi Chamharouch (sanctuaire) et Achayn aux toits de pierre.","cout":0,"tag":"Nature"},
                    {"heure":"12:00","icon":"fa-utensils","titre":"Déjeuner Chez l'Habitant — Gîte Local","description":"Femmes berbères qui cuisinent à l'âtre : tagine de poulet fermier aux légumes du jardin, couscous de semoule d'orge et thé à la menthe poivrée fraîche cueilli ce matin.","cout":80,"tag":"Gastronomie"},
                    {"heure":"14:00","icon":"fa-tree","titre":"Vergers & Apiculture Berbère","description":"Les apiculteurs d'Imlil produisent un miel d'euphorbes, de romarin et de lavande sauvage. Visite des ruches et dégustation gratuite. Achat possible : 150 MAD/500g.","cout":150,"tag":"Nature"},
                    {"heure":"16:00","icon":"fa-water","titre":"Cascade d'Imlil — Rivière Réroua","description":"30 min de marche sur sentier tracé. Cascade de 12 m dans un bassin naturel entouré de figuiers sauvages. Baignade possible en été (eau à 14°C).","cout":0,"tag":"Nature"},
                    {"heure":"19:00","icon":"fa-moon","titre":"Dîner Étoilé — Kasbah du Toubkal","description":"Terrasse à 1 740 m d'altitude avec vue frontale sur le Toubkal enneigé. Tajine de légumes du jardin, soupe harira et fondant aux amandes grillées.","cout":200,"tag":"Gastronomie"},
                ]
            },
            {
                "titre": "Ascension Refuge du Toubkal — J1",
                "activites": [
                    {"heure":"07:00","icon":"fa-mountain","titre":"Départ Trek — Imlil → Refuge (3207m)","description":"6h de marche (10 km, 1 500 m D+) avec guide certifié. Paysages alpins spectaculaires : moraines, cirques glaciaires et névés éternels au-delà de 3 000 m.","cout":300,"tag":"Sport"},
                    {"heure":"10:30","icon":"fa-utensils","titre":"Pause Grignotage — Sidi Chamharouch","description":"Sanctuaire à 2 310 m : marabout blanc accroché à un rocher de granit. Thé et biscuits aux amandes servis par les gardiens du sanctuaire (offrandes bienvenues).","cout":30,"tag":"Culture"},
                    {"heure":"13:30","icon":"fa-snowflake","titre":"Plateau du Mizane — Paysage Lunaire","description":"Au-dessus de 3 000 m, la végétation disparaît. Plateaux de pierres, névés et panorama à 360° sur les sommets de l'Atlas : Ouanoukrim (4 088 m) et Aksoual (3 912 m).","cout":0,"tag":"Nature"},
                    {"heure":"16:00","icon":"fa-home","titre":"Arrivée Refuge CAF du Toubkal","description":"Refuge à 3 207 m du Club Alpin Français : dortoirs (60 places), cuisine collective. Rencontre avec les alpinistes internationaux du monde entier. Vue sommet à 1 km.","cout":180,"tag":"Nature"},
                    {"heure":"19:00","icon":"fa-utensils","titre":"Dîner & Repos au Refuge","description":"Soupe chaude, tagine de légumes en boîte et thé sucré. Coucher à 20h pour le départ au sommet à 4h du matin. Température : -5°C à -15°C selon la saison.","cout":100,"tag":"Gastronomie"},
                ]
            },
            {
                "titre": "Sommet Toubkal 4167m — J2",
                "activites": [
                    {"heure":"04:30","icon":"fa-moon","titre":"Départ Nocturne — Refuge → Sommet","description":"2h30 de marche frontale à la lampe dans la neige et les pierriers. Pente à 30-35°. L'effort est intense mais chaque pas rapproche du toit de l'Afrique du Nord.","cout":0,"tag":"Sport"},
                    {"heure":"07:00","icon":"fa-sun","titre":"Sommet du Toubkal — 4167 m","description":"Lever du soleil depuis le plus haut sommet d'Afrique du Nord. Vue à 600 km : Atlantique, Sahara, Canaries par temps clair. Un moment de plénitude absolue. Croix des alpinistes au sommet.","cout":0,"tag":"Nature"},
                    {"heure":"09:30","icon":"fa-hiking","titre":"Descente Refuge → Imlil","description":"Descente en 4h par le même itinéraire. Les jambes tremblent mais l'âme est en paix. Les guides berbères chantent en chemin — tradition après l'ascension réussie.","cout":0,"tag":"Nature"},
                    {"heure":"14:00","icon":"fa-utensils","titre":"Repas du Vainqueur — Imlil","description":"Table gargantuesque offerte par le guide : méchoui d'agneau, couscous aux 7 légumes, salade de carottes à la coriandre et thé à la menthe ultra-sucré bien mérité.","cout":120,"tag":"Gastronomie"},
                    {"heure":"16:00","icon":"fa-spa","titre":"Hammam & Soins Berbères","description":"Hammam du village (10 MAD) ou hammam privé du gîte (100 MAD). Gommage au savon de cèdre, massage aux huiles essentielles d'Atlas. Récupération musculaire complète.","cout":100,"tag":"Bien-être"},
                ]
            },
        ]
    },
    "Todra": {
        "emoji": "🏞️", "type": "Nature",
        "slogan": "Gorges du Todra — Cathedral de Pierre Rouge",
        "hotel": {"nom": "Hôtel Yasmina", "etoiles": 3, "prix_nuit": 300, "quartier": "Gorges, bord rivière"},
        "transport_arrivee": "Bus CTM depuis Errachidia (1h30) : 35 MAD | depuis Ouarzazate (3h) : 60 MAD",
        "budget_jour": {"hebergement": 300, "repas": 140, "activites": 180, "transport": 80},
        "conseils": [
            "Les gorges sont à leur plus beau à 11h (lumière zénithale entre les parois)",
            "Escalade : voies de tous niveaux (3 à 7b). Location de matériel sur place.",
            "La rivière est froide même en été (10-14°C) — idéale après la randonnée",
            "Tinghir (10 km) : palmeraie de 14 km à visiter à vélo ou à pied"
        ],
        "jours": [
            {
                "titre": "Les Gorges — Le Canyon Rouge",
                "activites": [
                    {"heure":"08:00","icon":"fa-mountain","titre":"Entrée des Gorges à l'Aube","description":"300 m de hauteur pour 10 m de largeur au point le plus étroit. Les falaises de calcaire rouge virent à l'orange au lever du soleil. Peu de touristes avant 9h.","cout":0,"tag":"Nature"},
                    {"heure":"09:30","icon":"fa-climbing","titre":"Escalade dans les Gorges","description":"40+ voies d'escalade équipées sur calcaire compact (3 à 7b). Guide local optionnel (200 MAD/j). Vues vertigineuses du haut des parois sur la palmeraie de Tinghir.","cout":200,"tag":"Sport"},
                    {"heure":"13:00","icon":"fa-utensils","titre":"Déjeuner au Bord de l'Oued Todra","description":"Restaurant Yasmina : tajine de truite du Todra aux amandes, salade de betterave au cumin et pain khobz frais. Table les pieds dans l'eau de la rivière.","cout":100,"tag":"Gastronomie"},
                    {"heure":"15:00","icon":"fa-hiking","titre":"Trek Gorges Supérieures — Tamtattouchte","description":"Randonnée (4h, 12 km) dans les gorges supérieures moins connues. Canyon encore plus étroit, troupeaux de chèvres, nomades berbères et vautours fauves en vol.","cout":150,"tag":"Nature"},
                    {"heure":"18:00","icon":"fa-sun","titre":"Coucher de Soleil — Belvédère","description":"Montée au belvédère (30 min) pour le coucher de soleil sur les gorges. Les parois deviennent cramoisies, puis violettes, puis noires. Clichés de cartes postales garantis.","cout":0,"tag":"Nature"},
                    {"heure":"20:00","icon":"fa-moon","titre":"Dîner Kasbah Taborihte","description":"Kasbah rénovée dans le village d'Aït Baddou : méchoui aux herbes du désert, harira épicée et crème de semoule aux raisins secs et cannelle.","cout":160,"tag":"Gastronomie"},
                ]
            },
            {
                "titre": "Palmeraie de Tinghir & Kasbahs",
                "activites": [
                    {"heure":"08:30","icon":"fa-tree","titre":"Palmeraie de Tinghir — 14 km","description":"Une des plus grandes oasis du Maroc : 14 km de palmiers-dattiers, champs de henné, jardins de tomates et canaux d'irrigation millénaires. Balade à pied ou à vélo.","cout":50,"tag":"Nature"},
                    {"heure":"10:30","icon":"fa-landmark","titre":"Kasbah Aït Benhaddou — Route des Kasbahs","description":"À 2h de route : Kasbah la plus spectaculaire du Maroc, classée UNESCO. 5 films tournés ici (Game of Thrones, Gladiateur, Babel). Guide inclus dans le billet.","cout":80,"tag":"Patrimoine"},
                    {"heure":"13:30","icon":"fa-utensils","titre":"Déjeuner Kasbah Tifoultout — Ouarzazate","description":"Ancienne résidence du Pacha Glaoui transformée en restaurant : tajine de poulet confit aux citrons et olives, salade marocaine et msemen au beurre.","cout":150,"tag":"Gastronomie"},
                    {"heure":"15:30","icon":"fa-film","titre":"Cinéma Studios Atlas — Ouarzazate","description":"Hollywood d'Afrique : décors de péplum, costumes de Game of Thrones et Gladiateur. Montez sur le trône de fer en fer forgé — la photo parfaite.","cout":100,"tag":"Culture"},
                    {"heure":"18:00","icon":"fa-water","titre":"Retour Gorges — Baignade Rivière","description":"Retour dans les gorges pour la baignade au crépuscule. La lumière déclinante colore les falaises en rose et saumon. L'eau froide revigore après la chaleur de la route.","cout":0,"tag":"Nature"},
                    {"heure":"20:00","icon":"fa-moon","titre":"Dîner Sous les Étoiles","description":"Restaurant El Mansour : terrasse en pisé sous un ciel sans pollution lumineuse. Méchoui, couscous aux pois chiches et nougat aux amandes et miel de thym.","cout":140,"tag":"Gastronomie"},
                ]
            },
            {
                "titre": "Oasis, Artisanat & Nomades",
                "activites": [
                    {"heure":"07:30","icon":"fa-seedling","titre":"Visite des Jardins Oasis — Iriqui","description":"Jardins suspendus au-dessus de l'oued : figuiers, grenadiers, poivrons séchés et champs de luzerne. Les femmes berbères récoltent à la main les herbes aromatiques.","cout":20,"tag":"Nature"},
                    {"heure":"09:30","icon":"fa-hands","titre":"Coopérative Tapis Berbère — Tinghir","description":"20 femmes tissent des tapis Kilim, Boucherouite et Berber à la main sur des métiers en bois. Chaque tapis raconte une histoire — formes géométriques et couleurs naturelles.","cout":400,"tag":"Art"},
                    {"heure":"13:00","icon":"fa-utensils","titre":"Marché de Tinghir — Spécialités","description":"Marché du mardi/dimanche : dattes Medjool de 1ère qualité, fromage de brebis affiné, miel de l'Atlas, huile d'olive et épices sahariennes. Repas sur place pour 25 MAD.","cout":60,"tag":"Gastronomie"},
                    {"heure":"15:00","icon":"fa-users","titre":"Village Nomade — Rencontre Authentique","description":"Le guide vous emmène chez une famille nomade semi-sédentarisée : tente noire en poils de chameau, hospitalité traditionnelle, thé fort et échange culturel sincère.","cout":100,"tag":"Culture"},
                    {"heure":"17:30","icon":"fa-mountain","titre":"Randonnée Crête — Panorama Infini","description":"2h de randonnée sur la crête au-dessus des gorges. Vue sur 3 systèmes géologiques : le Haut Atlas (nord), le Jbel Saghro (sud) et l'Erg Sahara à l'horizon.","cout":80,"tag":"Nature"},
                    {"heure":"20:00","icon":"fa-fire","titre":"Soirée Berbère — Feu & Musique","description":"Les habitants du village organisent une ahwach (danse collective berbère) autour du feu : percussions de bendir, chants en amazigh et tajine de chevreau.","cout":100,"tag":"Culture"},
                ]
            },
        ]
    },
}


@app.route('/api/plan', methods=['POST'])
def api_plan():
    """Générer un plan de voyage riche et personnalisé"""
    import json as _json
    data = request.get_json()

    dest_raw  = data.get('destination', 'Marrakech')
    duree     = max(1, min(int(data.get('duree', 3)), 30))
    voyageurs = max(1, int(data.get('voyageurs', 2)))
    budget    = int(data.get('budget', 5000))
    save_plan = data.get('save', False)

    # Résoudre l'alias du select vers le nom complet
    dest = DEST_ALIASES.get(dest_raw, dest_raw)

    # Chercher dans PLANS_DEST, sinon générer un plan générique
    if dest in PLANS_DEST:
        info = PLANS_DEST[dest]
    else:
        info = generate_generic_plan(dest)

    templates = info['jours']
    nb_tpl    = len(templates)

    plan = []
    for day in range(1, duree + 1):
        tpl = templates[(day - 1) % nb_tpl]
        plan.append({
            "jour":      day,
            "titre":     tpl['titre'],
            "activites": tpl['activites']
        })

    # Budget estimé (par personne)
    bj   = info['budget_jour']
    total_pp = sum(bj.values()) * duree          # par personne
    total    = total_pp * voyageurs

    budget_breakdown = {
        "hebergement": bj['hebergement'] * duree * voyageurs,
        "repas":       bj['repas']       * duree * voyageurs,
        "activites":   bj['activites']   * duree * voyageurs,
        "transport":   bj['transport']   * duree * voyageurs,
        "total_estime": total,
        "par_personne": total_pp
    }

    # Stats SQLite + sauvegarde optionnelle
    plan_id = None
    if 'user_id' in session:
        conn = get_db()
        try:
            conn.execute(
                "UPDATE users SET trips_planned = trips_planned + 1 WHERE id = ?",
                (session['user_id'],)
            )
            if save_plan:
                cur = conn.execute(
                    "INSERT INTO trip_plans (user_id, destination, duree, voyageurs, budget_utilisateur, plan_json, created_at) VALUES (?,?,?,?,?,?,?)",
                    (session['user_id'], dest, duree, voyageurs, budget,
                     _json.dumps({"plan": plan, "budget_breakdown": budget_breakdown,
                                  "hotel": info['hotel'], "conseils": info['conseils'],
                                  "slogan": info['slogan'], "emoji": info['emoji'],
                                  "transport_arrivee": info['transport_arrivee']},
                                ensure_ascii=False),
                     datetime.now().isoformat())
                )
                plan_id = cur.lastrowid
            conn.commit()
        finally:
            conn.close()

    return jsonify({
        "success":           True,
        "plan_id":           plan_id,
        "destination":       dest,
        "duree":             duree,
        "voyageurs":         voyageurs,
        "budget_utilisateur": budget,
        "emoji":             info['emoji'],
        "slogan":            info['slogan'],
        "hotel":             info['hotel'],
        "transport_arrivee": info['transport_arrivee'],
        "budget_breakdown":  budget_breakdown,
        "conseils":          info['conseils'],
        "plan":              plan
    })


# ============================================================
# API - SERVICE 3: GROUPES DE VOYAGE
# ============================================================

@app.route('/api/groups', methods=['GET'])
def api_get_groups():
    """Lister tous les groupes"""
    import json as _json
    conn = get_db()
    try:
        rows = conn.execute("SELECT * FROM groups ORDER BY created_at DESC").fetchall()
        groups = []
        for g in rows:
            g_dict = dict(g)
            members = conn.execute(
                "SELECT user_id FROM group_members WHERE group_id = ?", (g['id'],)
            ).fetchall()
            g_dict['members'] = [m['user_id'] for m in members]
            try: g_dict['interets'] = _json.loads(g_dict.get('interets') or '[]')
            except: g_dict['interets'] = []
            groups.append(g_dict)
        return jsonify({"success": True, "groups": groups})
    finally:
        conn.close()


@app.route('/api/groups', methods=['POST'])
def api_create_group():
    """Créer un nouveau groupe (enrichi)"""
    import json as _json
    data = request.get_json()

    name = data.get('name', '').strip()
    dest = data.get('dest', '').strip()
    date = data.get('date', '').strip()
    max_members = int(data.get('max', 10))
    desc = data.get('desc', '').strip()
    user_id = session.get('user_id')

    # Champs enrichis
    type_activite      = data.get('type_activite', 'Mixte')
    niveau             = data.get('niveau', 'Tous niveaux')
    interets           = _json.dumps(data.get('interets', []), ensure_ascii=False)
    age_min            = int(data.get('age_min', 18) or 18)
    age_max            = int(data.get('age_max', 99) or 99)
    budget_par_personne= int(data.get('budget_par_personne', 0) or 0)
    langue             = data.get('langue', 'Arabe/Français')

    if not name or not dest or not date:
        return jsonify({"success": False, "message": "Nom, destination et date sont obligatoires"}), 400

    conn = get_db()
    try:
        cursor = conn.execute(
            """INSERT INTO groups
               (name, dest, date, max, current, desc, creator_id, created_at,
                type_activite, niveau, interets, age_min, age_max, budget_par_personne, langue)
               VALUES (?,?,?,?,1,?,?,?,?,?,?,?,?,?,?)""",
            (name, dest, date, max_members, desc, user_id, datetime.now().isoformat(),
             type_activite, niveau, interets, age_min, age_max, budget_par_personne, langue)
        )
        group_id = cursor.lastrowid
        if user_id:
            conn.execute("INSERT OR IGNORE INTO group_members (group_id, user_id) VALUES (?,?)", (group_id, user_id))
        conn.commit()
        group = dict(conn.execute("SELECT * FROM groups WHERE id=?", (group_id,)).fetchone())
        group['members'] = [user_id] if user_id else []
        return jsonify({"success": True, "message": "Groupe créé!", "group": group})
    finally:
        conn.close()


@app.route('/api/groups/match', methods=['POST'])
def api_match_groups():
    """Trouver les groupes compatibles — KMeans + Cosine Similarity (IA)"""
    import json as _json
    data = request.get_json()
    age           = int(data.get('age', 25) or 25)
    budget        = int(data.get('budget', 0) or 0)
    type_activite = data.get('type_activite', 'Mixte') or 'Mixte'
    niveau        = data.get('niveau', 'Tous niveaux') or 'Tous niveaux'
    interets_user = data.get('interets', [])

    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM groups WHERE current < max ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()

    # ── Étape 1 : cluster KMeans du user (âge, budget, sexe, marié) ──
    user_cluster      = None
    user_pref_type    = None   # type_destination prédit par le cluster
    sexe_val          = data.get('sexe', 'Homme')
    marie_val         = data.get('marie', 'Non')
    if KMEANS_READY and budget > 0:
        try:
            sexe_enc  = le_km_sexe.transform([sexe_val])[0]
        except Exception:
            sexe_enc  = 0
        try:
            marie_enc = le_km_marie.transform([marie_val])[0]
        except Exception:
            marie_enc = 0
        u_scaled     = scaler_km.transform([[age, budget, sexe_enc, marie_enc]])
        user_cluster = int(kmeans_model.predict(u_scaled)[0])
        user_pref_type = cluster_type_map.get(user_cluster)  # ex: "Ville" ou "Nature"

    # ── Étape 2 : vecteur user pour cosine similarity ──────────
    user_vec = encode_profile_vector(age, budget, type_activite, niveau, interets_user)

    results = []
    for g in rows:
        g_dict = dict(g)
        try: g_dict['interets'] = _json.loads(g_dict.get('interets') or '[]')
        except: g_dict['interets'] = []

        g_age    = ((g_dict.get('age_min') or 18) + (g_dict.get('age_max') or 99)) / 2
        g_budget = g_dict.get('budget_par_personne') or 0
        g_type   = g_dict.get('type_activite') or 'Mixte'
        g_niveau = g_dict.get('niveau') or 'Tous niveaux'
        g_ints   = g_dict.get('interets') or []

        # ── Étape 3 : cosine similarity ───────────────────────
        group_vec = encode_profile_vector(g_age, g_budget, g_type, g_niveau, g_ints)
        sim = float(cosine_similarity([user_vec], [group_vec])[0][0])

        # ── Étape 4 : bonus cluster KMeans (type destination prédit) ──
        cluster_bonus = 0.0
        cluster_msg   = ""
        if KMEANS_READY and user_pref_type:
            # Bonus si le type du groupe correspond au type prédit par le cluster
            if g_type == user_pref_type:
                cluster_bonus = 0.20
                cluster_msg   = f"IA prédit {user_pref_type} pour ton profil ✓"
            elif g_type == 'Mixte':
                cluster_bonus = 0.08
                cluster_msg   = "Groupe polyvalent compatible IA"

        # ── Étape 5 : pénalités contraintes dures ─────────────
        penalty = 0
        age_min = g_dict.get('age_min') or 18
        age_max = g_dict.get('age_max') or 99
        if not (age_min <= age <= age_max):
            penalty += 0.25

        if g_budget > 0 and budget > 0 and budget < g_budget * 0.7:
            penalty += 0.20

        # ── Score final (0-100) ───────────────────────────────
        final_score = max(0, min(100, round((sim + cluster_bonus - penalty) * 100)))

        # ── Explication lisible ───────────────────────────────
        reasons = []
        if sim >= 0.85:   reasons.append("Excellente compatibilité IA ✓")
        elif sim >= 0.65: reasons.append("Bonne compatibilité IA ✓")
        elif sim >= 0.45: reasons.append("Compatibilité modérée")
        else:             reasons.append("Profils peu similaires")
        if cluster_msg:   reasons.append(cluster_msg)
        if penalty >= 0.25 and not (age_min <= age <= age_max):
            reasons.append("⚠️ Hors tranche d'âge")
        if penalty >= 0.20 and g_budget > 0 and budget > 0 and budget < g_budget * 0.7:
            reasons.append("⚠️ Budget insuffisant")

        g_dict['score']          = final_score
        g_dict['match_reasons']  = reasons
        g_dict['ai_similarity']  = round(sim * 100, 1)
        g_dict['ai_cluster_match'] = cluster_msg != ""
        results.append(g_dict)

    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({"success": True, "groups": results, "ai_engine": "KMeans+Cosine"})


@app.route('/api/groups/<int:group_id>/join', methods=['POST'])
def api_join_group(group_id):
    """Rejoindre un groupe"""
    conn = get_db()
    try:
        group = conn.execute("SELECT * FROM groups WHERE id=?", (group_id,)).fetchone()
        if not group:
            return jsonify({"success": False, "message": "Groupe introuvable"}), 404

        user_id = session.get('user_id')
        if user_id:
            already = conn.execute(
                "SELECT 1 FROM group_members WHERE group_id=? AND user_id=?", (group_id, user_id)
            ).fetchone()
            if already:
                return jsonify({"success": False, "message": "Vous êtes déjà membre!"}), 400

        if group['current'] >= group['max']:
            return jsonify({"success": False, "message": "Groupe complet!"}), 400

        conn.execute("UPDATE groups SET current = current + 1 WHERE id=?", (group_id,))
        if user_id:
            conn.execute("INSERT OR IGNORE INTO group_members (group_id, user_id) VALUES (?,?)", (group_id, user_id))
            conn.execute("UPDATE users SET groups_joined = groups_joined + 1 WHERE id=?", (user_id,))
        conn.commit()
        return jsonify({"success": True, "message": f"Vous avez rejoint {group['name']}"})
    finally:
        conn.close()


@app.route('/api/my-groups')
def api_my_groups():
    """Groupes de l'utilisateur connecté"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success": True, "groups": []})

    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT DISTINCT g.* FROM groups g
               LEFT JOIN group_members gm ON g.id = gm.group_id
               WHERE gm.user_id = ? OR g.creator_id = ?
               ORDER BY g.created_at DESC""",
            (user_id, user_id)
        ).fetchall()
        return jsonify({"success": True, "groups": [dict(g) for g in rows]})
    finally:
        conn.close()


# ============================================================
# API - SERVICE 3: MATCHING COMPAGNONS (ML)
# ============================================================

@app.route('/api/find-buddies', methods=['POST'])
def api_find_buddies():
    """Prédit la destination ML et retourne les voyageurs compatibles"""
    data = request.get_json()

    try:
        age    = int(data['age'])
        sexe   = data['sexe']
        budget = int(data['budget'])
        marie  = data['marie']
        region = data['region']
        name   = data.get('name', session.get('user_name', 'Anonyme'))
    except (KeyError, ValueError) as e:
        return jsonify({"success": False, "message": f"Données invalides: {e}"}), 400

    # Prédiction ML
    predicted_dest = predire_destination(age, sexe, budget, marie, region)

    user_id = session.get('user_id')
    conn = get_db()

    # Supprimer l'ancien profil du même utilisateur connecté
    if user_id:
        conn.execute("DELETE FROM travel_profiles WHERE user_id = ?", (user_id,))

    conn.execute(
        """INSERT INTO travel_profiles
           (user_id, user_name, age, sexe, budget, marie, region, predicted_dest, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, name, age, sexe, budget, marie, region,
         predicted_dest, datetime.now().isoformat())
    )
    conn.commit()

    # Trouver les compagnons avec la même destination (exclu soi-même)
    if user_id:
        rows = conn.execute(
            """SELECT user_name, age, sexe, budget, marie, region, created_at
               FROM travel_profiles
               WHERE predicted_dest = ? AND (user_id IS NULL OR user_id != ?)
               ORDER BY created_at DESC LIMIT 20""",
            (predicted_dest, user_id)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT user_name, age, sexe, budget, marie, region, created_at
               FROM travel_profiles
               WHERE predicted_dest = ?
               ORDER BY created_at DESC LIMIT 20""",
            (predicted_dest,)
        ).fetchall()

    conn.close()

    return jsonify({
        "success": True,
        "predicted_dest": predicted_dest,
        "ml_used": ML_READY,
        "ml_accuracy": ML_ACCURACY,
        "buddies": [dict(r) for r in rows],
        "count": len(rows)
    })


@app.route('/api/buddy-stats')
def api_buddy_stats():
    """Statistiques des profils par destination"""
    conn = get_db()
    rows = conn.execute(
        "SELECT predicted_dest, COUNT(*) as total FROM travel_profiles GROUP BY predicted_dest"
    ).fetchall()
    conn.close()
    return jsonify({"success": True, "stats": [dict(r) for r in rows]})


# ============================================================
# API - ADMIN
# ============================================================

@app.route('/api/admin/login', methods=['POST'])
def api_admin_login():
    """Connexion admin"""
    data = request.get_json()
    if data.get('email') == ADMIN_EMAIL and data.get('password') == ADMIN_PASSWORD:
        session['is_admin'] = True
        return jsonify({"success": True, "message": "Bienvenue Admin!"})
    return jsonify({"success": False, "message": "Identifiants incorrects"}), 401


@app.route('/api/admin/users')
def api_admin_users():
    """Liste des utilisateurs (admin)"""
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Accès refusé"}), 403

    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, email, created_at, destinations_visited, trips_planned, groups_joined FROM users"
    ).fetchall()
    conn.close()
    users_safe = [{
        "id": r["id"], "name": r["name"], "email": r["email"],
        "createdAt": r["created_at"],
        "destinations_visited": r["destinations_visited"],
        "trips_planned": r["trips_planned"],
        "groups_joined": r["groups_joined"]
    } for r in rows]
    return jsonify({"success": True, "users": users_safe, "total": len(users_safe)})


@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
def api_admin_delete_user(user_id):
    """Supprimer un utilisateur (admin)"""
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Accès refusé"}), 403

    # ✅ AMÉLIORÉ: SQLite
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Utilisateur supprimé"})


@app.route('/api/admin/content', methods=['POST'])
def api_admin_content():
    """Modifier le contenu du site (admin)"""
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Accès refusé"}), 403

    data = request.get_json()
    admin_content['news'] = data.get('news', '')
    admin_content['welcome_msg'] = data.get('welcome_msg', '')
    return jsonify({"success": True, "message": "Contenu mis à jour!"})


@app.route('/api/admin/destinations', methods=['GET'])
def api_admin_get_destinations():
    """Liste toutes les destinations (CSV + DB)"""
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Accès refusé"}), 403

    dests = []
    # Destinations depuis CSV
    if DATA_READY:
        for _, row in villes_df.iterrows():
            dests.append({"id": None, "name": row.get("nom", ""), "type": "Ville",
                          "budget": row.get("budget", 0), "source": "csv"})
        for _, row in nature_df.iterrows():
            dests.append({"id": None, "name": row.get("nom", ""), "type": "Nature",
                          "budget": row.get("budget", 0), "source": "csv"})

    # Destinations ajoutées par l'admin (DB)
    conn = get_db()
    rows = conn.execute("SELECT * FROM destinations ORDER BY created_at DESC").fetchall()
    conn.close()
    for r in rows:
        dests.append({"id": r["id"], "name": r["name"], "type": r["type"],
                      "budget": r["budget"], "season": r["season"],
                      "description": r["description"], "source": "db"})

    return jsonify({"success": True, "destinations": dests, "total": len(dests)})


@app.route('/api/admin/destinations', methods=['POST'])
def api_admin_add_destination():
    """Ajouter une destination (admin)"""
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Accès refusé"}), 403

    data = request.get_json()
    name = data.get('name', '').strip()
    if not name:
        return jsonify({"success": False, "message": "Nom requis"}), 400

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO destinations (name, type, budget, season, description, created_at) VALUES (?,?,?,?,?,?)",
            (name, data.get('type', 'Ville'), int(data.get('budget', 0) or 0),
             data.get('season', ''), data.get('desc', ''), datetime.now().isoformat())
        )
        conn.commit()
    finally:
        conn.close()

    return jsonify({"success": True, "message": f"Destination '{name}' ajoutée!"})


@app.route('/api/admin/destinations/<int:dest_id>', methods=['DELETE'])
def api_admin_delete_destination(dest_id):
    """Supprimer une destination DB (admin)"""
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Accès refusé"}), 403

    conn = get_db()
    conn.execute("DELETE FROM destinations WHERE id = ?", (dest_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Destination supprimée"})


@app.route('/api/admin/stats')
def api_admin_stats():
    """Statistiques admin"""
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Accès refusé"}), 403

    villes_count = len(villes_df) if DATA_READY else 0
    nature_count = len(nature_df) if DATA_READY else 0

    conn = get_db()
    try:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_groups = conn.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "total_users": total_users,
        "total_destinations": villes_count + nature_count,
        "total_groups": total_groups,
        "ml_status": ML_READY,
        "ml_accuracy": f"{ML_ACCURACY}%"
    })


# ============================================================
# API - ADMIN: TOUS LES FAVORIS
# ============================================================

@app.route('/api/admin/saved-items')
def api_admin_saved_items():
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Accès refusé"}), 403
    conn = get_db()
    try:
        dest_rows = conn.execute(
            """SELECT sd.id, sd.nom, sd.type, sd.emoji, sd.budget, sd.created_at,
                      u.name as user_name, u.email as user_email
               FROM saved_destinations sd
               JOIN users u ON u.id = sd.user_id
               ORDER BY sd.created_at DESC"""
        ).fetchall()
        group_rows = conn.execute(
            """SELECT sg.id, sg.group_name, sg.destination, sg.created_at,
                      u.name as user_name, u.email as user_email
               FROM saved_groups sg
               JOIN users u ON u.id = sg.user_id
               ORDER BY sg.created_at DESC"""
        ).fetchall()
        plan_rows = conn.execute(
            """SELECT tp.id, tp.destination, tp.duree, tp.voyageurs, tp.budget_utilisateur, tp.created_at,
                      u.name as user_name, u.email as user_email
               FROM trip_plans tp
               JOIN users u ON u.id = tp.user_id
               ORDER BY tp.created_at DESC"""
        ).fetchall()
    finally:
        conn.close()
    return jsonify({
        "success": True,
        "saved_destinations": [dict(r) for r in dest_rows],
        "saved_groups": [dict(r) for r in group_rows],
        "saved_plans": [dict(r) for r in plan_rows]
    })


# ============================================================
# API - DESTINATIONS SAUVEGARDÉES (Service 1)
# ============================================================

@app.route('/api/saved-destinations', methods=['POST'])
def api_save_destination():
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Connexion requise"}), 401
    data = request.get_json()
    nom = data.get('nom', '').strip()
    if not nom:
        return jsonify({"success": False, "message": "Nom requis"}), 400
    import json as _json
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO saved_destinations (user_id, nom, type, emoji, description, budget, tags, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (session['user_id'], nom, data.get('type',''), data.get('emoji',''),
             data.get('description','')[:300], int(data.get('budget',0) or 0),
             _json.dumps(data.get('tags', []), ensure_ascii=False),
             datetime.now().isoformat())
        )
        conn.commit()
        return jsonify({"success": True, "message": f"'{nom}' sauvegardée!"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400
    finally:
        conn.close()


@app.route('/api/saved-destinations', methods=['GET'])
def api_get_saved_destinations():
    if 'user_id' not in session:
        return jsonify({"success": False, "saved": []}), 401
    conn = get_db()
    rows = conn.execute(
        "SELECT id, nom, type, emoji, description, budget, tags, created_at FROM saved_destinations WHERE user_id = ? ORDER BY created_at DESC",
        (session['user_id'],)
    ).fetchall()
    conn.close()
    items = [dict(r) for r in rows]
    return jsonify({"success": True,
                    "saved": [r['nom'] for r in items],
                    "ids": {r['nom']: r['id'] for r in items},
                    "items": items})


@app.route('/api/saved-destinations/<int:item_id>', methods=['DELETE'])
def api_delete_saved_destination(item_id):
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Connexion requise"}), 401
    conn = get_db()
    conn.execute("DELETE FROM saved_destinations WHERE id = ? AND user_id = ?",
                 (item_id, session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Destination retirée"})


# ============================================================
# API - GROUPES SAUVEGARDÉS (Service 3)
# ============================================================

@app.route('/api/saved-groups', methods=['POST'])
def api_save_group():
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Connexion requise"}), 401
    data = request.get_json()
    group_id = int(data.get('group_id', 0))
    if not group_id:
        return jsonify({"success": False, "message": "ID groupe requis"}), 400
    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO saved_groups (user_id, group_id, group_name, destination, created_at) VALUES (?,?,?,?,?)",
            (session['user_id'], group_id, data.get('group_name',''), data.get('destination',''),
             datetime.now().isoformat())
        )
        conn.commit()
        return jsonify({"success": True, "message": "Groupe sauvegardé!"})
    finally:
        conn.close()


@app.route('/api/saved-groups', methods=['GET'])
def api_get_saved_groups():
    if 'user_id' not in session:
        return jsonify({"success": False, "saved": []}), 401
    conn = get_db()
    rows = conn.execute(
        "SELECT id, group_id, group_name, destination, created_at FROM saved_groups WHERE user_id = ? ORDER BY created_at DESC",
        (session['user_id'],)
    ).fetchall()
    conn.close()
    items = [dict(r) for r in rows]
    return jsonify({"success": True,
                    "saved": [r['group_id'] for r in items],
                    "ids": {str(r['group_id']): r['id'] for r in items},
                    "items": items})


@app.route('/api/saved-groups/<int:item_id>', methods=['DELETE'])
def api_delete_saved_group(item_id):
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Connexion requise"}), 401
    conn = get_db()
    conn.execute("DELETE FROM saved_groups WHERE id = ? AND user_id = ?",
                 (item_id, session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Groupe retiré"})


# ============================================================
# API - PLANS DE VOYAGE SAUVEGARDÉS
# ============================================================

@app.route('/api/my-plans', methods=['GET'])
def api_my_plans():
    """Plans sauvegardés de l'utilisateur connecté"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Connexion requise"}), 401
    conn = get_db()
    rows = conn.execute(
        "SELECT id, destination, duree, voyageurs, budget_utilisateur, created_at FROM trip_plans WHERE user_id = ? ORDER BY created_at DESC",
        (session['user_id'],)
    ).fetchall()
    conn.close()
    return jsonify({"success": True, "plans": [dict(r) for r in rows]})


@app.route('/api/my-plans/<int:plan_id>', methods=['GET'])
def api_get_plan(plan_id):
    """Détails complets d'un plan sauvegardé"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Connexion requise"}), 401
    import json as _json
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM trip_plans WHERE id = ? AND user_id = ?",
        (plan_id, session['user_id'])
    ).fetchone()
    conn.close()
    if not row:
        return jsonify({"success": False, "message": "Plan introuvable"}), 404

    stored = _json.loads(row['plan_json'])
    return jsonify({
        "success":            True,
        "id":                 row['id'],
        "destination":        row['destination'],
        "duree":              row['duree'],
        "voyageurs":          row['voyageurs'],
        "budget_utilisateur": row['budget_utilisateur'],
        "created_at":         row['created_at'],
        "plan":               stored.get('plan', []),
        "budget_breakdown":   stored.get('budget_breakdown', {}),
        "hotel":              stored.get('hotel', {}),
        "conseils":           stored.get('conseils', []),
        "slogan":             stored.get('slogan', ''),
        "emoji":              stored.get('emoji', '📍'),
        "transport_arrivee":  stored.get('transport_arrivee', ''),
    })


@app.route('/api/my-plans/<int:plan_id>', methods=['DELETE'])
def api_delete_plan(plan_id):
    """Supprimer un plan sauvegardé"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Connexion requise"}), 401
    conn = get_db()
    conn.execute("DELETE FROM trip_plans WHERE id = ? AND user_id = ?", (plan_id, session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Plan supprimé"})


# ============================================================
# API - COMMENTAIRES UTILISATEURS
# ============================================================

@app.route('/api/comments', methods=['POST'])
def api_add_comment():
    """Ajouter un commentaire (utilisateur connecté)"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Connexion requise"}), 401

    data = request.get_json()
    destination = data.get('destination', '').strip()
    content = data.get('content', '').strip()
    rating = int(data.get('rating', 5))

    if not destination or not content:
        return jsonify({"success": False, "message": "Destination et commentaire requis"}), 400
    if len(content) < 5:
        return jsonify({"success": False, "message": "Commentaire trop court"}), 400
    if not (1 <= rating <= 5):
        rating = 5

    conn = get_db()
    try:
        # Récupérer user_name depuis session ou DB
        user_name = session.get('user_name')
        if not user_name:
            u = conn.execute("SELECT name FROM users WHERE id = ?", (session['user_id'],)).fetchone()
            user_name = u['name'] if u else 'Utilisateur'
            session['user_name'] = user_name
        conn.execute(
            "INSERT INTO comments (user_id, user_name, destination, content, rating, created_at) VALUES (?,?,?,?,?,?)",
            (session['user_id'], user_name, destination, content, rating, datetime.now().isoformat())
        )
        conn.commit()
    finally:
        conn.close()

    return jsonify({"success": True, "message": "Commentaire publié!"})


@app.route('/api/comments', methods=['GET'])
def api_get_comments():
    """Obtenir les commentaires (optionnel: filtrer par destination)"""
    destination = request.args.get('destination', '')
    conn = get_db()
    if destination:
        rows = conn.execute(
            "SELECT * FROM comments WHERE destination = ? ORDER BY created_at DESC",
            (destination,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM comments ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    conn.close()
    comments = [dict(r) for r in rows]
    return jsonify({"success": True, "comments": comments})


@app.route('/api/comments/<int:comment_id>', methods=['DELETE'])
def api_delete_own_comment(comment_id):
    """Supprimer son propre commentaire"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Connexion requise"}), 401
    conn = get_db()
    conn.execute("DELETE FROM comments WHERE id = ? AND user_id = ?",
                 (comment_id, session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Avis supprimé"})


@app.route('/api/admin/comments', methods=['GET'])
def api_admin_get_comments():
    """Tous les commentaires pour l'admin"""
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Accès refusé"}), 403

    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM comments ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return jsonify({"success": True, "comments": [dict(r) for r in rows], "total": len(rows)})


@app.route('/api/admin/comments/<int:comment_id>', methods=['DELETE'])
def api_admin_delete_comment(comment_id):
    """Supprimer un commentaire (admin)"""
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Accès refusé"}), 403

    conn = get_db()
    conn.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Commentaire supprimé"})


# ============================================================
# DÉMARRAGE
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  MAROCTOUR - Serveur Flask")
    print("=" * 50)
    print(f" ML Ready: {'[OK]' if ML_READY else '[X]'}")
    print(f" ML Accuracy: {ML_ACCURACY}%")   # ✅ AJOUT
    print(f" Data Ready: {'[OK]' if DATA_READY else '[X]'}")
    print(f" Admin: {ADMIN_EMAIL}")
    print("=" * 50)
    print("[>] Ouvrir: http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)