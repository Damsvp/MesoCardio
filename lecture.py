def parse_params(filepath):
    params = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # ignorer lignes vides
                continue

            # Séparer le label lisible du contenu après ":"
            parts = line.split(":")
            if len(parts) < 2:
                continue

            contenu = parts[-1].strip()  # tout ce qui est après le dernier ":"
            if not contenu:              # ligne sans valeur (ex: "Space length: d ,")
                continue

            # Séparer le nom de variable et les valeurs par ","
            valeurs = [v.strip().rstrip(";") for v in contenu.split(",")]
            nom = valeurs[0]             # ex: dt, T, w0, s...
            valeurs = valeurs[1:]        # le reste = les valeurs

            if not valeurs:              # pas de valeur
                params[nom] = None
                continue

            # Cas avec une seule valeur → constante
            if len(valeurs) == 1:
                v = valeurs[0]
                # Essayer de convertir en nombre
                try:
                    params[nom] = int(v)
                except ValueError:
                    try:
                        params[nom] = float(v)
                    except ValueError:
                        params[nom] = v.strip('"')  # string

            # Cas avec plusieurs valeurs → liste
            else:
                liste = []
                for v in valeurs:
                    v = v.strip('"')
                    try:
                        liste.append(int(v))
                    except ValueError:
                        try:
                            liste.append(float(v))
                        except ValueError:
                            liste.append(v)
                params[nom] = liste

    return params


# --- Chargement ---
params = parse_params("parametres.txt")

# --- Injection dans les variables locales ---
for nom, valeur in params.items():
    globals()[nom] = valeur

# --- Affichage ---
for nom, valeur in params.items():
    print(f"{nom} = {valeur}")

