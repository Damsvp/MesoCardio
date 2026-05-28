import importlib

def convert_to_function(param_list):
    """
    Prend une liste ['categorie', 'type', param1, param2, ...]
    Retourne la fonction correspondante
    """
    if not isinstance(param_list, list):
        return param_list  # si c'est une constante, on ne touche pas
    
    # Vérifier que le premier élément est une catégorie connue
    categories_connues = ["potential", "kernel", "dynamic"]
    
    if param_list[0] not in categories_connues:
        return param_list  # pas une fonction, on laisse tel quel
    
    categorie  = param_list[0]          # "potential"
    type_      = param_list[1]          # "gaussienne"
    parametres = param_list[2:]         # [5, 1]

    # Importer dynamiquement le bon fichier
    try:
        module = importlib.import_module(f"function.{categorie}")
    except ModuleNotFoundError:
        raise ImportError(f"Fichier 'function/{categorie}.py' introuvable")

    # Récupérer la bonne fonction
    try:
        func = getattr(module, type_)
    except AttributeError:
        raise ValueError(f"Fonction '{type_}' non définie dans '{categorie}.py'")

    return func(*parametres)

# Exemple d'utilisation
# w0=convert_to_function(["potential", "gaussienne", 5, 1])
# print(w0(4)) 