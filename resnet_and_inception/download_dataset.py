def download_dataset():
    try:
        import roboflow
        from roboflow import Roboflow
        # Initialize the Roboflow API
        rf = Roboflow(api_key="MwJCQLGjWOloVRSiezLR")
        project = rf.workspace("sameh-zaghloul").project("egyptianhieroglyphdataset")
        version = project.version(1)
        dataset = version.download("folder")
        return
    except RuntimeError:
        try :
            import subprocess
            # Exécuter une commande bash en utilisant une chaîne de caractères
            result = subprocess.run('curl -L "https://universe.roboflow.com/ds/Vl21o1OG2N?key=2HKm4ChyyD" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip', shell=True, capture_output=True, text=True)
            # Afficher la sortie
            print(result.stdout)
            return
        except RuntimeError:
            print("Impossible de télécharger le datset, voir https://universe.roboflow.com/sameh-zaghloul/egyptianhieroglyphdataset/dataset pour plus de détails.")
            return
