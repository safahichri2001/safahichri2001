import os
import cv2
import face_recognition
import random

# Chemin vers le dossier contenant les images de visages
dataset_folder = r"C:\Users\HP\Desktop\face recognition2\saf"

# Définir les pourcentages pour l'apprentissage et les tests
train_ratio = 0.8
test_ratio = 0.2

# Listes pour stocker les encodages des visages et les noms correspondants
known_face_encodings = []
known_face_names = []

# Parcourir chaque fichier dans le dossier dataset
for filename in os.listdir(dataset_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = face_recognition.load_image_file(os.path.join(dataset_folder, filename))
        face_encoding = face_recognition.face_encodings(image)[0]
        # Ajouter l'encodage et le nom correspondant
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Diviser les données en ensembles d'apprentissage et de test
num_images = len(known_face_encodings)
num_train = int(num_images * train_ratio)
num_test = num_images - num_train

# Créer des listes d'index pour l'apprentissage et les tests
indices = list(range(num_images))
random.shuffle(indices)
train_indices = indices[:num_train]
test_indices = indices[num_train:]

# Listes finales d'encodages et de noms pour l'apprentissage et les tests
train_face_encodings = [known_face_encodings[i] for i in train_indices]
train_face_names = [known_face_names[i] for i in train_indices]
test_face_encodings = [known_face_encodings[i] for i in test_indices]
test_face_names = [known_face_names[i] for i in test_indices]

# Ouvrir la webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture de frame par frame
    ret, frame = video_capture.read()

    # Convertir l'image de BGR (OpenCV) en RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Trouver tous les visages dans le cadre
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Boucle sur chaque visage trouvé dans le cadre
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparer le visage avec les visages connus de l'ensemble d'apprentissage
        matches = face_recognition.compare_faces(train_face_encodings, face_encoding)
        name = "Inconnu"

        # Trouver la meilleure correspondance dans l'ensemble d'apprentissage
        if True in matches:
            first_match_index = matches.index(True)
            name = train_face_names[first_match_index]
        else:
            # Comparer avec l'ensemble de test si aucune correspondance n'a été trouvée dans l'apprentissage
            matches = face_recognition.compare_faces(test_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = test_face_names[first_match_index]
                print("Nouvelle personne détectée: ", name)

        # Dessiner un rectangle autour du visage
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Écrire le nom de la personne détectée
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Afficher le cadre résultant
    cv2.imshow('Video', frame)

    # Quitter le programme lorsque la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video_capture.release()
cv2.destroyAllWindows()
