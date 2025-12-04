# Maturarbeit_Efraim_Sidiropoulos
In diesem File wird der Aufbauu des Githubs erläutert und eine Benutzungsanleitung zugefügt.

Es gibt 2 Python-Files. train.py und hyperparameter_analysis.py.1 train.py ist das Trainingsskript für das Standartmodell und hyperparameter_analysis.py ist das Skript für die Durchführung der Hyperparameter Experimente. Im Code sind auch Hashtag-Kommentare zu finden, welche manche Codeblöcke erklären.

Die Dateien stanford_train_yolo.json und der Imagefolder stanford_cars_train bilden den benötigten Datensatz. Diese sollten heruntergeldaden werden und am Besten separat im lokalen Downloads-Folder liegen.

Die Folder hyperparameter results und Standartmodell results beinhalten meine Auswertung aus den durchgeführten Trainings. Führt man das Training jedoch durch erhält man eigene neue Dateien.

Das Training wurde auf Windows Powershell durchgeführt und darauf basiert auch das Vorgehen. Dazu wird auch eine NVIDIA-Grafikskarte bevorzugt. Im Falle, dass diese nicht vorhanden ist, wird automatisch der CPU benutzt. Dabei sollte man jedoch mit einer steilen Erhöhung der Trainingsdauer rechnen.

Der requirements Folder weist manche Pakete auf, welche für das Training benötigt sind(wird in der Anleitung erklärt)

Anleitung Powershell:

1. Alle FIles (Python Skripte, Image-Folder und JSON-FIle) unter dem lokalen Downloadsfolder haben.
2. Requirements installieren -> einfach copy-paste un enter
3. In Powershell eingeben: cd (der eigene Pfad zum Downloads Folder. Dieser ist zusehen, falls man auf z.B das train.py geht und Properties drückt) -> in meinem Fall war es : cd C:\Users\efrem\Downloads
4. Danach noch zum Schluss das Skript laufen lassen: py train.py oder py hyperparameter_analysis.py


