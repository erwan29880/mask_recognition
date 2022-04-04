# mask_recognition  

Le but est de créer une apllication de reconnaissance de masque à partir d'une webcam. Pour ce faire, un modèle d'intelligence artificielle est entraîné par Data Augmentation et Transfer Learning (vgg16 avec les poids imagenet).  
Nous disposons d'environ 1500 données réparties en jeu d'entraînement, jeu de test et jeu de validation.

Nous traçons suite à l'entraînement les courbes d'apprentissage afin de vérifier si le modèle est en over-fitting. L'accuracy du modèle dépasse les 93%, le test en application donne de bons résultats.

L'apprentissage se fait via tensorflow-keras sur Google Colab; sur GPU.  
Le modèle est enregistré en fichier h5.




## Dans ce dossier il y a :

|-- le fichier pour l'apprentissage du modèle d'IA     
|-- le fichier pour l'application qui découle de l'apprentissage   
|-- un dossier d'images de validation       

Les images d'apprentissage ne sont pas disponibles sur ce dépôt.  
Le fichier du modèle est trop volumineux pour github.
