<center><h1>Dossier de SVM et réseaux de neurones</h1></center>
<p align="center">
<center><h5>Carles CERDÁ VILA - M2 ECAP</h5></center>
<p align="center">
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/iae.png" alt="Logo IAE.png" style="width:400px;"/>
</p>

#### Tableau des matières
[1. Introduction](#1-introduction)<br>

[2. Analyse Exploratoire des Données (EDA)](#2-analyse-exploratoire-des-données-eda)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.1 Analyse sur les variables](#21-analyse-sur-les-variables)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[A) Analyse univariée](#a-analyse-univariée)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[B) Analyse multivariée](#b-analyse-multivariée)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.2 Valeurs manquantes](#22-valeurs-manquantes)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.3 Outliers](#23-outliers)<br>

[3. Modélisation](#3-modélisation)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.1 SVM](#31-svm)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[3.1.1 Linear SVM](#311-linear-svm)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[3.1.2 Non linear SVM](#312-non-linear-svm)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.2 Réseaux de neurones avec Keras](#32-réseaux-de-neurones-avec-keras)<br>

[4. Meilleur modèle et interprétation des résultats](#4-meilleur-modèle-et-interprétation-des-résultats)<br>

[5. Conclusions](#5-conclusions)<br>


# 1. Introduction

Les déterminants de la répartition des revenus ont toujours été en tête des questions économiques. Pourquoi deux individus perçoivent-ils des revenus différents ? Dans ce projet, nous voulons répondre à cette question, avec la particularité que nous classerons les revenus en deux types : inférieurs ou égaux à 50 mille dollars par an, et supérieurs à cela. Les données proviennent du Machine Learning Repository de l'Université de Californie, Irvine : Census Income Data Set (UC Irvine). Nous allons essayer de créer un modèle de classification capable de donner une réponse précise pour savoir si un individu avec des caractéristiques données aura un revenu supérieur ou inférieur à $50k/an. Nous utilisons deux méthodologies : Les modèles SVM et les réseaux de neurones avec la librairie Keras. 

L'article a la structure suivante : une première partie dans laquelle nous effectuons une analyse exploratoire des données (statistiques descriptives, corrélations, valeurs manquantes et outliers) ; une deuxième partie dans laquelle nous créons des modèles avec les données, une troisième partie où nous sélectionnons le modèle le plus performant et interprétons ses résultats, et une dernière partie où nous fournissons nos idées et critiques sur l'analyse.

# 2. Analyse Exploratoire des Données (EDA)

Avant de créer un modèle ML, il faut comprendre les caractéristiques des données. Pour cette raison, on peut être amené à effectuer différentes analyses sur les variables, les valeurs manquantes et les points atypiques. Cette procédure permet de visualiser la distribution des données, les variables qui sont corrélées, les valeurs manquantes et les potentiels outliers. 

## 2.1 Analyse sur les variables

L'ensemble de données initial est composé de 14 variables explicatives et d'une variable cible catégorique qui correspond au revenu. Ainsi, si une personne a un revenu supérieur à 50 000 dollars par an, cette variable prendra la valeur 1 ; 0 sinon. 

Concernant l'ensemble des variables explicatives, 8 d'entre elles sont qualitatives et les 6 restantes sont quantitatives. La liste des variables est la suivante :

- __*age*__ : Âge
- __*fnlwgt*__ : Poids final estimé
- __*education-num*__ : Années d'études
- __*capital-gain*__ : Montant du capital acquis ($)
- __*capital-loss*__ : Montant du capital perdu ($)
- __*hours-per-week*__ : Nombre d'heures travaillées par semaine
- __*workclass*__ : Classe ouvrière
- __*education*__ : Niveau de diplôme
- __*marital-status*__ : Etat civil
- __*occupation*__ : Profession
- __*relationship*__ : Type de relation
- __*race*__ : Race
- __*sex*__ : Sexe
- __*native-country*__ : Pays d'origine

Nous allons ensuite représenter leur distribution par des histogrammes (diagrammes à barres) pour les variables explicatives quantitatives (qualitatives) et un graphique de camembert pour la variable à expliquer. Ensuite, nous allons dessiner les matrices de corrélation.

#### A) Analyse univariée

<ins>*Tableau 1 : Statistiques descriptives pour les variables quantitatives*</ins>
. | age | fnlwgt | education-num | capital-gain | capital-loss | hours-per-week
--- | --- | --- |--- | --- | --- | ---
__mean__ | 	38.6 |	189778.4 |	10.1 |	1077.6 |	87.3 |	40.4
__std__ |	13.6 |	105550.0 |	2.6 |	7385.3 |	403.0 |	12.3
__min__ |	17.0 |	12285.0 |	1.0 |	0.0 |	0.0 |	1.0
__25%__ |	28.0 |	117827.0 |	9.0 |	0.0 |	0.0 |	40.0
__50%__ |	37.0 |	178356.0 |	10.0 |	0.0 |	0.0 |	40.0
__75%__ |	48.0 |	237051.0 |	12.0 |	0.0 |	0.0 |	45.0
__max__ |	90.0 |	1484705.0 |	16.0 |	99999.0 |	4356.0 |	99.0

*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

<ins>*Graphique 1 : Histogrammes des variables quantitatives*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/hist.png" />
*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

Grâce au tableau 1 et les histogrammes on peut visualiser la distribution des variables quantitatives. Par exemple, on peut tirer des conclusions telles que : l'âge moyen des individus dans l'échantillon est d'environ 39 ans, ces individus ont étudié une moyenne de 10 ans, plus de 75% de l'échantillon n'ont pas de gains ou de pertes en capital,...

Ensuite les variables qualitatives

<ins>*Graphique 2 : Histogrammes des variables qualitatives*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/bar.png" style="width:750px;"/>

*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

Après le graphique 2 on peut visualiser la distribution des variables en catégories. On peut constater, par exemple, que pour l'échantillon d'apprentissage, la grande majorité des employées sont salariés du secteur privé, et l'individu le plus courant est un homme blanc marié né aux États-Unis. Vous pouvez trouver des rapports plus détaillées dans le dans le dossier stat_desc_reports de ce _repository_.

Nous avons décidé aussi de transformer la variable __*native-country*__ pour une variable qui indique si un individu est né aux Etats-Unis ou à l'étranger. Nous faisons cela parce que nous pensons que cette division peut être plus intéressante que d'avoir tous les différents pays. Il peut être plus déterminant de savoir si quelqu'un est né aux Etats-Unis ou ailleurs que de savoir s'il est spécifiquement né au Bangladesh ou en Chine.

Pour ce qui concerne à la variable du revenu, dans l'échantillon d'apprentissage on trouve que la distribution est la suivante :

<ins>*Graphique 3 : Distribution de la variable cible*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/pierev.png" />
*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

Grâce au graphique 3 on peut dire que presque 76% de l'échantillon gagne 50 000 $ par année ou moins. Un tel déséquilibre pour la variable cible peut être problématique pour le processus de modélisation puisque les modèles que nous créons auront tendance à classer une observation dans la catégorie prédominante. Par conséquent, nous observerons peut-être des taux de faux négatifs.

#### B) Analyse multivariée

Dans cette partie on va analyser les corrélations entre variables à travers des matrices de corrélation. L'une des principales critiques de ce document est l'absence d'analyse de la corrélation entre les variables quantitatives et qualitatives, c'est-à-dire que seules les corrélations entre les variables d'un même type ont été prises en compte. L'absence de cette partie de l'analyse est principalement due au fait que nous n'avons pas réussi à coder ces corrélations.

<ins>*Tableau 2 : Matrice de corrélation pour les variables quantitatives*</ins>
.| __age__ |	__fnlwgt__ |	__education-num__ |	__capital-gain__ |	__capital-loss__ |	__hours-per-week__
--- | --- | --- | --- | --- | --- | ---
__age__ |	1.000000 |	-0.078141 |	0.066345 |	0.124948 |	0.058484 |	0.142907
__fnlwgt__ |	-0.078141 |	1.000000 |	-0.035706 |	-0.006039 |	-0.006914 |	-0.021621
__education-num__ |	0.066345 |	-0.035706 |	1.000000 |	0.119140 |	0.074749 |	0.167215
__capital-gain__ |	0.124948 |	-0.006039 |	0.119140 |	1.000000 |	-0.066569 |	0.093322
__capital-loss__ |	0.058484 |	-0.006914 |	0.074749 |	-0.066569 |	1.000000 |	0.059852
__hours-per-week__ |	0.142907 |	-0.021621 |	0.167215 |	0.093322 |	0.059852 |	1.000000

*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

<ins>*Graphique 4 : Corrélations significatives dans les variables quantitatives*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/matcor.png" />
*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

Pour ce qui concerne les variables quantitatives, aucune corrélation significative n'a été trouvée. C'est ce que l'on recherche afin d'éviter tout problème de multicollinéarité. Ce que la multicollinéarité provoque, c'est que l'on peut ne pas observer le bon impact d'une variable explicative sur la variable à expliquer puisque d'autres variables affectent cette variable explicative en particulier.

<ins>*Tableau 3 : Matrice de corrélation pour les variables qualitatives*</ins>
. |	__workclass__ |	__education__ |	__marital-status__ |	__occupation__ |	__relationship__ |	__race__ |	__sex__ |	__born__
--- | --- | --- | --- | --- | --- | --- | --- | ---
__workclass__ |	1.000000 |	0.102225 |	0.077788 |	0.216008 |	0.089828 |	0.056970 |	0.143928 |	0.053312
__education__ |	0.102225 |	1.000000 |	0.091569 |	0.197449 |	0.122654 |	0.074900 |	0.095621 |	0.299973
__marital-status__ |	0.077788 |	0.091569 |	1.000000 |	0.131502 |	0.487963 |	0.084219 |	0.461827 |	0.119983
__occupation__ |	0.216008 |	0.197449 |	0.131502 |	1.000000 |	0.177980 |	0.082798 |	0.434261 |	0.120039
__relationship__ |	0.089828 |	0.122654 |	0.487963 |	0.177980 |	1.000000 |	0.098099 |	0.649000 |	0.120730
__race__ |	0.056970 |	0.074900 |	0.084219 |	0.082798 |	0.098099 |	1.000000 |	0.118115 |	0.399158
__sex__ |	0.143928 |	0.095621 |	0.461827 |	0.434261 |	0.649000 |	0.118115 |	1.000000 |	0.001289
__born__ |	0.053312 |	0.299973 |	0.119983 |	0.120039 |	0.120730 |	0.399158 |	0.001289 |	1.000000

*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

Le tableau 3 permet de constater la relation entre les variables qualitatives. Seulement la relation entre la variable __*sex*__ et __*relationship*__ est significative. Néanmoins, comme cette corrélation est inférieure à 0,65, nous avons décidé de ne pas toucher à toutes les variables.

## 2.2 Valeurs manquantes

Il n'y a pas un grand nombre de valeurs manquantes dans les ensembles de données. Les variables ayant les concentrations les plus élevées de ces valeurs n'enregistrent qu'un 6% sur les 94% de valeurs non nulles restantes. Pour cette raison, nous remplacerons ces valeurs manquantes par le mode des variables. Dans le cas où ce nombre serait plus important, nous devrions repenser notre approche.

## 2.3 Outliers

Pour les valeurs atypiques nous avons tracé un premier graphique qui contient les boîtes à moustache pour les variables quantitatives :

<ins>*Graphique 5 : Boîtes à moustache des variables quantitatives*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/outliers1.png" />
*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

En raison de l'échelle du poids final, nous ne pouvons pas apprécier si les autres variables ont des points atypiques. Nous allons analyser dans un premier étape la variable __*fnlwgt*__, puis la variable __*capital-gain*__ et enfin toutes les variables restantes.

<ins>*Graphique 6 : Boîte à moustache pour la variable poids final estimé*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/fnlwgt.png" />

*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

On peut souligner l'abondance de valeurs aberrantes potentielles pour cette variable. Pour cette raison, nous avons effectué le test de Grubbs pour nous assurer qu'il y a au moins un point atypique dans la série. La valeur du test a été superièure à la valeur statistique pour un seuil de 5%, alors nous avons identifié ces points à travers du test ESD. 82 points sont considerés comme outliers, donc nous avons rétiré les lignes contennant telles valeurs de notre jeu de données. 

<ins>*Graphique 7 : Boîte à moustache pour la variable capital acquis*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/capital-gain.png" />

*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

Pour le cas de la variable __*capital-gain*__, nous observons clairement un point qui est éloigné des autres valeurs de la variable. Ce phénomène aurait une influence sur la partie modélisation que nous voulons éviter. Cependant, nous devons être prudents avec l'élimination des points atypiques puisque cette variable représente la plus-value, qui est censée être fortement corrélée avec la variable cible. Par conséquent, un phénomène rare dans ce cas serait de dépasser un seuil de capital gagné. Ce phénomène peut être rare mais il n'est pas forcément intelligent de l'ignorer. Nous devrions illustrer les points dans un diagramme de dispersion et voir si les plus élevés correspondent à un revenu supérieur à 50 000 dollars par an (ce que nous attendrions d'une relation positive).

<ins>*Graphique 8 : Scatterplot pour la variable capital acquis*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/scatter.png" />

*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

Comme tous les points observés au-dessus du seuil de 80 000 dollars remplissent la condition d'un revenu supérieur à $50k/an, nous ne supprimerons aucune ligne. 

<ins>*Graphique 9 : Boîtes à moustache des variables quantitatives*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/outliers2.png" />
*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

Pour le cas de la variable __*capital-loss*__, les similitudes avec la variable capital-gain sont évidentes. Cependant, les observations les plus élevées ne doivent pas nécessairement correspondre à un revenu inférieur à 50 000 $/an. En effet, les hauts revenus ont tendance à dépenser plus, de sorte que la perte en capital peut refléter cet événement. D'autre part, cela peut également indiquer le cas dans lequel quelqu'un perd un gros investissement et a donc un revenu inférieur. Néanmoins, la variance de ces valeurs n'étant pas très grande, on peut dire que ces points n'auront pas d'influence sur la partie de modélisation.

Pour les autres variables, les valeurs obtenues pour le test de Grubbs sont inférieures à la valeur critique pour un 5% (4,8057), nous ne pouvons donc pas rejeter l'hypothèse nulle de l'existence d'une outlier.

# 3. Modélisation

Dans cette partie nous allons créer des modèles de classification avec deux classes. Tout d'abord, nous utiliserons un SVM linéaire et non linéaire. Ensuite, un réseau de neurones avec la librairie Keras sera également implémenté.

## 3.1 SVM

Un SVM consiste principalement en une méthode de classification dans laquelle nous traçons des frontières afin de différencier deux ou plusieurs catégories. Si les frontières de décision sont linéaires on parle d'un *Linear SVM*. Par contre, lorsque les données ne sont pas linéaires, on trace des frontiéres non-linéaires (*Non Linear SVM*).

### 3.1.1 Linear SVM

Dans cette partie nous avons employé trois fonctions différentes dans le code :
* __Linear SVM__
* __SVM avec kernel__
* __SGD Classifier__

De la même façon, les résultats obtenus sont les suivants :

Le modèle Linear SVM a une accuray de 0.8236702967800165 et un std : 0.011556249345156241
Le modèle SVM avec kernel a une accuray de 0.8192057456095443 et un std : 0.012651212894937926
Le modèle SGD Classifier a une accuray de 0.8149571667481996 et un std : 0.0047230804966721515

Nous observons que le modèle le plus performant a été le *Linear SVM* avec une précision de 82,37%. Nous allons chercher les meilleurs paramètres de ce modèle mais aussi du SGDClassifier afin d'avoir un contraste.

La meilleur combinaison d'estimateur est: {'C': 10, 'loss': 'squared_hinge', 'max_iter': 2000} pour un score de: 0.8239475896410353 avec le modèle *Linear SVM*.

<ins>*Graphique 10 : Learning curve pour le modèle Linear SVM*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/learning.png" />

*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

On obtient aussi la matrice de confusion suivante :

<ins>*Graphique 11 : Matrice de confusion pour le modèle Linear SVM*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/matconf.png" />

*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

Nous pensons qu'en raison des valeurs déséquilibrées de la variable cible, notre modèle a du mal à prédire les individus qui gagnent plus de 50 000 dollars/an. Le taux de faux négatifs est égal à 0,59. Par conséquent, chaque fois que nous entrons dans une nouvelle observation, le modèle prédira très probablement qu'il gagne moins de 50 000 dollars/an. Une bonne solution pourrait être de rééquilibrer l'échantillon.

Pour le modèle *SGD Classifier* nous obtenons même les pires résultats. Le taux de faux négatifs atteint 0,72. Si nous devions choisir un modèle, nous opterions pour le LinearSVC, même si les résultats ne semblent pas très satisfaisants. Vous pouvez observer la demarche pour le *SGD Classifier* sur le jupyter notebook.


### 3.1.2 Non linear SVM

Dans cette partie nous avons employé deux fonctions différentes dans le code :
* __Polynomial kernel__
* __Radial Basis Function kernel__

Les résultats obtenus :

Le modèle Polynomial kernel a une accuray de 0.8370022649101052 et un std : 0.00987210433183708
Le modèle Radial Basis Function kernel a une accuray de 0.8468237645858693 et un std : 0.010512970037405624

Les deux modèles semblent mieux fonctionner que les modèles SVM linéaires. Nous avons choisi pour notre analyse le modèle Radial Basis Function puisqu'il a obtenu un meilleur score.

Avec la meilleur combinaison d'estimateur : {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}, le score du modèle est : 0.8456231756137309.

<ins>*Graphique 12 : Learning curve pour le modèle Radial Basis Function kernel*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/learning2.png" />

*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

Cette fois, les curves de validation et d'appresntissage ne convergent pas, cela veut dire que le modèle a encore besoin de plus d'observations pour améliorer sa précision. Autrement dite, le modèle a de sous-apprentissage. 

<ins>*Graphique 13 : Matrice de confusion pour le modèle Radial Basis Function kernel*</ins>
<img src="https://github.com/carlescvila/Dossier_SVM_et_reseaux_de_neurones/blob/main/img/matconf2.png" />

*Source : Dossier de SVM et Réseaux de Neurones, Carles CERDÁ VILA*

On retrouve encore le problème de la classification des faux négatifs (0,45). Néanmoins, ce modèle est plus performant que les SVM linéaires en termes de précision. En revanche, les courbes d'apprentissage ne convergent pas, ce qui signifie que le modèle est sous-ajusté. Cela devrait nécessiter un ensemble d'entraînement plus important afin de réduire la variance. En revanche, le modèle classe bien les individus avec un revenu de 50 000 dollars par an ou inferieur.

## 3.2 Réseaux de neurones avec Keras

Les réseaux neuronaux fonctionnent de la manière suivante : ils observent une entrée et, par le biais de plusieurs couches, ils traitent l'information afin de fournir une sortie. La procédure Keras est une de les plus implementées à cause de sa flexibilitée.

Nous avons un premier modèle avec seulement une couche cachée composée de 300 neurones et une deuxième composée de 100. On a 34701 paramètres dans ce réseau de neurones. La couche output contient 1 biais plus les 100 neurones de la couche cachée précédente. Néanmoins, ni ce modèle ni aucun autre que nous avons construit n'a fonctionné pour nous. Il n'a pu classer aucune observation dans la catégorie des plus de 50000 $. Nous ne comprenons pas cette erreur. C'est pourquoi nous faisons un point avec les résultats de la partie précédente.

# 4. Meilleur modèle et interprétation des résultats

# 5. Conclusions



