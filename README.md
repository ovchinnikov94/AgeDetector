# AgeDetector
Программа решает задачу машинного обучения - классификации текстов по возрастным группам.
В программе была использована библиотека scikit-learn. 
Для обучения нужно вызвать метод класса AgeDetector train(texts, labels), принимающий на вход тексты и метки к текстам. 
В данной реализации используется классификатор PassiveAgressive с принименением OneVsRestClassifier для обеспечения многоклассовой классификации. Для извлечения текстов испоьлуется CountVectorizer() из библиотеки scikit-learn. 

This script implements one of the machinelearning problems - classification texts by age groups. 
It was implemented using scikit-learn library on Python3.5.
For training model, please, use train(texts, labels) method in AgeDetector class. 
OneVsRestClassifier with PassiveAgressiveClassifier are used to solve this problem. 
For extracting features from text is using CountVectorizer() with (1, 8) ngrams. 
