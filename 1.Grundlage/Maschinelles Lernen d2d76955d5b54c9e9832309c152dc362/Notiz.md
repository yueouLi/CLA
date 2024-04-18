# Maschinelles Lernen

- Anwendungen
    - N-gram Modell
    - Neuronalen Netz Modellen
    - Transformer LLMS
- Vor- und Nachteile
    - Maschinelles Lernen(Neuronale Netze)
        - Daten
        - Abdeckung
        - Annotationen
        - Statistische Kenntnisse
    - Regelbasierter Ansatz (Symbolische AI)
        - Regeln
        - Genauigkeit
        - Linguistisches Wissen
- Definitioin (ETPTPE)
    - learn experience E
    - respect to some class of tasks T
    - and performance measure P
    - if its performance at tasks in T,
    - as measured by P
    - improves with experience E
- Daten
    - Datensatz
        
        Sammlung von Instanz
        
        Bsp.
        
        ![截屏2023-06-07 18.44.44.png](Maschinelles%20Lernen%20d2d76955d5b54c9e9832309c152dc362/%25E6%2588%25AA%25E5%25B1%258F2023-06-07_18.44.44.png)
        
    - Design-Matrix
        
        "Merkmalsvektor"是指将一个对象的特征表示为向量的数学概念。在机器学习中，Merkmalsvektor通常由数字表示，每个数字对应于一个特定的特征。例如，在图像识别中，每个像素的亮度可以作为一个特征，每个像素可以表示为Merkmalsvektor的一个元素。在自然语言处理中，每个单词可以表示为Merkmalsvektor的一个元素，该元素包含有关该单词的语义和上下文信息。Merkmalsvektoren是机器学习算法的重要组成部分，它们可以用于训练模型和预测新数据的标签或属性。
        
        例如，如果您要训练一个机器学习模型来预测某个人是否患有糖尿病，您可以使用一个包含多个特征的Merkmalsvektor来描述每个人。这些特征可以包括年龄、BMI、血糖水平等。然后，您可以使用已知的数据来训练模型，并使用该模型预测新数据的标签（即该人是否患有糖尿病）。
        
        另一个例子是在自然语言处理中。假设您要使用机器学习算法来识别文本中的情感。您可以将每个单词表示为Merkmalsvektor的一个元素，并将每个元素与该单词的情感相关的数字相关联。例如，"happy"可能与正向情感相关联，而"sad"可能与负向情感相关联。然后，您可以使用这些Merkmalsvektoren来训练模型，并使用该模型预测新文本的情感。
        
        ![截屏2023-06-07 18.41.35.png](Maschinelles%20Lernen%20d2d76955d5b54c9e9832309c152dc362/%25E6%2588%25AA%25E5%25B1%258F2023-06-07_18.41.35.png)
        
        ![截屏2023-06-07 18.43.35.png](Maschinelles%20Lernen%20d2d76955d5b54c9e9832309c152dc362/%25E6%2588%25AA%25E5%25B1%258F2023-06-07_18.43.35.png)
        
    
    ![IMG_7E6187CBEC23-1.jpeg](Maschinelles%20Lernen%20d2d76955d5b54c9e9832309c152dc362/IMG_7E6187CBEC23-1.jpeg)
    
- Aufgaben/Problemklassen
    - Klassifizierung
        - Bsp.
            - Kategorisierung von Bildausschintten
                
                Merkmalsvektor: Farbteile für jedes Pixel; Davon abgeleitete Merkmale
                
                Vorfdifinerte Menge von Ausgabenkategorien
                
            - Erkennung von Spam Emails
                
                Merkmalsvektor: Hochdimensionaler Vektor mit wenigen Einträgen ≠ 0 (sparse). Jede DImension zeigt das Vorkommen eines bestimmten Wortes an.
                
        - Wichtige Konzepte:
            - **Lineares Modell**
                
                Jeder Merkmalswert wird mit einem Gewicht multipliziert, die Summe der Produkte ergibt die Vorhersage für eine Klasse.
                
            - **Fehlerfunktion**
                
                Misst, wie gut die echte Labels vorhergesagt werden.
                
            - **Test- und Trainingsdaten**
                - Auf den Großteil 70-80% der Daten wird trainiert, dabei werden 20-30% der Daten zum Test zurückgehalten. Beide Daten
                sollten die gleiche Wahrscheinlichkeitsverteilung aufweisen.
            - **Overfitting**
                
                Wenn auf den Trainingsdaten ein kleiner Fehler, aber auf den Test-Daten ein großer Fehler gemessen wird.
                
            - **Regularisierung**
                
                Verhindern von Overfitting während des Trainings, durch Einschränkung des Modells. Zum Beispiel, indem einzelne Merkmale nicht beliebig große Modell-Gewichte erhalten können.
                
    - Performanz-Maße(Fehlerfunktionen)
        - Klassifizierung (Klausur relevant)
            - Accuracy
                
                $error rate = 1 - accuracy (1: 100%)$%)
                
                ![截屏2023-06-08 09.51.02.png](Maschinelles%20Lernen%20d2d76955d5b54c9e9832309c152dc362/%25E6%2588%25AA%25E5%25B1%258F2023-06-08_09.51.02.png)
                
            - F1-Score
                
                ![截屏2023-06-08 09.53.38.png](Maschinelles%20Lernen%20d2d76955d5b54c9e9832309c152dc362/%25E6%2588%25AA%25E5%25B1%258F2023-06-08_09.53.38.png)
                
                - “relevant” : Menge der Instanzen, die das relevante Label haben
                - “retrieved”: Menge der Instanzen für die das relevante Label vorhergesagt wurde
                    
                    → Für das F-Measure muss immer ausgewählt werden, welche die relevante Kategorie ist.
                    
                
                ![截屏2023-06-08 09.53.49.png](Maschinelles%20Lernen%20d2d76955d5b54c9e9832309c152dc362/%25E6%2588%25AA%25E5%25B1%258F2023-06-08_09.53.49.png)
                
                Der **F1-Score** ist ein Maß für die Genauigkeit eines Modells, das die Präzision und den Recall
                kombiniert. Ein perfekter F1-Score ist 1 und ein schlechter F1-Score liegt nahe bei 0.
                
                Die **Präzision** misst, wie viele der vom Modell als positiv identifizierten Instanzen tatsächlich
                positiv sind. Es gibt an, wie genau das Modell in der Identifikation von positiven Instanzen ist.
                
                Der **Recall** gibt hingegen an, wie viele der tatsächlich positiven Instanzen vom Modell korrekt
                identifiziert wurden. Er gibt an, wie vollständig das Modell positive Instanzen erfasst hat.
                
                ![IMG_87895A4CBA87-1.jpeg](Maschinelles%20Lernen%20d2d76955d5b54c9e9832309c152dc362/IMG_87895A4CBA87-1.jpeg)
                
                - Bsp（um besser zu verstehen)
                    
                    苹果区🍊🍊🍊🍊🍎🍎🍎🍎🍎 ｜橙子区🍎🍎🍎🍊🍊🍊🍊🍊
                    
                    |  | apple | orange |
                    | --- | --- | --- |
                    | apple(gold) | 5 | 3 |
                    | orange(gold) | 4 | 5 |
                    
                    一共8个苹果，9个橙子
                    
                    上面那行就代表在苹果区或橙子区真正有多少，他在这个区真正有多少并不代表，他一共就有多少
                    
                    Accuracy = 10/17
                    
                    apple:
                    
                    Precision = 5/ 9
                    
                    Recall = 5/8
                    
                    f1
                    
                    orange：
                    
                    Precision：5/8
                    
                    Recall：5/9
                    
        - Ranking
            
            Mean Average Precision
            
            Spearman’s Rank Correlation
            
        - Regression
            
            • Mean Squared Error
            
        - Textüberschlappung(maschinelle Übersetzung)
            
            BLEU
            
        - Wahrscheinlichkeitsmodell
            
            Wahrscheinlichkeit der Daten (Likelihood) kann als Maß verwendet werden
            
    - Auswahl der Daten
        - Trainingsdaten 60%
        - Testdaten 20%
        - Entwicklungsdaten 20%
- Sentiment-Analysis
    - Bsp.
        
        Posted by: XYZ | Date: September 16, 2011
        
        “The new ABC camera is amazing! I bought it a few months ago and I really like it. The battery life is much longer than
        that of my previous camera. However, my partner thinks its too heavy and was too expensive.”
        
        - **Entity:** Meinung über ABC camera
        - **Aspects:** Batteriedauer, Gewicht, Preis
        - **Opinion Holder:** XYZ, XYZ’s Partner
        - **Sentiment:** Bewertung der jeweiligen Aspekte durch Opinion Holder
        - **Date:** September 16, 2011
- Perzeptron Algorithmus
    - Idee
        - Der Perzeptron-Algorithmus fällt in die Gruppe der **diskriminativen** Modelle
        - Optimiert die Qualität der Vorhersage.
        - *Gegeben die Merkmale, was ist die korrekte Klasse?*
        - P( Klasse | Merkmale ) = ?
        
    - Anpassung der Gewichte
        - Falls richtige Vorhersage für Trainings-Instanz: mache nichts
        - Sonst: Erhöhe/verringere Gewichte für jede Instanz so, dass der Score sich in die richtige Richtung verändert.
        - Gewichte für Merkmale, die besonders häufig mit einer der beiden Klassen vorkommen, erhalten viele Anpassungen in die jeweilige Richtung.
        - Die Gewichte für Merkmale, die in beiden Klassen ungefähr gleich häufig vorkommen (z.B. movie) werden mal in die eine und mal in die andere Richtung verändert
        
        在感知机算法的训练步骤中，对于每个训练样本，如果感知机对该样本的分类结果与真实标签不一致，则需要更新权重向量。对于二分类问题，权重向量的更新如下：
        
        $w = w + y * x$
        
        其中，w 是权重向量，y 是样本的真实标签，x 是样本的特征向量。
        
        实例 x 的特征为 {'the': 1, 'movie': 2, 'bad': 1}，标签为 False。
        
        权重向量为 {'movie': 2, 'very':1, 'good':-1,'bad':-1}。
        
        因为该实例的真实标签为 False，我们需要更新权重向量。
        
        根据上述更新公式，可以得到：
        
        w = w + (-1) * {'the': 1, 'movie': 2, 'bad': 1}
        
        在给定的代码中，`y` 被设置为 `-1`，因为与此更新相关的训练示例是负面示例。在感知机算法中，正面示例的标签为 `1`，而负面示例的标签为 `-1`。每个示例的权重更新取决于示例的标签和预测标签。如果预测标签与真实标签相同，则不更新权重。如果预测标签与真实标签不同，则根据预测标签和真实标签之间的差异乘以示例的特征向量来更新权重。在这种情况下，由于示例是负面的（`y` 是 `-1`），而预测标签是正面的（因为权重正在被减去），因此更新使用 `-1` 作为权重更新公式中的标签。
        
        将上式代入，得到：
        
        w = {'movie': 2, 'very':1, 'good':-1,'bad':-1} + {'the': -1, 'movie': -2, 'bad': -1}
        
        w = {'the': -1, 'movie': 0, 'very': 1, 'good': -1, 'bad': -1}
        
        但是，由于我们在这个问题中使用了一个修改后的感知机算法，它允许我们将权重向量中所有值为 0 的元素删除。因此，将权重向量 w 中的值为 0 的元素删除后，得到：
        
        w = {'the': 0, 'movie': 2, 'very': 1, 'good': -1, 'bad': 0}
        
        这就是训练步骤之后，权重向量变为 {'the': [0], 'movie': [2], 'very': [1], 'good': [-1], 'bad': [0]} 的原因。
        
        最后一步是使用更新后的权重向量来预测该实例的分类结果。
        
        在这个问题中，给定的实例 x 的特征为 {'the': 1, 'movie': 2, 'bad': 1}，标签为 False。权重向量为 {'the': 0, 'movie': 2, 'very': 1, 'good': -1, 'bad': 0}。我们使用以下公式来计算 x 的预测标签：
        
        prediction = sign(w · x)
        
        其中，w 是更新后的权重向量，· 表示点积运算，sign() 是符号函数，它返回一个数的符号，即如果数为正，则返回 1，如果数为零，则返回 0，如果数为负，则返回 -1。
        
        将 x 和 w 的点积计算如下：
        
        w · x = (0 * 1) + (2 * 2) + (1 * 1) + (-1 * 0) + (0 * 1)
        
        w · x = 5
        
        因为 w · x 是正数，所以预测标签为 True。
        
        因此，使用更新后的权重向量 {'the': 0, 'movie': 2, 'very': 1, 'good': -1, 'bad': 0}，可以正确地预测该实例的分类结果为 True。
        
    - Perzeptron-Update für eine Instanz
        
        ● Wenn das Label übereinstimmt, kein Update der Gewicht
        ● Ansonsten:
        
        - Wenn Vorhersage True und wahres Label False: error = 1
        (Verringern der Gewichte in Abhängigkeit des Merkmalswertes)
        - Wenn Vorhersage False und wahres Label True: error = -1
        (Erhöhen der Gewichte in Abhängigkeit des Merkmalswertes)
        - Update für Gewicht wj (und Merkmalsvorkommen xj(i))
        wj ← wj - error * xj(i)