|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Нейронные сети для обнаружения аномалий в многомерных временных рядах
    :Тип научной работы: M1P
    :Автор: Дмитрий Сергеевич Прокудин
    :Научный руководитель: TBA
    :Научный консультант(при наличии): н.с., Кропотов Дмитрий Александрович

Abstract
========

Задача обнаружения аномалий в многомерных временных рядах является актуальной во многих областях, например, в производстве и технологических процессах, где аномальные значения могут указывать на неисправности в системе. Данные могут быть взаимосвязаны друг с другом и иметь сложную структуру, поэтому нейронные сети хорошо подходят для решения данной задачи.  
    
В данной работе сравниваются модели со сложными архитектурами, предназначенными конкретно для прогнозирования многомерных временных рядов, такие как Autoformer и TimesNet, и модели с более простыми архитектурами, предназначенными для работы с последовательностями в общем, такие как рекуррентные нейронные сети и временные свёрточные нейронные сети. Проводится экспериментальное исследование качества прогноза и обнаружения аномалий данными моделями. Рассматривается два метода обнаружения аномалий: на основе разности предсказанных значений и значений исходного ряда и на основе вероятностного подхода. Эксперименты подтверждают применимость рассмотренных моделей и методов для решения данной задачи и показывают преимущество временных свёрточных сетей над специализированными моделями - при более простой архитектуре точность прогноза оказывается выше.

Ключевые слова: нейронные сети, многомерные временные ряды, обнаружение аномалий.

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.
