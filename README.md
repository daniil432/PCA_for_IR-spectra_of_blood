# Анализ ИК-спектров при помощи МГК и визуализация результатов
## Описание проекта
Данная программа считывает файлы формата .dpt, содержащие ИК-спектры, производит анализ спектров при помощи метода главных компонент, сравнивает спектры по характеристическим особенностям, визуализирует результаты в виде 2D и 3D точечных графиков, радиальных и столбчатых диаграмм.
Считывание файлов осуществляется при помощи pandas, что позволяет удобно редактировать и обрабаотывать спектры. Метод главных компонент реализован методом сингулярного разложения при помощи numpy. Взаимодействие с программой осуществляется через PyQt5-интерфейс. Графики отрисовываются на полотнах из matplotlib.
Дальнейшее развитие будет осуществляться путём реализации кластеризации результатов метода главных компонент, а также путём добавления возможности сохранять результаты исследований для повторого использования на графиках.

## Требования для установки
- Python версии 3.x
- Установка пакетов, перечисленных в requirements.txt

## Как использовать программу
Запуск нужно проводить из файла FirstWindow.py ```py FirstWindow.py```
Запустится интерфейс программы, в котором можно выбирать параметры обрабоки данных и путь к файлам (по умолчанию это папка input_dpt). После обработки спектров можно выбрать отображаемый тип графиков.
Для того, чтобы выбрать волновые числа, на которых будут искаться характеристические точки спектров, можно отредактировать аттрибуты в вызове метода на 78 стр. кода файла SecondWindow.py. 

## Участники
Автор проекта: Чернышев Даниил Александрович, студент магистратуры СПбГУ, факультет физики, кафедра биофизики.
Научный руководитель: Поляничко Александр Михайлович, доцент СПбГУ, факультет физики, кафедра биофизики.
