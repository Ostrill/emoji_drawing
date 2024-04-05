from PIL import Image, ImageFont, ImageDraw
from pilmoji import Pilmoji
import numpy as np
import pickle


#===============================================================
#=========== СОЗДАНИЕ СЛОВАРЕЙ И РИСОВАНИЕ КАРТИНОК ============
#===============================================================


def create_emoji_dict(emojis, 
                      background_color=None,
                      emoji_resolution=100, 
                      disp=True):
    """
    Создать словарь смайликов, где каждому смайлику
    соответствует подходящий ему цвет и доля занимаемой
    им площади в квадратной ячейке

    PARAMETERS
    ----------
    emojis : list[str]
        | список из смайликов, из которых создается 
          словарь
    
    background_color : (int, int, int) или None
        | фоновый цвет для смайликов в формате RGB, 
          если None, используется прозрачный фон
    
    emoji_resolution : int
        | каждый смайлик рассматривается как квадрат,
          этот параметр задает размер стороны квадрата.
          Влияет на точность определения среднего цвета
          смайлика, рекомендуется просто оставить 100
    
    disp : bool
        | выводить ли прогресс создания словаря
          (шкала в процентах)
    """
    emoji_dict = {}

    # Формирование RGBA цвета фона с A=0
    img_back = (*(background_color or [0]*3), 0)
    # Размер картинки, в которой будут рисоваться смайлики
    img_size = (emoji_resolution, emoji_resolution)

    # Цикл по всем смайликам
    for i, emoji in enumerate(emojis, 1):
        # Создание новой картинки со смайликом
        with Image.new('RGBA', img_size, img_back) as image:
            # Шрифт для отображения смайлика
            font = ImageFont.truetype('fonts/arial.ttf', emoji_resolution)
            # Рисование смайлика на картинке
            with Pilmoji(image) as pilmoji:
                pilmoji.text((0, 0), emoji, font=font)

        # Веса каждого пикселя (насколько каждый непрозрачен)
        np_image = np.array(image)
        weights = np_image[:, :, 3:] / 255
        # Доля занимаемой смайликом площади в квадратной ячейке
        area = weights.sum() / (emoji_resolution ** 2)
        
        # Вычисление среднего цвета получившейся картинки смайлика
        if background_color is None:
            # Средний цвет только по непрозрачным пикселям
            avg_color = (np_image[:, :, :3] * weights).sum(axis=(0, 1)) / weights.sum()
        else:
            # Средний цвет по каналам RGB (без учета прозрачности)
            avg_color = np.array(image)[:, :, :3].mean(axis=(0, 1))

        # Добавление смайлика в словарь
        emoji_dict[emoji] = (avg_color, area)

        # Вывод прогресса
        if disp:
            progress = f'{round(i / len(emojis) * 100, 2)}%'
            print(f'\rПостроение словаря: {progress:<7} ({i}/{len(emojis)})', end='')
    if disp:
        print()

    return emoji_dict


def load_emoji_list(list_name):
    """
    Загрузить список из смайликов {list_name}, лежащий в
    папке emoji_sources/
    """
    with open(f'emoji_sources/{list_name}.txt', 'r') as file:
        emoji_list = [line.strip() for line in file]
    return emoji_list
    

def load_emoji_dict(dict_name):
    """
    Загрузить словарь смайликов {dict_name}, лежащий в
    папке emoji_dicts/
    """
    with open(f'emoji_dicts/{dict_name}.pkl', 'rb') as file:
        emoji_dict = pickle.load(file)
    return emoji_dict


def calc_color_diff(rgb1, rgb2):
    """Расстояние между двумя цветами"""
    return np.linalg.norm(rgb1 - rgb2)


def get_nearest_emoji(rgb, emoji_dict):
    """Найти ближайший по цвету смайлик"""
    return min(emoji_dict.keys(), 
               key=lambda k: calc_color_diff(rgb, emoji_dict[k][0]))


def build_emoji_matrix(filename, width, emoji_dict, disp=True):
    """
    Нарисовать картинку из смайликов-квадратов в виде
    2D-numpy-массива (матрицы)

    PARAMETERS
    ----------
    filename : str
        | название файла картинки, лежащей в папке input/, 
          которую необходимо нарисовать в виде смайликов
    
    width : int
        | ширина итоговой картинки из смайликов

    emoji_dict : dict
        | специальный словарь смайликов, где каждому
          смайлику соответствует назначенный ему цвет
          в формате numpy-массива из трех значений (RGB)
          и доля занимаемой в квадрате площади
    
    disp : bool
        | отображать ли результат и логи
    
    RETURNS
    -------
    emoji_matrix : 2D-ndarray[str]
        | картинка из смайликов в виде 2D numpy-массива
    """    
    # Загрузка картинки
    image = Image.open(f'input/{filename}')

    # Вывод размеров картинки
    if disp:
        source_w, source_h = image.size
        print(f'Исходная картинка: {source_h}x{source_w}')
        
    # Уменьшение картинки, преобразование в numpy
    # и сохранение только трех каналов под R, G, B
    image.thumbnail((width, -1))
    image = np.array(image)[:, :, :3]
    
    # Размеры картинки
    img_h, img_w, _ = image.shape
    
    # Вывод размеров картинки
    if disp:
        print(f'Из смайликов:      {img_h}x{img_w}')
    
    # Матрица из смайликов
    emoji_matrix = np.full((img_h, img_w), '', dtype='<U2')
            
    for i in range(img_h):
        for j in range(img_w):
            # Поиск подходящего смайлика и добавление его в матрицу
            nearest_emoji = get_nearest_emoji(image[i, j], emoji_dict)
            emoji_matrix[i, j] = nearest_emoji

            # Вывод прогресса
            if disp:
                total_px = img_h * img_w
                current_px = i * img_w + j + 1
                progress = f'{round(current_px / total_px * 100, 2)}%'
                print(f'\rФормирование матрицы: {progress:<7}', end='')
    if disp:
        print()
    
    return emoji_matrix


def save_as_text(emoji_matrix, image_name):
    """
    Сохранить результат (матрицу смайликов) в файл
    в виде текста

    PARAMETERS
    ----------
    emoji_matrix : 2D-ndarray[str]
        | двумерный numpy-массив (матрица) из смайликов
    
    image_name : str
        | название картинки, текстовый файл для 
          которой будет сохранен в папку output/
    """
    with open(f'output/{image_name}.txt', 'w') as file:
        as_string = '\n'.join(''.join(line) 
                              for line in emoji_matrix)
        file.write(as_string)


def save_as_image(emoji_matrix, 
                  image_name, 
                  emoji_resolution, 
                  background_color=None, 
                  disp=False):
    """
    Сохранить результат (матрицу смайликов) в файл
    в виде картинки формата PNG

    PARAMETERS
    ----------
    emoji_matrix : ndarray[str]
        | 2D numpy-массив (матрица) из смайликов
    
    image_name : str
        | название картинки, png-файл для 
          которой будет сохранен в папку output/

    emoji_resolution : int
        | размер каждого смайлика на итоговой картинке
          в пикселях (каждый смайлик рисуется в квадратной 
          области, и этот параметр задает сторону квадрата)

    background_color : (int, int, int) или None или 'dynamic'
        | цвет фона у итоговой картинки:
          - если задан кортеж из трех чисел, фон будет окрашен
            в соответствующий RGB-цвет;
          - если None, фон будет прозрачным
    
    disp : bool
        | выводить ли прогресс сохранения картинки
    """
    # Высота и ширина массива из смайликов
    emojis_h, emojis_w = emoji_matrix.shape
    # Высота и ширина итоговой картинки из смайликов
    img_h, img_w = np.array(emoji_matrix.shape) * emoji_resolution

    # Определение цвета фона для разных режимов
    dynamic_mode = False
    if background_color is None:
        img_back = (0, 0, 0, 0)
    else:
        img_back = (*background_color, 255)
    
    # Создание картинки из смайликов
    with Image.new('RGBA', (img_w, img_h), img_back) as image:
        # Шрифт для отображения смайликов
        font = ImageFont.truetype('fonts/arial.ttf', emoji_resolution)

        # Рисование смайликов на картинке
        with Pilmoji(image) as pilmoji:
            for i in range(emojis_w):
                for j in range(emojis_h):
                    # Индексы пикселей в итоговой картинке
                    w, h = i * emoji_resolution, j * emoji_resolution
                    # Отрисовка смайлика
                    pilmoji.text((w, h), emoji_matrix[j, i], font=font)
                    
                    # Вывод прогресса
                    if disp:
                        total_px = emojis_h * emojis_w
                        current_px = i * emojis_h + j + 1
                        progress = f'{round(current_px / total_px * 100, 2)}%'
                        print(f'\rРисование изображения: {progress:<7}', end='')
            if disp:
                print()

    image.save(f'output/{image_name}.png')
    return image
    

def draw(image_filename,
         save_image_name,
         emoji_width=50,
         emoji_resolution=50,
         background_color=(0, 0, 0),
         emojis=None, 
         emoji_dict_name='classic_black', 
         disp=True):
    """
    Главная функция для рисования картинки

    PARAMETERS
    ----------
    image_filename : str
        | название файла картинки, которую необходимо 
          нарисовать смайликами. Файл обязатнльно должен 
          находиться в папке input/ 
    
    save_image_name : str
        | имя картинки для сохранения
    
    emoji_width : int
        | ширина итоговой картинки (в смайликах)
    
    emoji_resolution : int
        | каждый смайлик рисуется в квадратной области, 
          и этот параметр задает размер стороны квадрата
          в пикселях
    
    background_color : (int, int, int)
        | фоновый цвет итоговой картинки из смайликов
    
    emojis : str или None
        | смайлики, которыми необходимо нарисовать картинку:
          - если указана строка, состоящаяя из смайликов через
            пробел, то для них будет построен словарь, на основе 
            которого и будет рисоваться картинка (при этом 
            параметр emoji_dict не учитывается);
          - если None, в качестве словаря смайликов будет
            использоваться сохраненный в папке emoji_dicts/
            словарь с названием, указанным в параметре
            emoji_dict
    
    emoji_dict_name : str или None
        | название словаря смайликов из папки emoji_dicts/
          - если указана строка, а параметр emojis = None,
            то для рисования используется указанный словарь;
          - если None, для рисования используются смайлики из
            параметра emojis (но если этот параметр = None,
            будет выброшена ошибка)
    
    disp : bool
        | выводить ли прогресс создания картинки
    """
    # Проверка, задан ли набор смайликов, которыми нужно рисовать
    error_text = 'Необходимо указать либо emojis, либо emojis_dict!'
    assert not(emojis is None and emoji_dict_name is None), error_text

    # Подготовка словаря смайликов
    if emojis is None:
        emoji_dict = load_emoji_dict(emoji_dict_name)
    else:
        # Для построения словаря используется то же разрешение
        # (размер смайлика), что и для итоговой картинки, но
        # это необязательно, можно просто установить константу
        emoji_dict = create_emoji_dict(emojis=emojis.split(' '), 
                                       background_color=background_color,
                                       emoji_resolution=emoji_resolution, 
                                       disp=disp)
        
    # Построение двумерной матрицы из смайликов
    emoji_matrix = build_emoji_matrix(filename=image_filename, 
                                      width=emoji_width, 
                                      emoji_dict=emoji_dict, 
                                      disp=disp)

    # Сохранение в виде текста
    save_as_text(emoji_matrix=emoji_matrix, 
                 image_name=save_image_name)
    if disp:
        print('Изображение в виде текста сохранено')

    # Сохранение в виде изображения
    result_image = save_as_image(emoji_matrix=emoji_matrix, 
                                 image_name=save_image_name, 
                                 emoji_resolution=emoji_resolution, 
                                 background_color=background_color, 
                                 disp=disp)
    if disp:
        print('Изображение в png-формате сохранено')
        
    return result_image


#===============================================================
#=========== РИСОВАНИЕ КАРТИНОК С ДИНАМИЧЕСКИМ ФОНОМ ===========
#===============================================================


def get_nearest_background(target_rgb, emoji_rgb, emoji_area):
    """
    Найти цвет, в который нужно покрасить фон смайлика, 
    чтобы у всей квадратной ячейки средний цвет стал наиболее 
    близким к target_rgb

    PARAMETERS
    ----------
    target_rgb : (int, int, int)
        | средний цвет, который должен получиться после
          подбора цвета фона
    
    emoji_rgb : (int, int, int)
        | средний цвет смайлика
    
    emoji_area : float
        | доля занимаемой смайликом площади в квадратной
          ячейке (от 0 до 1)
    """
    target = np.array(target_rgb)
    emoji = np.array(emoji_rgb)
    calculated = (target - emoji * emoji_area) / (1 - emoji_area)
    # Иногда значения цветов выходят за рамки [0, 255], 
    # поэтому их нужно обрезать:
    return calculated.clip(0, 255)


def get_nearest_emoji_and_back(rgb, emoji_dict):
    """
    Найти ближайшие по цвету сочетание смайлика с фоном

    PARAMETERS
    ----------
    rgb : (int, int, int)
        | цвет, к которому требуется подобрать наиболее
          похожее на него сочетание смайлика и фона
    
    emoji_dict : 
        | используемый словарь смайликов
    
    RETURNS
    -------
    tuple(
        best_emoji : str
            | наиболее подходящий смайлик из словаря
        
        emoji_back : 1D-ndarray[float]
            | RGB-цвет фона для смайликав формате numpy-массива
    )
    """
    def _calc_nearest_color_diff(emoji):
        """
        Внутренняя функция для определения расстояния до максимально 
        близкого к target среднего цвета из сочетания смайлика
        с каким-либо цветным фоном
        """
        emoji_rgb, emoji_area = emoji_dict[emoji]
        # Поиск наиболее подходящего цвета фона
        nearest_back = get_nearest_background(target_rgb=rgb, 
                                              emoji_rgb=emoji_rgb, 
                                              emoji_area=emoji_area)
        # Общий средний цвет сочетания смайлика и фона
        avg = emoji_rgb*emoji_area + nearest_back*(1 - emoji_area)
        return calc_color_diff(rgb, avg)
        
    best_emoji = min(emoji_dict.keys(), key=_calc_nearest_color_diff)
    emoji_back = get_nearest_background(rgb, *emoji_dict[best_emoji])
    return best_emoji, emoji_back


def draw_dynamic(image_filename,
                 save_image_name,
                 emoji_width=50,
                 emoji_resolution=50,
                 emojis=None, 
                 emoji_dict_name='classic_black', 
                 disp=True):
    """
    Функция для рисования картинки с динамическим фоном
    (для каждого смайлика подбирается собственный цвет фона)

    PARAMETERS
    ----------
    image_filename : str
        | название файла картинки, которую необходимо 
          нарисовать смайликами. Файл обязатнльно должен 
          находиться в папке input/ 
    
    save_image_name : str
        | имя картинки для сохранения
    
    emoji_width : int
        | ширина итоговой картинки (в смайликах)
    
    emoji_resolution : int
        | каждый смайлик рисуется в квадратной области, 
          и этот параметр задает размер стороны квадрата
          в пикселях
    
    emojis : str или None
        | смайлики, которыми необходимо нарисовать картинку:
          - если указана строка, состоящаяя из смайликов через
            пробел, то для них будет построен словарь, на основе 
            которого и будет рисоваться картинка (при этом 
            параметр emoji_dict не учитывается);
          - если None, в качестве словаря смайликов будет
            использоваться сохраненный в папке emoji_dicts/
            словарь с названием, указанным в параметре
            emoji_dict
    
    emoji_dict_name : str или None
        | название словаря смайликов из папки emoji_dicts/
          - если указана строка, а параметр emojis = None,
            то для рисования используется указанный словарь;
          - если None, для рисования используются смайлики из
            параметра emojis (но если этот параметр = None,
            будет выброшена ошибка)
    
    disp : bool
        | выводить ли прогресс создания картинки
    """
    # Проверка, задан ли набор смайликов, которыми нужно рисовать
    error_text = 'Необходимо указать либо emojis, либо emojis_dict!'
    assert not(emojis is None and emoji_dict_name is None), error_text

    # Подготовка словаря смайликов
    if emojis is None:
        emoji_dict = load_emoji_dict(emoji_dict_name)
    else:
        # Для построения словаря используется то же разрешение
        # (размер смайлика), что и для итоговой картинки, но
        # это необязательно, можно просто установить константу
        emoji_dict = create_emoji_dict(emojis=emojis.split(' '), 
                                       background_color=None,
                                       emoji_resolution=emoji_resolution, 
                                       disp=disp)
    
    # Загрузка картинки-исходника
    image = Image.open(f'input/{image_filename}')

    # Вывод размеров картинки до уменьшения
    if disp:
        source_w, source_h = image.size
        print(f'Размер картинки: {source_h}x{source_w}', end='')
        
    # Уменьшение картинки, преобразование в numpy
    # и сохранение только трех каналов под R, G, B
    image.thumbnail((emoji_width, -1))
    image = np.array(image)[:, :, :3]
    
    # Размеры матрицы смайликов
    img_h, img_w, _ = image.shape
    # Размеры итоговой картинки
    result_h, result_w = img_h*emoji_resolution, img_w*emoji_resolution
    
    # Вывод размеров картинки после уменьшения
    if disp:
        print(f' → {img_h}x{img_w}')

    # Создание итогового изображения result_image
    with Image.new('RGBA', (result_w, result_h), (0,)*4) as result_image:
        # Компонент для рисования на картинке
        img_draw = ImageDraw.Draw(result_image)
        # Шрифт для отображения смайликов
        font = ImageFont.truetype('fonts/arial.ttf', emoji_resolution)
        with Pilmoji(result_image) as pilmoji:
            for i in range(img_h):
                for j in range(img_w):
                    # Поиск подходящего сочетания смайлика с фоном
                    emoji, back = get_nearest_emoji_and_back(image[i, j], 
                                                             emoji_dict)
                    # Индексы пикселей в итоговой картинке
                    h, w = i * emoji_resolution, j * emoji_resolution
                    # Отрисовка прямоугольника (фона для смайлика)
                    img_draw.rectangle((w, h, w+emoji_resolution, h+emoji_resolution), 
                                       fill=tuple(np.rint(back).astype(int)))
                    # Отрисовка смайлика
                    pilmoji.text((w, h), emoji, font=font)
        
                    # Вывод прогресса
                    if disp:
                        total_px = img_h * img_w
                        current_px = i * img_w + j + 1
                        progress = f'{round(current_px / total_px * 100, 2)}%'
                        print(f'\rРисование: {progress:<7}', end='')
            if disp:
                print()

    # Сохранение изображения
    result_image.save(f'output/{save_image_name}.png')

    if disp:
        print('Изображение в png-формате сохранено')
        
    return result_image













def draw_ultra(image_filename,
               save_image_name,
               emoji_width=50,
               emoji_resolution=50,
               background_color=(0, 0, 0),
               emojis=None, 
               emoji_dict_name='classic_black', 
               dynamic_background=False,
               disp=True):
    """
    Функция для рисования картинки с динамическим фоном
    (для каждого смайлика подбирается собственный цвет фона)

    PARAMETERS
    ----------
    image_filename : str
        | название файла картинки, которую необходимо 
          нарисовать смайликами. Файл обязатнльно должен 
          находиться в папке input/ 
    
    save_image_name : str
        | имя картинки для сохранения
    
    emoji_width : int
        | ширина итоговой картинки (в смайликах)
    
    emoji_resolution : int
        | каждый смайлик рисуется в квадратной области, 
          и этот параметр задает размер стороны квадрата
          в пикселях
    
    background_color : (int, int, int)
        | фоновый цвет итоговой картинки из смайликов
    
    emojis : str или None
        | смайлики, которыми необходимо нарисовать картинку:
          - если указана строка, состоящаяя из смайликов через
            пробел, то для них будет построен словарь, на основе 
            которого и будет рисоваться картинка (при этом 
            параметр emoji_dict не учитывается);
          - если None, в качестве словаря смайликов будет
            использоваться сохраненный в папке emoji_dicts/
            словарь с названием, указанным в параметре
            emoji_dict
    
    emoji_dict_name : str или None
        | название словаря смайликов из папки emoji_dicts/
          - если указана строка, а параметр emojis = None,
            то для рисования используется указанный словарь;
          - если None, для рисования используются смайлики из
            параметра emojis (но если этот параметр = None,
            будет выброшена ошибка)

    dynamic_background : bool
        | использовать ли динамическое создание фона, при
          котором для каждого смайлика отрисовывается собственный 
          цвет фона, в сочетании с которым он становится еще ближе
          к исходной картинке
    
    disp : bool
        | выводить ли прогресс создания картинки
    """
    # Проверка, задан ли набор смайликов, которыми нужно рисовать
    error_text = 'Необходимо указать либо emojis, либо emojis_dict!'
    assert not(emojis is None and emoji_dict_name is None), error_text

    # Подготовка словаря смайликов
    if emojis is None:
        emoji_dict = load_emoji_dict(emoji_dict_name)
    else:
        # Для построения словаря используется то же разрешение
        # (размер смайлика), что и для итоговой картинки, но
        # это необязательно, можно просто установить константу
        emoji_dict = create_emoji_dict(emojis=emojis.split(' '), 
                                       background_color=None,
                                       emoji_resolution=emoji_resolution, 
                                       disp=disp)
    
    # Загрузка картинки-исходника
    image = Image.open(f'input/{image_filename}')

    # Вывод размеров картинки до уменьшения
    if disp:
        source_w, source_h = image.size
        print(f'Размер картинки: {source_h}x{source_w}', end='')
        
    # Уменьшение картинки, преобразование в numpy
    # и сохранение только трех каналов под R, G, B
    image.thumbnail((emoji_width, -1))
    image = np.array(image)[:, :, :3]
    
    # Размеры матрицы смайликов
    img_h, img_w, _ = image.shape
    # Размеры итоговой картинки
    result_h, result_w = img_h*emoji_resolution, img_w*emoji_resolution
    
    # Вывод размеров картинки после уменьшения
    if disp:
        print(f' → {img_h}x{img_w}')

    # Текстовое представление картинки из смайликов
    emoji_image_text = ''

    # Установка цвета для итоговой картинки
    if background_color is None:
        background_RGBA = (0, 0, 0, 0)
    else:
        background_RGBA = (*background_color, 255)
        
    # Создание итогового изображения result_image
    with Image.new('RGBA', (result_w, result_h), background_RGBA) as result_image:
        # Компонент для рисования на картинке
        img_draw = ImageDraw.Draw(result_image)
        # Шрифт для отображения смайликов
        font = ImageFont.truetype('fonts/arial.ttf', emoji_resolution)
        with Pilmoji(result_image) as pilmoji:
            for i in range(img_h):
                for j in range(img_w):
                    # Индексы пикселей в итоговой картинке
                    h, w = i * emoji_resolution, j * emoji_resolution

                    if dynamic_background:
                        # Поиск подходящего сочетания смайлика с фоном
                        emoji, back = get_nearest_emoji_and_back(image[i, j], emoji_dict)
                        # Отрисовка прямоугольника (фона для смайлика)
                        img_draw.rectangle((w, h, w+emoji_resolution, h+emoji_resolution), 
                                           fill=tuple(np.rint(back).astype(int)))
                    else:
                        # Поиск ближайшего по среднему цвету смайлика
                        emoji = get_nearest_emoji(image[i, j], emoji_dict)
                        
                    # Отрисовка смайлика
                    pilmoji.text((w, h), emoji, font=font)
                    # Добавление смайлика в текстовое представление картинки
                    emoji_image_text += emoji
                    
                    # Вывод прогресса
                    if disp:
                        total_px = img_h * img_w
                        current_px = i * img_w + j + 1
                        progress = f'{round(current_px / total_px * 100, 2)}%'
                        print(f'\rРисование: {progress:<7}', end='')
                        
                # Перенос строки в текстовом представлении картинки из смайликов
                if i + 1 < img_h:
                    emoji_image_text += '\n'
                    
            # Перенос строки из-за вывода прогресса с '\r'
            if disp:
                print()

    # Сохранение картинки в виде текста
    with open(f'output/{save_image_name}.txt', 'w') as file:
        file.write(emoji_image_text)
    
    # Сохранение изображения
    result_image.save(f'output/{save_image_name}.png')
    
    if disp:
        print('Изображение сохранено в папку output')
        
    return result_image