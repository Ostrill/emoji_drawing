from PIL import Image, ImageFont, ImageDraw
from pilmoji import Pilmoji
from pilmoji.source import EmojiCDNSource
import numpy as np
import pickle


# Поддерживаемые стили для смайликов
STYLES = ['twitter', 'apple', 'google', 'facebook']


def get_emoji_style(style):
    """
    Возвращает класс стиля смайликов. Стилей всего 4:
    - twitter
    - apple
    - google
    - facebook
    """
    # Проверка, поддерживается ли стиль
    assert style in STYLES, f'Поддерживаемые стили: {", ".join(STYLES)}'

    # Класс, который используется для стилей в Pilmoji
    class StyleClass(EmojiCDNSource):
        STYLE = style
        
    return StyleClass


def create_emoji_dict(emojis, 
                      background_color=None,
                      emoji_resolution=100, 
                      emoji_styles=STYLES,
                      disp=True):
    """
    Создать словарь смайликов, где каждому смайлику
    соответствует подходящий ему цвет и доля занимаемой
    им площади в квадратной ячейке. Для каждого стиля
    создается собственный словарь, итоговая структура
    получается следующей:
    emoji_dict = {
        '<стиль>' : {
            '<смайлик>' : (
                <средний цвет> : array([float, float, float]),
                <занимаемая площадь> : float
            )
        }
    }

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

    emoji_styles : list[str]
        | список из стилей смайликов, для которых нужно
          построить словари
    
    disp : bool
        | выводить ли прогресс создания словаря
          (шкала в процентах)
    """
    emoji_dict = {}

    # Формирование списка стилей
    styles = [get_emoji_style(style) for style in emoji_styles]

    # Формирование RGBA цвета фона с A=0
    img_back = (*(background_color or [0]*3), 0)
    # Размер картинки, в которой будут рисоваться смайлики
    img_size = (emoji_resolution, emoji_resolution)

    # Цикл по всем рассматриваемым стилям
    for style, style_name in zip(styles, emoji_styles):
        # Инициализация словаря для стиля
        emoji_dict[style_name] = {}
        # Цикл по всем смайликам
        for i, emoji in enumerate(emojis, 1):
            # Создание новой картинки со смайликом
            with Image.new('RGBA', img_size, img_back) as image:
                # Шрифт для отображения смайлика
                font = ImageFont.truetype('fonts/arial.ttf', emoji_resolution)
                # Рисование смайлика на картинке
                with Pilmoji(image, source=style) as pilmoji:
                    pilmoji.text((0, 0), emoji, font=font)
    
            # Веса каждого пикселя (насколько каждый непрозрачен)
            np_image = np.array(image)
            weights = np_image[:, :, 3:] / 255
            # Доля занимаемой смайликом площади в квадратной ячейке
            area = weights.sum() / (emoji_resolution ** 2)
            
            # Вычисление среднего цвета получившейся картинки смайлика
            if background_color is None:
                # Средний цвет только по непрозрачным пикселям
                weighted_sum = (np_image[:, :, :3] * weights).sum(axis=(0, 1))
                avg_color =  weighted_sum / weights.sum()
            else:
                # Средний цвет по каналам RGB (без учета прозрачности)
                avg_color = np.array(image)[:, :, :3].mean(axis=(0, 1))
    
            # Добавление смайлика в словарь
            emoji_dict[style_name][emoji] = (avg_color, area)
    
            # Вывод прогресса
            if disp:
                progress = f'{round(i / len(emojis) * 100, 2)}%'
                progress_text = f'Построение словаря для стиля "{style_name}":'
                print(f'\r{progress_text} {progress:<7} ({i}/{len(emojis)})', end='')
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


def unite_emoji_dict(emoji_dict, styles=None):
    """
    Объединить смайлики из разных стилей в один словарь,
    добавив к ключам название стиля.

    PARAMETERS
    ----------
    emoji_dict : dict
        | словарь смайликов с разными стилями
    
    styles : list[str] or None
        | список стилей, которые необходимо объединить
          (если None, объединятся все имеющиеся стили)
    """
    united = {}
    if styles is None:
        styles = emoji_dict.keys()
    for style in styles:
        for k, v in emoji_dict[style].items():
            united[f'{style}_{k}'] = v
    return united


def save_matrix_as_text(emoji_matrix, name):
    """
    Сохранить результат (матрицу смайликов) в файл
    в виде текста

    PARAMETERS
    ----------
    emoji_matrix : ndarray[str]
        | 2D numpy-массив из смайликов
    
    name : str
        | название картинки, текстовый файл для 
          которой будет сохранен в папку output/
    """
    with open(f'output/{name}.txt', 'w') as file:
        as_string = '\n'.join(''.join(line) 
                              for line in emoji_matrix)
        file.write(as_string)


def draw(image_filename,
         save_image_name='image',
         emoji_width=50,
         emoji_resolution=50,
         background_color=(0, 0, 0),
         emojis=None, 
         emoji_dict_name='classic_black',
         emoji_style='google',
         dynamic_background=False,
         save_as_text=True,
         disp=True):
    """
    Главная функция для рисования картинки из символов

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
    
    emoji_style : str или list[str]
        | стиль отображения смайликов, всего поддерживается четыре
          разных стиля: 'twitter', 'apple', 'google' и 'facebook'.
          Можно указать как какой-либо конкретный стиль, так и 
          список из стилей

    dynamic_background : bool
        | использовать ли динамическое создание фона, при
          котором для каждого смайлика отрисовывается собственный 
          цвет фона, в сочетании с которым он становится еще ближе
          к исходной картинке

    save_as_text : bool
        | сохранять ли картинку в текстовом формате (в виде набора
          смайликов)
    
    disp : bool
        | выводить ли прогресс создания картинки
    """
    # Проверка, задан ли набор смайликов, которыми нужно рисовать
    error_text = 'Необходимо указать либо emojis, либо emojis_dict!'
    assert not(emojis is None and emoji_dict_name is None), error_text

    # Формирование стиля для смайликов
    if isinstance(emoji_style, str):
        emoji_style = [emoji_style]
        
    # Это делается по той причине, что внутри get_emoji_style есть assert,
    # завершающий функцию с ошибкой в случае некорректных стилей
    _ = [get_emoji_style(s) for s in emoji_style]
    
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
                                       emoji_styles=emoji_style,
                                       disp=disp)
    # Объединение словарей со стилями
    emoji_dict = unite_emoji_dict(emoji_dict, emoji_style)
    
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

    # Установка цвета для итоговой картинки
    if background_color is None:
        background_RGBA = (0, 0, 0, 0)
    else:
        background_RGBA = (*background_color, 255)

    # Матрицы со стилями и смайликами по каждой ячейке картинки
    style_matrix = np.full((img_h, img_w), '', dtype='object')
    emoji_matrix = np.full((img_h, img_w), '', dtype='object')
        
    # Создание итогового изображения result_image
    with Image.new('RGBA', (result_w, result_h), background_RGBA) as result_image:
        # Компонент для рисования на картинке
        img_draw = ImageDraw.Draw(result_image)
        # Шрифт для отображения смайликов
        font = ImageFont.truetype('fonts/arial.ttf', emoji_resolution)

        # Построение матриц смайликов и соответствующих им стилей
        for i in range(img_h):
            for j in range(img_w):
                if dynamic_background:
                    # Индексы пикселей в итоговой картинке
                    h, w = i * emoji_resolution, j * emoji_resolution
                    # Поиск подходящего сочетания смайлика с фоном
                    emoji, back = get_nearest_emoji_and_back(image[i, j], emoji_dict)
                    # Отрисовка прямоугольника (фона для смайлика)
                    img_draw.rectangle((w, h, w+emoji_resolution, h+emoji_resolution), 
                                       fill=tuple(np.rint(back).astype(int)))
                else:
                    # Поиск ближайшего по среднему цвету смайлика
                    emoji = get_nearest_emoji(image[i, j], emoji_dict)

                style_matrix[i, j], emoji_matrix[i, j] = emoji.split('_')

                # Вывод прогресса
                if disp:
                    total_px = img_h * img_w
                    current_px = i * img_w + j + 1
                    progress = f'{round(current_px / total_px * 100, 2)}%'
                    print(f'\rФормирование матрицы: {progress:<7}', end='')
        if disp:
            print()

        # Отрисовка смайликов для каждого стиля по отдельности
        # (потому что если это делать в одном цикле, это будет намного дольше)
        for style_index, style_name in enumerate(emoji_style):
            with Pilmoji(result_image, source=get_emoji_style(style_name)) as pilmoji:
                for i in range(img_h):
                    for j in range(img_w):
                        if style_matrix[i, j] == style_name:
                            # Индексы пикселей в итоговой картинке
                            h, w = i * emoji_resolution, j * emoji_resolution
                            pilmoji.text((w, h), emoji_matrix[i, j], font=font)

                        # Вывод прогресса
                        if disp:
                            total_iters = img_h * img_w * len(emoji_style)
                            current_iter = style_index * img_h * img_w + i * img_w + j + 1
                            progress = f'{round(current_iter / total_iters * 100, 2)}%'
                            print(f'\rРисование: {progress:<7}', end='')
        if disp:
            print()

    # Сохранение картинки в виде текста
    if save_as_text:
        save_matrix_as_text(emoji_matrix, save_image_name)
    
    # Сохранение изображения
    result_image.save(f'output/{save_image_name}.png')
    
    if disp:
        print('Изображение сохранено в папку output')
        
    return result_image