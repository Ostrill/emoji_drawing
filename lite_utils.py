from PIL import Image, ImageFont
from pilmoji import Pilmoji
import numpy as np
import pickle
import matplotlib.pyplot as plt


def create_emoji_dict(emojis, 
                      background_color=(0, 0, 0),
                      emoji_resolution=100, 
                      disp=True):
    """
    Создать словарь смайликов, где каждому смайлику
    соответствует подходящий ему цвет

    PARAMETERS
    ----------
    emojis : list[str]
        | список из смайликов, из которых создается 
          словарь
    
    background_color : (int, int, int)
        | фоновый цвет для смайликов в формате RGB
    
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

    # Цикл по всем смайликам
    for i, emoji in enumerate(emojis, 1):
        # Создание новой картинки со смайликом
        with Image.new('RGB', (emoji_resolution, emoji_resolution), 
                       background_color) as image:
            # Шрифт для отображения смайлика
            font = ImageFont.truetype('fonts/arial.ttf', emoji_resolution)
            # Рисование смайлика на картинке
            with Pilmoji(image) as pilmoji:
                pilmoji.text((0, 0), emoji, font=font)

        # Вычисление среднего цвета получившейся картинки смайлика
        avg_color = np.array(image).mean(axis=(0, 1))
        emoji_dict[emoji] = avg_color

        # Вывод прогресса
        if disp:
            progress = round(i / len(emojis) * 100, 2)
            print(f'\rПостроение словаря: {progress}%', end='')

    if disp:
        print()

    return emoji_dict


def load_emoji_dict(name):
    """
    Загрузить словарь смайликов из папки emoji_dicts/
    по его названию
    """
    with open(f'emoji_dicts/{name}.pkl', 'rb') as file:
        emoji_dict = pickle.load(file)
    return emoji_dict


def color_diff(rgb1, rgb2):
    """Расстояние между двумя цветами"""
    return np.linalg.norm(rgb1 - rgb2)


def nearest_emoji(rgb, emoji_dict):
    """Найти ближайший по цвету смайлик"""
    return min(emoji_dict.keys(), 
               key=lambda k: color_diff(rgb, emoji_dict[k]))


def draw_emojis(filename, width, emoji_dict, disp=True):
    """
    Нарисовать картинку из смайликов-квадратов

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
    
    disp : bool
        | отображать ли результат и логи
    
    RETURNS
    -------
    emoji_array : ndarray[str]
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
    
    # Массив из смайликов
    emoji_array = np.full((img_h, img_w), '', dtype='<U2')

    for i in range(img_h):
        for j in range(img_w):
            # Поиск подходящего смайлика и добавление его в массив
            emoji = nearest_emoji(image[i, j], emoji_dict)
            emoji_array[i, j] = emoji

            # Вывод прогресса
            if disp:
                total_px = img_h * img_w
                current_px = i * img_w + j + 1
                progress = round(current_px / total_px * 100, 2)
                print(f'\rРисование: {progress}%', end='')
    if disp:
        print()

    return emoji_array


def save_as_text(emoji_array, image_name):
    """
    Сохранить результат (массив смайликов) в файл
    в виде текста

    PARAMETERS
    ----------
    emoji_array : ndarray[str]
        | 2D numpy-массив из смайликов
    
    image_name : str
        | название картинки, текстовый файл для 
          которой будет сохранен в папку output/
    """
    with open(f'output/{image_name}.txt', 'w') as file:
        as_string = '\n'.join(''.join(line) 
                              for line in emoji_array)
        file.write(as_string)


def save_as_image(emoji_array, image_name, emoji_resolution, 
                  background_color=(0, 0, 0), disp=False):
    """
    Сохранить результат (массив смайликов) в файл
    в виде картинки формата PNG

    PARAMETERS
    ----------
    emoji_array : ndarray[str]
        | 2D numpy-массив из смайликов
    
    image_name : str
        | название картинки, png-файл для 
          которой будет сохранен в папку output/

    emoji_resolution : int
        | размер каждого смайлика на итоговой картинке
          в пикселях (каждый смайлик рисуется в квадратной 
          области, и этот параметр задает сторону квадрата)

    background_color : (int, int, int)
        | цвет фона картинки в формате RGB

    disp : bool
        | выводить ли прогресс сохранения картинки
    """
    # Высота и ширина массива из смайликов
    emojis_h, emojis_w = emoji_array.shape
    # Высота и ширина итоговой картинки из смайликов
    img_h, img_w = np.array(emoji_array.shape) * emoji_resolution
    
    # Создание картинки из смайликов
    with Image.new('RGB', (img_w, img_h), background_color) as image:
        # Шрифт для отображения смайликов
        font = ImageFont.truetype('fonts/arial.ttf', emoji_resolution)
        # Рисование смайликов на картинке
        with Pilmoji(image) as pilmoji:
            for i in range(emojis_w):
                for j in range(emojis_h):
                    # Индексы пикселей в итоговой картинке
                    w, h = i * emoji_resolution, j * emoji_resolution
                    pilmoji.text((w, h), emoji_array[j, i], font=font)
                    
                    # Вывод прогресса
                    if disp:
                        total_px = emojis_h * emojis_w
                        current_px = i * emojis_h + j + 1
                        progress = round(current_px / total_px * 100, 2)
                        print(f'\rСохранение изображения: {progress}%', end='')
            if disp:
                print()

    image.save(f'output/{image_name}.png')
    return image


def show_image(image, background_color=(0, 0, 0)):
    """Показать изображение с помощью matplotlib"""
    fig, ax = plt.subplots(dpi=300)
    fig.set_facecolor(np.array(background_color) / 255)
    ax.imshow(np.array(image))
    ax.axis('off')
    plt.tight_layout()
    plt.show()


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
        
    # Рисование картинки (построение массива из смайликов)
    emoji_array = draw_emojis(filename=image_filename, 
                              width=emoji_width, 
                              emoji_dict=emoji_dict, 
                              disp=disp)

    # Сохранение в виде текста
    save_as_text(emoji_array=emoji_array, 
                 image_name=save_image_name)
    if disp:
        print('Изображение в виде текста сохранено')

    # Сохранение в виде изображения
    result_image = save_as_image(emoji_array=emoji_array, 
                                 image_name=save_image_name, 
                                 emoji_resolution=emoji_resolution, 
                                 background_color=background_color, 
                                 disp=disp)
    if disp:
        print('Изображение в png-формате сохранено')

    # Рисование картинки в ячейке
    if disp:
        show_image(image=result_image, 
                   background_color=background_color)
        
    return result_image
    