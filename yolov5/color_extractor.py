from PIL import Image
import extcolors


def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax - cmin

    if cmax == cmin:
        h = 0

    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360

    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360

    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100

    v = cmax * 100
    return h, s, v


def get_hsv_color(img):
    output_width = 900
    wpercent = (output_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((output_width, hsize), Image.Resampling.LANCZOS)
    colors_x = extcolors.extract_from_image(img, tolerance=12, limit=12)
    colors, _ = colors_x
    dominant_color = colors[0][0]
    h, s, v = rgb_to_hsv(dominant_color[0], dominant_color[1], dominant_color[2])
    index = 0
    while is_sky(h, s, v):
        # try:
        # print('is_sky : ', is_sky(h, s, v))
        dominant_color = colors[index][0]
        h, s, v = rgb_to_hsv(dominant_color[0], dominant_color[1], dominant_color[2])
        index+=1
        # print('dominant color is => ', dominant_color, (h, s, v))

        # except:
        #     pass

    print('dominant color is => ', dominant_color, (h, s, v))
    return h, s, v


def map_color(h, s, v):
    green_boundaries = [i for i in range(80, 150)]
    yellow_boundaries = [i for i in range(50, 70)]
    orange_boundaries = [i for i in range(15, 50)]
    red_boundaries = [i for i in range(0, 15)]
    red_boundaries.extend([i for i in range(350, 360)])
    purple_boundaries = [i for i in range(270, 295)]
    color_range = {
        "green": green_boundaries,
        "yellow": yellow_boundaries,
        "orange": orange_boundaries,
        "red": red_boundaries,
        "purple": purple_boundaries
    }
    result = ""
    if s > 20 and v > 10:
        for key, val in color_range.items():
            if int(h) in val:
                # print(key)
                result = key
    return result


def is_sky(h, s, v):
    sky_range = [i for i in range(160, 240)]
    # s < 40 and v > 70
    if s < 40 and v > 50:
        if int(h) in sky_range:
            return True
    return False
