def divisible(dim):
    """
    Make width and height divisible by 32
    """
    width, height = dim
    return width - (width % 32), height - (height % 32)


def resize_image(image, width=None, height=None):
    dim = None
    w, h = image.size
    # print(w,h)

    if width and height:
        # return cv2.resize(image, divisible((width, height)), interpolation=inter)
        return image.resize(divisible((width, height)))

    if width is None and height is None:
        # return cv2.resize(image, divisible((w, h)), interpolation=inter)
        return image.resize(divisible((w, h)))

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # return cv2.resize(image, divisible(dim), interpolation=inter)
    return image.resize(divisible((dim)))
