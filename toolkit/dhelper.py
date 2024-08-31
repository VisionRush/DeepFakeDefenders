import os


def get_file_name_ext(filepath):
    # analyze
    file_name, file_ext = os.path.splitext(filepath)
    # return
    return file_name, file_ext


def get_file_ext(filepath):
    return get_file_name_ext(filepath)[1]


def traverse_recursively(fileroot, filepathes=[], extension='.*'):
    """Traverse all file path in specialed directory recursively.

    Args:
        h: crop height.
        extension: e.g. '.jpg .png .bmp .webp .tif .eps'
    """
    items = os.listdir(fileroot)
    for item in items:
        if os.path.isfile(os.path.join(fileroot, item)):
            filepath = os.path.join(fileroot, item)
            fileext = get_file_ext(filepath).lower()
            if extension == '.*':
                filepathes.append(filepath)
            elif fileext in extension:
                filepathes.append(filepath)
            else:
                pass                    
        elif os.path.isdir(os.path.join(fileroot, item)):
            traverse_recursively(os.path.join(fileroot, item), filepathes, extension)
        else:
            pass
