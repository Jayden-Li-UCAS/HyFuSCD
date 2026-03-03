import numpy as np

def color_map_SECOND():
    cmap = np.zeros((7, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([0, 0, 255])
    cmap[2] = np.array([128, 128, 128])
    cmap[3] = np.array([0, 128, 0])
    cmap[4] = np.array([0, 255, 0])
    cmap[5] = np.array([128, 0, 0])
    cmap[6] = np.array([255, 0, 0])
    return cmap

def color_map_Landsat():
    cmap = np.zeros((5, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([0, 155, 0])
    cmap[2] = np.array([255, 165, 0])
    cmap[3] = np.array([230, 30, 100])
    cmap[4] = np.array([0, 170, 240])
    return cmap

def color_map_OpenMapCD():
    cmap = np.zeros((8, 3), dtype=np.uint8)
    cmap[0] = np.array([0, 0, 0])
    cmap[1] = np.array([128, 0, 0])
    cmap[2] = np.array([70, 181, 121])
    cmap[3] = np.array([148, 148, 148])
    cmap[4] = np.array([222, 184, 70])
    cmap[5] = np.array([28, 140, 189])
    cmap[6] = np.array([167, 187, 27])
    cmap[7] = np.array([181, 70, 70])
    return cmap

def color_map_SCSCD7():
    cmap = np.zeros((8, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([237, 125, 49])
    cmap[2] = np.array([0, 0, 255])
    cmap[3] = np.array([255, 0, 0])
    cmap[4] = np.array([255, 255, 0])
    cmap[5] = np.array([0, 255, 0])
    cmap[6] = np.array([0, 128, 0])
    cmap[7] = np.array([128, 128, 128])
    return cmap

COLOR_MAP_MAPPING = {
    "SECOND": color_map_SECOND,
    "Landsat": color_map_Landsat,
    "SCSCD7": color_map_SCSCD7,
    "OpenMapCD": color_map_OpenMapCD
}

def get_color_map(data_name):
    if data_name not in COLOR_MAP_MAPPING:
        raise ValueError(f"Invalid data_name: {data_name}, available options: {list(COLOR_MAP_MAPPING.keys())}")
    return COLOR_MAP_MAPPING[data_name]()
