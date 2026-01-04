class GeneralPath:
    LOG_PATH:str = r".logs/"

class DataPath:
    IMG_TRAIN:str = r"data/images/train/"
    IMG_VAL:str = r"data/images/val/"
    IMG_TEST:str = r"data/images/test/"
    LABEL_TRAIN:str = r"data/labels/train/"
    LABEL_VAL:str = r"data/labels/val/"
    LABEL_TEST:str = r"data/labels/test/"

class ResultPath:
    PREDICTION_PATH:str = r"data/results/predictions/"
    TEST_PATH:str = r"data/results/tests/"

class DataGeneralInfo:
    CLASS_NAME:dict = {
        0: "Field",
        1: "Building",
        2: "Woodland",
        3: "Water",
        4: "Road",
    }
    CLASS_COLOR:dict = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [0, 255, 0],
        3: [0, 0, 255],
        4: [128, 128, 128],
    }
