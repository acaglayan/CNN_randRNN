import numpy as np

class_id_to_name = {
    "0": "apple",
    "1": "ball",
    "2": "banana",
    "3": "bell_pepper",
    "4": "binder",
    "5": "bowl",
    "6": "calculator",
    "7": "camera",
    "8": "cap",
    "9": "cell_phone",
    "10": "cereal_box",
    "11": "coffee_mug",
    "12": "comb",
    "13": "dry_battery",
    "14": "flashlight",
    "15": "food_bag",
    "16": "food_box",
    "17": "food_can",
    "18": "food_cup",
    "19": "food_jar",
    "20": "garlic",
    "21": "glue_stick",
    "22": "greens",
    "23": "hand_towel",
    "24": "instant_noodles",
    "25": "keyboard",
    "26": "kleenex",
    "27": "lemon",
    "28": "lightbulb",
    "29": "lime",
    "30": "marker",
    "31": "mushroom",
    "32": "notebook",
    "33": "onion",
    "34": "orange",
    "35": "peach",
    "36": "pear",
    "37": "pitcher",
    "38": "plate",
    "39": "pliers",
    "40": "potato",
    "41": "rubber_eraser",
    "42": "scissors",
    "43": "shampoo",
    "44": "soda_can",
    "45": "sponge",
    "46": "stapler",
    "47": "tomato",
    "48": "toothbrush",
    "49": "toothpaste",
    "50": "water_bottle"
}
class_name_to_id = {v: k for k, v in class_id_to_name.items()}

class_names = set(class_id_to_name.values())


def get_class_names(ids):
    names = []
    for cls_id in ids:
        cls_name = class_id_to_name[str(cls_id)]
        names.append(cls_name)
    return np.asarray(names)
