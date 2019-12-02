import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def convertFromBGRToGray(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray

def loadTemplates(names, dirPath):
    templates = {}
    for name in names:
        template_image = cv2.imread(dirPath + '{}.png'.format(name))
        templates[name] = convertFromBGRToGray(template_image)
    
    return templates

def templateMatching(image, template):
    end_row = image.shape[0] - template.shape[0] + 1
    end_col = image.shape[1] - template.shape[1] + 1
    
    temp_rows = template.shape[0]
    temp_cols = template.shape[1]
    
    max_val = float("-inf")
    
    for i in range(end_row):
        for j in range(end_col):
            I = image[i:i + temp_rows, j: j + temp_cols]
#             score = np.sum(np.multiply(template, I))
            score = np.dot(template.flatten().T, I.flatten().T)
            i_bar = math.sqrt(np.sum(np.square(I)))
            t_bar = math.sqrt(np.sum(np.square(template)))
            normalised_factor = i_bar * t_bar
#             normalised_factor = template.sum() * I.sum()
            norm_score = score / normalised_factor
            if norm_score > max_val:
                max_val = norm_score
            
    return max_val


def get_candidate_name(img):
    initials_to_names = {
        'as': 'Andrew Scheer',
        'em': 'Elizabeth May',
        'js': 'Jagmeet Singh',
        'jt': 'Justin Trudeau',
        'mb': 'Maxim Bernier',
        'yb': 'Yves-François Blanchet'
    }
    
    # img = cv2.imread('js_name.png')
    image = img.copy()
    # image = convertFromBGRToGray(image)
    
    template_names = ['as', 'em', 'js', 'jt', 'mb', 'yb']
    
    templates = loadTemplates(template_names, 'templates/')
    max_score = 0
    max_contendor = None
    
    for key in templates:
        template = templates[key]

        print(image)
        print("")
        print(template)
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        # max_val = templateMatching(image, template)
    
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > max_score:
            max_score = max_val
            max_contendor = key
    
    print(max_score, initials_to_names[max_contendor])