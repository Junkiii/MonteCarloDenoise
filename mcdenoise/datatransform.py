# Hendrik Junkawitsch; Saarland University

# This is the data transform module that transforms 
# the raw training and validation data to a usable 
# training and test data set

import os
from shutil import copyfile

def transform_data(from_path, to_path):
    print("Transforming the raw training/validation data")
    root = os.getcwd()
    path_raw = os.path.join(root, from_path)
    print(">>>  training data: ", path_raw)
    path_new = os.path.join(root, to_path)
    print(">>>  Making new directory for transformed data:",to_path)
    try:
        os.mkdir(path_new)
    except FileExistsError:
        print(">>>  Directory already exists. Using the existing directory")
    except OSError:
        print(">>>  Creation of directory failed!")
        return
    else:
        print(">>>  Succesfully created new directory!")

    samples = ["4", "8", "12", "16", "20", "24", "28", "32", "36", "60", 
               "120", "180", "240", "300", "360", "400", "800", "1200", 
               "1600", "2000", "2400", "2800", "3200", "3600", "4000", 
               "8000", "12000", "16000", "20000", "24000", "28000", "32000"]

    data_index = 0
    for dir in os.listdir(path_raw):
        subdir_path = os.path.join(path_raw, dir)
        for subdir in os.listdir(subdir_path):
            sample_path = os.path.join(subdir_path, subdir)
            for s in samples:
                src = os.path.join(sample_path, s+"spp.png")
                dst = os.path.join(path_new, str(data_index)+"_noisy.png")
                copyfile(src,dst)
                src = os.path.join(sample_path, "albedo.png")
                dst = os.path.join(path_new, str(data_index)+"_albedo.png")
                copyfile(src,dst)
                src = os.path.join(sample_path, "normal.png")
                dst = os.path.join(path_new, str(data_index)+"_normal.png")
                copyfile(src,dst)
                src = os.path.join(sample_path, "gt.png")
                dst = os.path.join(path_new, str(data_index)+"_gt.png")
                copyfile(src,dst)
                src = os.path.join(sample_path, "diffuse.png")
                dst = os.path.join(path_new, str(data_index)+"_diffuse.png")
                copyfile(src, dst)
                src = os.path.join(sample_path, "specular.png")
                dst = os.path.join(path_new, str(data_index)+"_specular.png")
                copyfile(src, dst)
                data_index += 1

    print(">>>  FINISHED!")
    print(">>>  Created ", data_index, " data points with ", data_index*6, " images in total")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__ == "__main__":
    from_path = "validation_data/data_raw"
    to_path = "validation_data/data"
    transform_data(from_path, to_path)
