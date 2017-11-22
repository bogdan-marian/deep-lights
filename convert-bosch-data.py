import tensorflow as tf
import yaml
import os
import sys
import ntpath
import exceptions
import shutil
import progressbar
from object_detection.utils import dataset_util




#bosh_data = "bosh-data"
#data_folder = "dataset_train_rgb"
#yaml_file = "train.yaml"

#the fist folder from INPUT_YAML needs to exist in the curent folder
#the 'bosh-data' folder is ignored by gitingore
INPUT_YAML = "bosh-data/dataset_train_rgb/train.yaml"
#INPUT_YAML = "bosh-data/dataset_test_rgb/test.yaml"


# the "results" folder is ignored by gitingore
#results = "results/test.record"
results = "results/bos-train.record"


flags = tf.app.flags
flags.DEFINE_string('output_path', results, 'Path to output TFRecord')
FLAGS = flags.FLAGS

def check_file_names():

    #create results directory if absent
    result_dir = os.path.dirname(results)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #if results exist just stop so wee do not override previous work
    if os.path.exists(results):
        raise Exception(results, 'allready exists. Please delete and try agin!')


    #check the bos-data directory
    bosh_data = INPUT_YAML.split('/',1)[0]
    if not os.path.exists(bosh_data):
        raise Exception (bosh_data, "directory is missing please download and then continue")





LABEL_DICT =  {
    "Green" : 1,
    "Red" : 2,
    "GreenLeft" : 3,
    "GreenRight" : 4,
    "RedLeft" : 5,
    "RedRight" : 6,
    "Yellow" : 7,
    "off" : 8,
    "RedStraight" : 9,
    "GreenStraight" : 10,
    "GreenStraightLeft" : 11,
    "GreenStraightRight" : 12,
    "RedStraightLeft" : 13,
    "RedStraightRight" : 14
    }


#used only for training. the path inside yaml file are scrambled
#PATO_TO_IMAGES = "bosh-data/dataset_test_rgb/rgb/test/"

def create_tf_example(example):

    # Bosch
    height = 720 # Image height
    width = 1280 # Image width

    filename = None
    if 'test' in INPUT_YAML:
        test_path = os.path.dirname(INPUT_YAML)
        test_path += '/rgb/test/'
        #PATO_TO_IMAGES = "bosh-data/dataset_test_rgb/rgb/test/"


        temppath = example['path'] # Filename of the image. Empty if image is not from file
        basename = ntpath.basename(temppath)
        filename = test_path + basename

    if 'train' in INPUT_YAML:
        filename = example['path'] # Filename of the image. Empty if image is not from file

    if not os.path.exists(filename):
        raise Exception("file not found: " + filename)

    filename = filename.encode()

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_image = fid.read()

    image_format = 'png'.encode()

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in example['boxes']:
        #if box['occluded'] is False:
        #print("adding box")
        xmins.append(float(box['x_min'] / width))
        xmaxs.append(float(box['x_max'] / width))
        ymins.append(float(box['y_min'] / height))
        ymaxs.append(float(box['y_max'] / height))
        classes_text.append(box['label'].encode())
        classes.append(int(LABEL_DICT[box['label']]))


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    check_file_names()

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    examples = yaml.load(open(INPUT_YAML, 'rb').read())

    #examples = examples[:10]  # for testing
    len_examples = len(examples)

    print("Loaded ", len(examples), "examples")

    for i in range(len(examples)):
        #i = 8333 threre is a bug
        examples[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(INPUT_YAML), examples[i]['path']))

    print ("converting bosh data to google tensor record data")
    counter = 0
    bar = progressbar.ProgressBar(maxval=len_examples, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for example in examples:
        tf_example = create_tf_example(example)

        writer.write(tf_example.SerializeToString())

        counter += 1
        bar.update(counter)

    writer.close()
    bar.finish()



if __name__ == '__main__':
    tf.app.run()
