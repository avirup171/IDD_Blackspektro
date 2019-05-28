"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    
    if row_label == 'parking':
        return 1
    if row_label == 'drivable fallback':
        return 2
    if row_label == 'sidewalk':
        return 3
    if row_label == 'rail track':
        return 4
    if row_label == 'non-drivable fallback':
        return 5
    if row_label == 'person':
        return 6
    if row_label == 'animal':
        return 7
    if row_label == 'rider':
        return 8
    if row_label == 'motorcycle':
        return 9
    if row_label == 'bivyvle':
        return 10
    if row_label == 'autorickshaw':
        return 11
    if row_label == 'car':
        return 12
    if row_label == 'truck':
        return 13
    if row_label == 'bus':
        return 14
    if row_label == 'caravan':
        return 15
    if row_label == 'trailer':
        return 16
    if row_label == 'train':
        return 17
    if row_label == 'vehicle fallback':
        return 18
    if row_label == 'curb':
        return 19
    if row_label == 'wall':
        return 20
    if row_label == 'fence':
        return 21
    if row_label == 'guard rail':
        return 22
    if row_label == 'billboard':
        return 23
    if row_label == 'traffic sign':
        return 24
    if row_label == 'traffic light':
        return 25
    if row_label == 'pole':
        return 26
    if row_label == 'polegroup':
        return 27
    if row_label == 'obs-str-bar-fallback':
        return 28
    if row_label == 'building':
        return 29
    if row_label == 'bridge':
        return 30
    if row_label == 'tunnel':
        return 31
    if row_label == 'vegitation':
        return 32
    if row_label == 'sky':
        return 33
    else:
        return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    print(group)
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        print(group.filename)
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
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
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filepath')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
