# each of the ML benchmark tests
import tensorflow as tf
import tensorflow_hub as hub
import timeit
import sys
import time

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

def display_output(envData, test_title, testtime):
    print('----------------------------------------------------------------')
    sys.stdout.write("Total running time for the " + test_title + ": %d:%d:%d.\n" % (testtime[0], testtime[1], testtime[2]))
    print('----------------------------------------------------------------')

    # write out the test data to outputFileName, append if already there
    with open(envData['outputFileName'],'a', newline='\n') as fd:
        fd.write(str(envData['runDate']) + ',' + str(envData['runTime']) + ',' + envData['machineType'] + ',' + str(test_title) + ',' + str(int(testtime[0])) + ':' + str(int(testtime[1])) + ':' + str(testtime[2]) + '\n')

def how_long(start, stop):
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs

def test1(epochs=5):
    # basic MNIST test
    mnist = tf.keras.datasets.mnist

    # regular data setup
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # very simple MNIST model, same as tuitorial
    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model1.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

    # perform a timed training session
    start = timeit.default_timer()
    model1.fit(x_train, y_train, epochs=epochs)
    stop = timeit.default_timer()

    # let's see how it did
    results_accuracy = model1.evaluate(x_test,  y_test, verbose=2)

    return how_long(start, stop)

def test2(epochs=10, activation_f='relu'):
    # start of fashion MNIST test -----------------------------------------------------------
    fashion_mnist = tf.keras.datasets.fashion_mnist

    model2 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(2048,   activation=activation_f),
        tf.keras.layers.Dense(1024,   activation=activation_f),
        tf.keras.layers.Dense(512,    activation=activation_f),
        tf.keras.layers.Dense(256,    activation=activation_f),
        tf.keras.layers.Dense(128,    activation=activation_f),
        tf.keras.layers.Dense(64,     activation=activation_f),
        tf.keras.layers.Dense(10)
    ])

    # fashion data setup
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    model2.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    start = timeit.default_timer()
    model2.fit(train_images, train_labels, epochs=epochs)
    stop = timeit.default_timer()

    test_loss, test_acc = model2.evaluate(test_images,  test_labels, verbose=0)

    return how_long(start, stop)

def test3(epochs=10, feature_extractor_model='https://tfhub.dev/tensorflow/resnet_50/feature_vector/1'):
    # Test training a normal off-the-shelf CNN
    # Good test networks are    https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4
    #                           https://tfhub.dev/tensorflow/resnet_50/feature_vector/1'
    batch_size = 32
    img_height = 224
    img_width = 224
    
    data_root = tf.keras.utils.get_file(
        'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_root),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    num_classes = len(class_names)

    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'])

    batch_stats_callback = CollectBatchStats()

    start = timeit.default_timer()
    history = model.fit(train_ds, epochs=epochs,
                    callbacks=[batch_stats_callback])
    stop = timeit.default_timer()

    return how_long(start, stop)

def test4(batch_size=32, classifier_model='https://tfhub.dev/tensorflow/resnet_50/classification/1'):
    # Generic CNN inference test
    # https://tfhub.dev/tensorflow/resnet_50/classification/1
    # https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4
    IMAGE_SHAPE = (224, 224)

    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
    ])

    # get some data
    data_root = tf.keras.utils.get_file(
        'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True)
    
    # this should be modified if the model choosen supports a different input size
    img_height = 224
    img_width = 224

    inference_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_root),
        validation_split=None,
        subset=None,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    inference_ds = inference_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    inference_ds = inference_ds.cache().prefetch(buffer_size=AUTOTUNE)

    start = timeit.default_timer()
    result_batch = classifier.predict(inference_ds)
    stop = timeit.default_timer()

    return how_long(start, stop)
