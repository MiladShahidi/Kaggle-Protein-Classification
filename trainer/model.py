import tensorflow as tf
from tensorflow.contrib.metrics import f1_score
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.estimator import model_to_estimator
from functools import partial

def build_model(input_shape, n_classes):
    instance_keys_in = tf.keras.layers.Input(shape=[None], dtype=tf.string, name='key_in')

    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Adding the final fully connected layer
    x = Flatten()(base_model.output)
    predictions = Dense(n_classes, activation='sigmoid', name='scores')(x)
    instance_keys_out = tf.keras.layers.Lambda(lambda x: x, name='key')(instance_keys_in)

    inputs = [base_model.input, instance_keys_in]
    outputs = [predictions, instance_keys_out]

    model = Model(inputs=inputs, outputs=outputs)

    return model
    

def input_fn(mode, data_files, batch_size, input_names,
              input_shape, n_classes = 28):
    def _parse_example(example_proto, is_training):
        image_feature_description = {
            'ID': tf.FixedLenFeature([], tf.string),
            'png_bytes': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        }

        # Parse the input tf.Example proto using the dictionary above.
        parsed_features = tf.parse_single_example(example_proto, image_feature_description)
        image = tf.image.decode_png(parsed_features['png_bytes'], channels=3)
                
        # decode_png does not return a shape. So we need to use set_shape here
        image.set_shape(input_shape)

        image = tf.cast(image, tf.float32) / 255. # maps t0 (0, 1)

        # Data augmentation during training
        if is_training:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            # rotating by 90 degrees for a random number of times (between 0 and 3)
            num_rot90 = tf.random_uniform([], 0, 4, dtype=tf.int32)
            image = tf.image.rot90(image, k=num_rot90)

        label = tf.decode_raw(parsed_features['label'], out_type = tf.int32)
        label = tf.cast(label, tf.float32)
        
        features = {input_names[0]: image, input_names[1]: parsed_features['ID']}
        labels = {'scores': label, 'key': 'dummy_string'}
        
        return features, labels

    def train_input_fn():
        train_files = tf.data.Dataset.list_files(data_files).shuffle(100)
        train_dataset = (
                tf.data.TFRecordDataset(train_files)
                .map(partial(_parse_example, is_training=True))
                .shuffle(1000)
                .repeat(None)  # supply data indefinitely
                .batch(batch_size)
                .prefetch(1)  # prefetch one batch for better performance
                )
        features, labels = train_dataset.make_one_shot_iterator().get_next()
        return features, labels

    # in test mode (using the predict method), the predict method throughs out
    # the (fake) labels (the second element of the tuple returned.
    # But the test TFRecord files do contain (fake) labels
    # to keep the format consistent with train and validate TFRecords
    # so that I can use the same function for evaluation and testing
    def predict_input_fn():
        validation_files = tf.data.Dataset.list_files(data_files)
        val_dataset = (
                tf.data.TFRecordDataset(validation_files)
                .map(partial(_parse_example, is_training=False))
                .batch(batch_size)
                )
        features, labels = val_dataset.make_one_shot_iterator().get_next()
        return features, labels 
        
    if mode == tf.estimator.ModeKeys.TRAIN:
        input_fn = train_input_fn
    elif mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        input_fn = predict_input_fn

    return input_fn


def f1(labels, predictions):
    n_classes = 28
    pred_vals = predictions['scores']
    labels = labels['scores']
    res = {}
    f1_list = []
    op_list = []
    for i in range(n_classes):
        f1, op = f1_score(labels[:,i], pred_vals[:,i])
        res['f1_%d'%i] = (f1, op)
        f1_list.append(f1)
        op_list.append(op)

    res['f1_macro'] = (tf.reduce_mean(tf.stack(f1_list)), tf.group(*op_list))
    return res


def train_and_eval(args):
    n_classes = 28
    input_shape = (512, 512, 3) # Input image dimensions (and 3 color channels)
    
    keras_model = build_model(input_shape=input_shape, n_classes=n_classes)

    global_step = tf.Variable(0, trainable=False, name='global_step')

    boundaries = [10000, 20000, 27500, 44000]
    values = [5E-4, 2.5E-4, 5E-5, 2E-5, 1E-5]
    piecewise_lr = tf.train.piecewise_constant(global_step, boundaries, values)
    adam_optimizer = tf.keras.optimizers.Adam(lr=piecewise_lr)
    losses = {
            'scores': 'binary_crossentropy',
            'key': lambda y_true, y_pred: tf.constant(0, dtype=tf.float32)
            }
    
    # weights chosen to make the average equal to the loss associated with scores
    loss_weights = {'scores': 2.0, 'key': 0.0}

    keras_model.compile(adam_optimizer, loss=losses, loss_weights=loss_weights)

    # This will be used in the input functions
    feature_names = keras_model.input_names

    run_config = tf.estimator.RunConfig(model_dir=args['model_dir'])
    model = model_to_estimator(keras_model=keras_model, config=run_config)
    model = tf.contrib.estimator.add_metrics(model, f1)

    # training and evaluation
    train_input_fn = input_fn(tf.estimator.ModeKeys.TRAIN, args['train_data_files'],
                              args['batch_size'], feature_names, input_shape)
    eval_input_fn = input_fn(tf.estimator.ModeKeys.EVAL, args['eval_data_files'],
                              args['batch_size'], feature_names, input_shape)

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None) # Exhaust eval data
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=None) # train indefinitely
    
    # Launches training
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)