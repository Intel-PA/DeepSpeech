from __future__ import absolute_import, division, print_function

import os
import sys

LOG_LEVEL_INDEX = sys.argv.index("--log_level") + 1 if "--log_level" in sys.argv else 0
DESIRED_LOG_LEVEL = (
    sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else "3"
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = DESIRED_LOG_LEVEL

import optuna
import absl.app
import numpy as np
import progressbar
import shutil
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import time
import json
import sys

from multiprocessing import cpu_count

tfv1.logging.set_verbosity(
    {
        "0": tfv1.logging.DEBUG,
        "1": tfv1.logging.INFO,
        "2": tfv1.logging.WARN,
        "3": tfv1.logging.ERROR,
    }.get(DESIRED_LOG_LEVEL)
)

from datetime import datetime
from ds_ctcdecoder import ctc_beam_search_decoder, ctc_beam_search_decoder_batch, Scorer
from deepspeech_training.evaluate import evaluate
from six.moves import zip, range
from deepspeech_training.util.config import Config, initialize_globals
from deepspeech_training.util.checkpoints import (
    load_or_init_graph_for_training,
    load_graph_for_evaluation,
    reload_best_checkpoint,
)
from deepspeech_training.util.evaluate_tools import save_samples_json
from deepspeech_training.util.feeding import create_dataset, audio_to_features, audiofile_to_features
from deepspeech_training.util.flags import create_flags, FLAGS
from deepspeech_training.util.helpers import check_ctcdecoder_version, ExceptionBox
from deepspeech_training.util.logging import (
    create_progressbar,
    log_debug,
    log_error,
    log_info,
    log_progress,
    log_warn,
)
from deepspeech_training.util.io import (
    open_remote,
    remove_remote,
    listdir_remote,
    is_remote_path,
    isdir_remote,
)

from deepspeech_training.train import *
from tensorflow.compat.v1.keras import backend as K

check_ctcdecoder_version()

RNN_IMPL = rnn_impl_lstmblockfusedcell
BATCH_SIZE = 32
SEQ_LEN = None
DROPUT = 0.3
CHKPT_DIR = "checkpoints/optuna_trials"
MODEL_DIR = "model/optuna_trials"




def hps_create_optimizer(trial):
    learning_rate = trial.suggest_float("adam_lr", 1e-5, 1e-1, log=True)
    with tf.variable_scope("learning_rate", reuse=tf.AUTO_REUSE):
        learning_rate_var = tfv1.get_variable(
            "learning_rate", initializer=learning_rate, trainable=False
        )
    optimizer = tfv1.train.AdamOptimizer(
        learning_rate=learning_rate_var, beta1=0.9, beta2=0.999, epsilon=1e-08
    )
    return optimizer, learning_rate_var


def hps_evaluate(test_csvs, create_model):
    if FLAGS.scorer_path:
        scorer = Scorer(
            FLAGS.lm_alpha, FLAGS.lm_beta, FLAGS.scorer_path, Config.alphabet
        )
    else:
        scorer = None

    test_sets = [
        create_dataset(
            [csv],
            batch_size=FLAGS.test_batch_size,
            train_phase=False,
            reverse=FLAGS.reverse_test,
            limit=FLAGS.limit_test,
        )
        for csv in test_csvs
    ]
    iterator = tfv1.data.Iterator.from_structure(
        tfv1.data.get_output_types(test_sets[0]),
        tfv1.data.get_output_shapes(test_sets[0]),
        output_classes=tfv1.data.get_output_classes(test_sets[0]),
    )
    test_init_ops = [iterator.make_initializer(test_set) for test_set in test_sets]

    batch_wav_filename, (batch_x, batch_x_len), batch_y = iterator.get_next()

    # One rate per layer
    no_dropout = [None] * 6
    logits, _ = create_model(
        batch_x=batch_x, seq_length=batch_x_len, dropout=no_dropout
    )

    # Transpose to batch major and apply softmax for decoder
    transposed = tf.nn.softmax(tf.transpose(a=logits, perm=[1, 0, 2]))

    loss = tfv1.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_x_len)

    tfv1.train.get_or_create_global_step()

    # Get number of accessible CPU cores for this process
    try:
        num_processes = cpu_count()
    except NotImplementedError:
        num_processes = 1

    with tfv1.Session(config=Config.session_config) as session:
        load_graph_for_evaluation(session)

        def run_test(init_op, dataset):
            wav_filenames = []
            losses = []
            predictions = []
            ground_truths = []

            bar = create_progressbar(
                prefix="Test epoch | ",
                widgets=["Steps: ", progressbar.Counter(), " | ", progressbar.Timer()],
            ).start()
            log_progress("Test epoch...")

            step_count = 0

            # Initialize iterator to the appropriate dataset
            session.run(init_op)

            # First pass, compute losses and transposed logits for decoding
            while True:
                try:
                    batch_wav_filenames, batch_logits, batch_loss, batch_lengths, batch_transcripts = session.run(
                        [batch_wav_filename, transposed, loss, batch_x_len, batch_y]
                    )
                except tf.errors.OutOfRangeError:
                    break

                decoded = ctc_beam_search_decoder_batch(
                    batch_logits,
                    batch_lengths,
                    Config.alphabet,
                    FLAGS.beam_width,
                    num_processes=num_processes,
                    scorer=scorer,
                    cutoff_prob=FLAGS.cutoff_prob,
                    cutoff_top_n=FLAGS.cutoff_top_n,
                )
                predictions.extend(d[0][1] for d in decoded)
                ground_truths.extend(
                    sparse_tensor_value_to_texts(batch_transcripts, Config.alphabet)
                )
                wav_filenames.extend(
                    wav_filename.decode("UTF-8") for wav_filename in batch_wav_filenames
                )
                losses.extend(batch_loss)

                step_count += 1
                bar.update(step_count)

            bar.finish()

            # Print test summary
            test_samples = calculate_and_print_report(
                wav_filenames, ground_truths, predictions, losses, dataset
            )
            return test_samples, losses[-1]

        samples = []
        for csv, init_op in zip(test_csvs, test_init_ops):
            print("Testing model on {}".format(csv))
            s, loss = run_test(init_op, dataset=csv)
            samples.extend(s)
        return samples, loss


def hps_train(trial, session):
    exception_box = ExceptionBox()
    final_dev_loss = None
    if FLAGS.horovod:
        import horovod.tensorflow as hvd

    # Create training and validation datasets
    split_dataset = FLAGS.horovod

    train_set = create_dataset(
        FLAGS.train_files.split(","),
        batch_size=FLAGS.train_batch_size,
        epochs=FLAGS.epochs,
        augmentations=Config.augmentations,
        cache_path=FLAGS.feature_cache,
        train_phase=True,
        exception_box=exception_box,
        process_ahead=Config.num_devices * FLAGS.train_batch_size * 2,
        reverse=FLAGS.reverse_train,
        limit=FLAGS.limit_train,
        buffering=FLAGS.read_buffer,
        split_dataset=split_dataset,
    )

    iterator = tfv1.data.Iterator.from_structure(
        tfv1.data.get_output_types(train_set),
        tfv1.data.get_output_shapes(train_set),
        output_classes=tfv1.data.get_output_classes(train_set),
    )

    # Make initialization ops for switching between the two sets
    train_init_op = iterator.make_initializer(train_set)

    if FLAGS.dev_files:
        dev_sources = FLAGS.dev_files.split(",")
        dev_sets = [
            create_dataset(
                [source],
                batch_size=FLAGS.dev_batch_size,
                train_phase=False,
                exception_box=exception_box,
                process_ahead=Config.num_devices * FLAGS.dev_batch_size * 2,
                reverse=FLAGS.reverse_dev,
                limit=FLAGS.limit_dev,
                buffering=FLAGS.read_buffer,
                split_dataset=split_dataset,
            )
            for source in dev_sources
        ]
        dev_init_ops = [iterator.make_initializer(dev_set) for dev_set in dev_sets]

    if FLAGS.metrics_files:
        metrics_sources = FLAGS.metrics_files.split(",")
        metrics_sets = [
            create_dataset(
                [source],
                batch_size=FLAGS.dev_batch_size,
                train_phase=False,
                exception_box=exception_box,
                process_ahead=Config.num_devices * FLAGS.dev_batch_size * 2,
                reverse=FLAGS.reverse_dev,
                limit=FLAGS.limit_dev,
                buffering=FLAGS.read_buffer,
                split_dataset=split_dataset,
            )
            for source in metrics_sources
        ]
        metrics_init_ops = [
            iterator.make_initializer(metrics_set) for metrics_set in metrics_sets
        ]

    # Dropout
    dropout_rates = [
        tfv1.placeholder(tf.float32, name="dropout_{}".format(i)) for i in range(6)
    ]
    dropout_feed_dict = {
        dropout_rates[0]: FLAGS.dropout_rate,
        dropout_rates[1]: FLAGS.dropout_rate2,
        dropout_rates[2]: FLAGS.dropout_rate3,
        dropout_rates[3]: FLAGS.dropout_rate4,
        dropout_rates[4]: FLAGS.dropout_rate5,
        dropout_rates[5]: FLAGS.dropout_rate6,
    }
    no_dropout_feed_dict = {rate: 0.0 for rate in dropout_rates}

    # Building the graph
    # learning_rate_var = tfv1.get_variable(
    #     "learning_rate", initializer=FLAGS.learning_rate, trainable=False
    # )
    
    if FLAGS.horovod:
        # Effective batch size in synchronous distributed training is scaled by the number of workers. An increase in learning rate compensates for the increased batch size.
        optimizer = hps_create_optimizer(learning_rate_var * hvd.size())
        optimizer = hvd.DistributedOptimizer(optimizer)
    else:
        optimizer, learning_rate_var = hps_create_optimizer(trial)
    
    
    reduce_learning_rate_op = learning_rate_var.assign(
        tf.multiply(learning_rate_var, FLAGS.plateau_reduction)
    )

    # Enable mixed precision training
    if FLAGS.automatic_mixed_precision:
        log_info("Enabling automatic mixed precision training.")
        optimizer = tfv1.train.experimental.enable_mixed_precision_graph_rewrite(
            optimizer
        )

    if FLAGS.horovod:
        loss, non_finite_files = calculate_mean_edit_distance_and_loss(
            iterator, dropout_rates, reuse=False
        )
        gradients = optimizer.compute_gradients(loss)

        tfv1.summary.scalar(
            name="step_loss", tensor=loss, collections=["step_summaries"]
        )
        log_grads_and_vars(gradients)

        # global_step is automagically incremented by the optimizer
        global_step = tfv1.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(
            gradients, global_step=global_step
        )
    else:
        gradients, loss, non_finite_files = get_tower_results(
            iterator, optimizer, dropout_rates
        )

        # Average tower gradients across GPUs
        avg_tower_gradients = average_gradients(gradients)
        log_grads_and_vars(avg_tower_gradients)

        # global_step is automagically incremented by the optimizer
        global_step = tfv1.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(
            avg_tower_gradients, global_step=global_step
        )

    # Summaries
    step_summaries_op = tfv1.summary.merge_all("step_summaries")
    step_summary_writers = {
        "train": tfv1.summary.FileWriter(
            os.path.join(FLAGS.summary_dir, "train"), max_queue=120
        ),
        "dev": tfv1.summary.FileWriter(
            os.path.join(FLAGS.summary_dir, "dev"), max_queue=120
        ),
        "metrics": tfv1.summary.FileWriter(
            os.path.join(FLAGS.summary_dir, "metrics"), max_queue=120
        ),
    }

    human_readable_set_names = {
        "train": "Training",
        "dev": "Validation",
        "metrics": "Metrics",
    }

    # Checkpointing
    if Config.is_master_process:
        checkpoint_saver = tfv1.train.Saver(max_to_keep=FLAGS.max_to_keep)
        checkpoint_path = os.path.join(FLAGS.save_checkpoint_dir, "train")

        best_dev_saver = tfv1.train.Saver(max_to_keep=1)
        best_dev_path = os.path.join(FLAGS.save_checkpoint_dir, "best_dev")

        # Save flags next to checkpoints
        if not is_remote_path(FLAGS.save_checkpoint_dir):
            os.makedirs(FLAGS.save_checkpoint_dir, exist_ok=True)
        flags_file = os.path.join(FLAGS.save_checkpoint_dir, "flags.txt")
        with open_remote(flags_file, "w") as fout:
            fout.write(FLAGS.flags_into_string())

    if FLAGS.horovod:
        bcast = hvd.broadcast_global_variables(0)

    # with tfv1.Session(config=Config.session_config) as session:
    # log_debug("Session opened.")

    # Prevent further graph changes
    # tfv1.get_default_graph().finalize()

    # Load checkpoint or initialize variables
    load_or_init_graph_for_training(session)
    if FLAGS.horovod:
        bcast.run()

    def run_set(set_name, epoch, init_op, dataset=None):
        is_train = set_name == "train"
        train_op = apply_gradient_op if is_train else []
        feed_dict = dropout_feed_dict if is_train else no_dropout_feed_dict

        total_loss = 0.0
        step_count = 0

        step_summary_writer = step_summary_writers.get(set_name)
        checkpoint_time = time.time()

        if is_train and FLAGS.cache_for_epochs > 0 and FLAGS.feature_cache:
            feature_cache_index = FLAGS.feature_cache + ".index"
            if epoch % FLAGS.cache_for_epochs == 0 and os.path.isfile(
                feature_cache_index
            ):
                log_info("Invalidating feature cache")
                remove_remote(
                    feature_cache_index
                )  # this will let TF also overwrite the related cache data files

        # Setup progress bar
        class LossWidget(progressbar.widgets.FormatLabel):
            def __init__(self):
                progressbar.widgets.FormatLabel.__init__(
                    self, format="Loss: %(mean_loss)f"
                )

            def __call__(self, progress, data, **kwargs):
                data["mean_loss"] = total_loss / step_count if step_count else 0.0
                return progressbar.widgets.FormatLabel.__call__(
                    self, progress, data, **kwargs
                )

        if Config.is_master_process:
            prefix = "Epoch {} | {:>10}".format(
                epoch, human_readable_set_names[set_name]
            )
            widgets = [
                " | ",
                progressbar.widgets.Timer(),
                " | Steps: ",
                progressbar.widgets.Counter(),
                " | ",
                LossWidget(),
            ]
            suffix = " | Dataset: {}".format(dataset) if dataset else None
            pbar = create_progressbar(
                prefix=prefix, widgets=widgets, suffix=suffix
            ).start()

        # Initialize iterator to the appropriate dataset
        session.run(init_op)

        # Batch loop
        while True:
            try:
                _, current_step, batch_loss, problem_files, step_summary = session.run(
                    [
                        train_op,
                        global_step,
                        loss,
                        non_finite_files,
                        step_summaries_op,
                    ],
                    feed_dict=feed_dict,
                )
                exception_box.raise_if_set()
            except tf.errors.OutOfRangeError:
                exception_box.raise_if_set()
                break

            if problem_files.size > 0:
                problem_files = [f.decode("utf8") for f in problem_files[..., 0]]
                log_error(
                    "The following files caused an infinite (or NaN) "
                    "loss: {}".format(",".join(problem_files))
                )

            total_loss += batch_loss
            step_count += 1

            if Config.is_master_process:
                pbar.update(step_count)

                step_summary_writer.add_summary(step_summary, current_step)

                if (
                    is_train
                    and FLAGS.checkpoint_secs > 0
                    and time.time() - checkpoint_time > FLAGS.checkpoint_secs
                ):
                    checkpoint_saver.save(
                        session, checkpoint_path, global_step=current_step
                    )
                    checkpoint_time = time.time()

        if Config.is_master_process:
            pbar.finish()
        mean_loss = total_loss / step_count if step_count > 0 else 0.0
        return mean_loss, step_count

    log_info("STARTING Optimization")
    train_start_time = datetime.utcnow()
    best_dev_loss = float("inf")
    dev_losses = []
    epochs_without_improvement = 0
    try:
        for epoch in range(FLAGS.epochs):
            # Training
            if Config.is_master_process:
                log_progress("Training epoch %d..." % epoch)
            train_loss, _ = run_set("train", epoch, train_init_op)
            if Config.is_master_process:
                log_progress(
                    "Finished training epoch %d - loss: %f" % (epoch, train_loss)
                )
                checkpoint_saver.save(
                    session, checkpoint_path, global_step=global_step
                )

            if FLAGS.dev_files:
                # Validation
                dev_loss = 0.0
                total_steps = 0
                for source, init_op in zip(dev_sources, dev_init_ops):
                    if Config.is_master_process:
                        log_progress(
                            "Validating epoch %d on %s..." % (epoch, source)
                        )
                    set_loss, steps = run_set("dev", epoch, init_op, dataset=source)
                    dev_loss += set_loss * steps
                    total_steps += steps
                    if Config.is_master_process:
                        log_progress(
                            "Finished validating epoch %d on %s - loss: %f"
                            % (epoch, source, set_loss)
                        )

                dev_loss = dev_loss / total_steps
                dev_losses.append(dev_loss)

                # Count epochs without an improvement for early stopping and reduction of learning rate on a plateau
                # the improvement has to be greater than FLAGS.es_min_delta
                if dev_loss > best_dev_loss - FLAGS.es_min_delta:
                    epochs_without_improvement += 1
                else:
                    epochs_without_improvement = 0

                if Config.is_master_process:
                    # Save new best model
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        save_path = best_dev_saver.save(
                            session,
                            best_dev_path,
                            global_step=global_step,
                            latest_filename="best_dev_checkpoint",
                        )
                        log_info(
                            "Saved new best validating model with loss %f to: %s"
                            % (best_dev_loss, save_path)
                        )

                # Early stopping
                if (
                    FLAGS.early_stop
                    and epochs_without_improvement == FLAGS.es_epochs
                ):
                    if Config.is_master_process:
                        log_info(
                            "Early stop triggered as the loss did not improve the last {} epochs".format(
                                epochs_without_improvement
                            )
                        )
                    break

                # Reduce learning rate on plateau
                # If the learning rate was reduced and there is still no improvement
                # wait FLAGS.plateau_epochs before the learning rate is reduced again
                if (
                    FLAGS.reduce_lr_on_plateau
                    and epochs_without_improvement > 0
                    and epochs_without_improvement % FLAGS.plateau_epochs == 0
                ):
                    # Reload checkpoint that we use the best_dev weights again
                    reload_best_checkpoint(session)

                    # Reduce learning rate
                    session.run(reduce_learning_rate_op)
                    current_learning_rate = learning_rate_var.eval()
                    if Config.is_master_process:
                        log_info(
                            "Encountered a plateau, reducing learning rate to {}".format(
                                current_learning_rate
                            )
                        )

                        # Overwrite best checkpoint with new learning rate value
                        save_path = best_dev_saver.save(
                            session,
                            best_dev_path,
                            global_step=global_step,
                            latest_filename="best_dev_checkpoint",
                        )
                        log_info(
                            "Saved best validating model with reduced learning rate to: %s"
                            % (save_path)
                        )

            if FLAGS.metrics_files:
                # Read only metrics, not affecting best validation loss tracking
                for source, init_op in zip(metrics_sources, metrics_init_ops):
                    if Config.is_master_process:
                        log_progress(
                            "Metrics for epoch %d on %s..." % (epoch, source)
                        )
                    set_loss, _ = run_set("metrics", epoch, init_op, dataset=source)
                    if Config.is_master_process:
                        log_progress(
                            "Metrics for epoch %d on %s - loss: %f"
                            % (epoch, source, set_loss)
                        )

            print("-" * 80)

    except KeyboardInterrupt:
        pass
    if Config.is_master_process:
        log_info(
            "FINISHED optimization in {}".format(
                datetime.utcnow() - train_start_time
            )
        )

    final_dev_loss = dev_losses[-1]
    log_debug("Session closed.")

    return final_dev_loss


def setup_dirs(study_name, trial_number):
    #os.makedirs(f"{CHKPT_DIR}/logs/{study_name}/{trial_number}", exist_ok=True)
    os.makedirs(f"{CHKPT_DIR}/{study_name}/{trial_number}", exist_ok=True)
    #os.makedirs(f"{MODEL_DIR}/{study_name}/{trial_number}", exist_ok=True)

    return f"{CHKPT_DIR}/{study_name}/{trial_number}"

def hps_test():
    samples, loss = hps_evaluate(FLAGS.test_files.split(","), create_model)
    if FLAGS.test_output_file:
        save_samples_json(samples, FLAGS.test_output_file)
    return loss


def new_trial_callback(study, trial):
    chkpt_path = setup_dirs(study.name, trial.number + 1)
    FLAGS.checkpoint_dir = chkpt_path 

def objective(trial, session):
    if FLAGS.train_files:
        val_loss = hps_train(trial, session)
        # with tf.variable_scope("learning_rate", reuse=tf.AUTO_REUSE) as scope:
        #     val_loss = hps_train(trial)

        # tfv1.reset_default_graph()
        # tfv1.set_random_seed(FLAGS.random_seed)

    return float(val_loss)

def objective_tf(trial):
    # Clear clutter form previous session graphs.
    # tfv1.reset_default_graph()
    # tfv1.set_random_seed(FLAGS.random_seed)
    K.clear_session()

    with tfv1.Graph().as_default():
        with tfv1.Session(config=Config.session_config) as session:
            K.set_session(session)
            return objective(trial, session)

def main(_):
    initialize_globals()
    early_training_checks()

    lr_study = optuna.create_study(study_name="lr_study", direction='minimize')
    chkpt_dir = setup_dirs(lr_study.study_name, 0)
    FLAGS.checkpoint_dir = chkpt_dir
    lr_study.optimize(objective_tf, n_trials=25, callbacks=[new_trial_callback])




if __name__ == "__main__":
    create_flags()
    absl.app.run(main)

