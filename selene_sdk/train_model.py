"""
This module provides the `TrainModel` class and supporting methods.
"""
import logging
import math
import os
import shutil
from time import strftime
from time import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from .utils import initialize_logger
from .utils import load_model_from_state_dict
from .utils import PerformanceMetrics


logger = logging.getLogger("selene")


def auc_u_test(labels, predictions):
    len_pos = int(np.sum(labels))
    len_neg = len(labels) - len_pos
    rank_sum = np.sum(rankdata(predictions)[labels == 1])
    u_value = rank_sum - (len_pos * (len_pos + 1)) / 2
    auc = u_value / (len_pos * len_neg)
    return auc


def _metrics_logger(name, out_filepath):
    logger = logging.getLogger("{0}".format(name))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    file_handle = logging.FileHandler(
        os.path.join(out_filepath, "{0}.txt".format(name)))
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)
    return logger


class TrainModel(object):
    """
    This class ties together the various objects and methods needed to
    train and validate a model.

    TrainModel saves a checkpoint model (overwriting it after
    `save_checkpoint_every_n_steps`) as well as a best-performing model
    (overwriting it after `report_stats_every_n_steps` if the latest
    validation performance is better than the previous best-performing
    model) to `output_dir`.

    TrainModel also outputs 2 files that can be used to monitor training
    as Selene runs: `selene_sdk.train_model.train.txt` (training loss) and
    `selene_sdk.train_model.validation.txt` (validation loss & average
    ROC AUC). The columns in these files can be used to quickly visualize
    training history (e.g. you can use `matplotlib`, `plt.plot(auc_list)`)
    and see, for example, whether the model is still improving, if there are
    signs of overfitting, etc.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    data_sampler : selene_sdk.samplers.Sampler
        The example generator.
    loss_criterion : torch.nn._Loss
        The loss function to optimize.
    optimizer_class : torch.optim.Optimizer
        The optimizer to minimize loss with.
    optimizer_kwargs : dict
        The dictionary of keyword arguments to pass to the optimizer's
        constructor.
    batch_size : int
        Specify the batch size to process examples. Should be a power of 2.
    report_stats_every_n_steps : int
        The frequency with which to report summary statistics. You can
        set this value to be equivalent to a training epoch
        (`n_steps * batch_size`) being the total number of samples
        seen by the model so far. Selene evaluates the model on the validation
        dataset every `report_stats_every_n_steps` and, if the model obtains
        the best performance so far (based on the user-specified loss function),
        Selene saves the model state to a file called `best_model.pth.tar` in
        `output_dir`.
    output_dir : str
        The output directory to save model checkpoints and logs in.
    save_checkpoint_every_n_steps : int or None, optional
        Default is 1000. If None, set to the same value as
        `report_stats_every_n_steps`
    save_new_checkpoints_after_n_steps : int or None, optional
        Default is None. The number of steps after which Selene will
        continually save new checkpoint model weights files
        (`checkpoint-<TIMESTAMP>.pth.tar`) every
        `save_checkpoint_every_n_steps`. Before this point,
        the file `checkpoint.pth.tar` is overwritten every
        `save_checkpoint_every_n_steps` to limit the memory requirements.
    n_validation_samples : int or None, optional
        Default is `None`. Specify the number of validation samples in the
        validation set. If `n_validation_samples` is `None` and the data sampler
        used is the `selene_sdk.samplers.IntervalsSampler` or
        `selene_sdk.samplers.RandomSampler`, we will retrieve 32000
        validation samples. If `None` and using
        `selene_sdk.samplers.MultiFileSampler`, we will use all
        available validation samples from the appropriate data file.
    n_test_samples : int or None, optional
        Default is `None`. Specify the number of test samples in the test set.
        If `n_test_samples` is `None` and

            - the sampler you specified has no test partition, you should not
              specify `evaluate` as one of the operations in the `ops` list.
              That is, Selene will not automatically evaluate your trained
              model on a test dataset, because the sampler you are using does
              not have any test data.
            - the sampler you use is of type `selene_sdk.samplers.OnlineSampler`
              (and the test partition exists), we will retrieve 640000 test
              samples.
            - the sampler you use is of type
              `selene_sdk.samplers.MultiFileSampler` (and the test partition
              exists), we will use all the test samples available in the
              appropriate data file.

    cpu_n_threads : int, optional
        Default is 1. Sets the number of OpenMP threads used for parallelizing
        CPU operations.
    use_cuda : bool, optional
        Default is `False`. Specify whether a CUDA-enabled GPU is available
        for torch to use during training.
    data_parallel : bool, optional
        Default is `False`. Specify whether multiple GPUs are available
        for torch to use during training.
    logging_verbosity : {0, 1, 2}, optional
        Default is 2. Set the logging verbosity level.

            * 0 - Only warnings will be logged.
            * 1 - Information and warnings will be logged.
            * 2 - Debug messages, information, and warnings will all be\
                  logged.

    checkpoint_resume : str or None, optional
        Default is `None`. If `checkpoint_resume` is not None, it should be the
        path to a model file generated by `torch.save` that can now be read
        using `torch.load`.

    Attributes
    ----------
    model : torch.nn.Module
        The model to train.
    sampler : selene_sdk.samplers.Sampler
        The example generator.
    loss_criterion : torch.nn._Loss
        The loss function to optimize.
    optimizer_class : torch.optim.Optimizer
        The optimizer to minimize loss with.
    batch_size : int
        The size of the mini-batch to use during training.
    nth_step_report_stats : int
        The frequency with which to report summary statistics.
    nth_step_save_checkpoint : int
        The frequency with which to save a model checkpoint.
    use_cuda : bool
        If `True`, use a CUDA-enabled GPU. If `False`, use the CPU.
    data_parallel : bool
        Whether to use multiple GPUs or not.
    output_dir : str
        The directory to save model checkpoints and logs.
    training_loss : list(float)
        The current training loss.
    metrics : dict
        A dictionary that maps metric names (`str`) to metric functions.
        By default, this contains `"roc_auc"`, which maps to
        `sklearn.metrics.roc_auc_score`, and `"average_precision"`,
        which maps to `sklearn.metrics.average_precision_score`.
    multidatasets : bool
        If `True`, use multiple datasets and multihead model
    """

    def __init__(self,
                 model,
                 data_sampler,
                 loss_criterion,
                 optimizer_class,
                 optimizer_kwargs,
                 batch_size,
                 report_stats_every_n_steps,
                 output_dir,
                 save_checkpoint_every_n_steps=1000,
                 save_new_checkpoints_after_n_steps=None,
                 report_gt_feature_n_positives=10,
                 n_validation_samples=None,
                 n_test_samples=None,
                 cpu_n_threads=1,
                 use_cuda=False,
                 data_parallel=False,
                 logging_verbosity=2,
                 checkpoint_resume=None,
                 metrics=dict(roc_auc=roc_auc_score,
                              average_precision=average_precision_score),
                 compute_metrics_on=None,
                 multidatasets=False,
                 disable_scheduler=False):
        """
        Constructs a new `TrainModel` object.
        """
        self.model = model
        self.sampler = data_sampler
        self.criterion = loss_criterion
        self.optimizer = optimizer_class(
            self.model.parameters(), **optimizer_kwargs)
        print("optim: {0}".format(optimizer_class))
        self.batch_size = batch_size
        self.nth_step_report_stats = report_stats_every_n_steps
        self.nth_step_save_checkpoint = None
        if not save_checkpoint_every_n_steps:
            self.nth_step_save_checkpoint = report_stats_every_n_steps
        else:
            self.nth_step_save_checkpoint = save_checkpoint_every_n_steps

        self.save_new_checkpoints = save_new_checkpoints_after_n_steps
        torch.set_num_threads(1)  #cpu_n_threads)

        self.use_cuda = use_cuda
        self.data_parallel = data_parallel

        if self.data_parallel:
            self.model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            logger.debug("Wrapped model in DataParallel")

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()
            logger.debug("Set modules to use CUDA")

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        initialize_logger(
            os.path.join(self.output_dir, "{0}.log".format(__name__)),
            verbosity=logging_verbosity)

        self._train_sampler = self.sampler._samplers["train"]

        self._create_validation_set(n_samples=n_validation_samples,
                                    compute_metrics_on=compute_metrics_on)
        self._validation_metrics = PerformanceMetrics(
            self.sampler.get_feature_from_index,
            report_gt_feature_n_positives=report_gt_feature_n_positives,
            metrics=dict(roc_auc_approx=auc_u_test),
            num_workers=cpu_n_threads)

        if "test" in self.sampler.modes:
            self._all_test_seqs = None
            self._n_test_samples = n_test_samples
            self._test_metrics = PerformanceMetrics(
                self.sampler.get_feature_from_index,
                report_gt_feature_n_positives=report_gt_feature_n_positives,
                metrics=dict(roc_auc_approx=auc_u_test),
                num_workers=cpu_n_threads)

        # self._start_step = 0
        self._min_loss = float("inf") # TODO: Should this be set when it is used later? Would need to if we want to train model 2x in one run.
        if checkpoint_resume is not None:
            checkpoint = torch.load(
                checkpoint_resume,
                map_location=lambda storage, location: storage)

            self.model = load_model_from_state_dict(
                checkpoint["state_dict"], self.model)
            self._min_loss = checkpoint["min_loss"]
            self.optimizer.load_state_dict(
                checkpoint["optimizer"])
            if self.use_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            """
            logger.info(
                ("Resuming from checkpoint: step {0}, min loss {1}").format(
                    self._start_step, self._min_loss))
            """
        self._train_logger = _metrics_logger(
                "{0}.train".format(__name__), self.output_dir)
        self._validation_logger = _metrics_logger(
                "{0}.validation".format(__name__), self.output_dir)

        self._train_logger.info("loss")
        self._validation_logger.info("\t".join(["loss"] +
            sorted([x for x in self._validation_metrics.metrics.keys()])))

        self.multidatasets  = multidatasets
        self.disable_scheduler = disable_scheduler

    def _create_validation_set(self, n_samples=None, compute_metrics_on=None):
        """
        Generates the set of validation examples.

        Parameters
        ----------
        n_samples : int or None, optional
            Default is `None`. The size of the validation set. If `None`,
            will use all validation examples in the sampler.

        """
        logger.info("Creating validation dataset.")
        t_i = time()
        self._all_validation_seqs, self._all_validation_targets = \
            self.sampler.get_validation_set(
                self.batch_size, n_samples=n_samples)

        n_cols = self._all_validation_targets.shape[1]
        if compute_metrics_on and isinstance(compute_metrics_on, list):
            self._check_cols = list(
                range(compute_metrics_on[0], compute_metrics_on[1]))
        elif compute_metrics_on and isinstance(compute_metrics_on, int):
            self._check_cols = list(range(0, n_cols, compute_metrics_on))
        else:
            self._check_cols = list(range(n_cols))
        t_f = time()
        logger.info(("{0} s to load {1} validation examples ({2} validation "
                     "batches) to evaluate after each training step.").format(
                      t_f - t_i,
                      self._all_validation_seqs.shape[0],
                      self._all_validation_seqs.shape[0] // self.batch_size))

    def create_test_set(self):
        """
        Loads the set of test samples.
        We do not create the test set in the `TrainModel` object until
        this method is called, so that we avoid having to load it into
        memory until the model has been trained and is ready to be
        evaluated.

        """
        logger.info("Creating test dataset.")
        t_i = time()
        self._all_test_seqs, self._all_test_targets = \
            self.sampler.get_test_set(
                self.batch_size, n_samples=self._n_test_samples)
        t_f = time()
        logger.info(("{0} s to load {1} test examples ({2} test batches) "
                     "to evaluate after all training steps.").format(
                      t_f - t_i,
                      self._all_test_seqs.shape[0],
                      self._all_test_seqs.shape[0] // self.batch_size))
        """
        output_file = os.path.join(self.output_dir, "test_targets_and_preds.h5")
        with h5py.File(output_file, 'a') as fh:
            fh.create_dataset("targets", data=self._all_test_targets.todense())
        """

    def _get_batch(self):
        """
        Fetches a mini-batch of examples

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the examples and targets.

        """
        t_i_sampling = time()
        batch_sequences, batch_targets = self.sampler.sample(
            batch_size=self.batch_size)
        t_f_sampling = time()
        logger.debug(
            ("[BATCH] Time to sample {0} examples: {1} s.").format(
                 self.batch_size,
                 t_f_sampling - t_i_sampling))
        return (batch_sequences, batch_targets)

    def train_and_validate(self):
        """
        Trains the model and measures validation performance.

        """
        min_loss = self._min_loss
        if not self.disable_scheduler:
            scheduler = ReduceLROnPlateau(
                self.optimizer, 'max', patience=24, verbose=True,
                factor=0.8)

        self.model.train()
        time_per_step = []
        for i, batch in enumerate(self._train_sampler):
            t_i = time()
            inputs, targets = Variable(batch[0]), Variable(batch[1])
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            if self.multidatasets:
                #TODO: deal with this better
                try:
                    self.model.module.model.current_classifier = self._train_sampler.current_dataset
                except:
                    try:
                        self.model.model.current_classifier = self._train_sampler.current_dataset
                    except:
                        self.model.current_classifier = self._train_sampler.current_dataset

            predictions = self.model.forward(inputs.transpose(1, 2))
            assert (predictions.data.cpu().numpy().all() >= 0. and
                    predictions.data.cpu().numpy().all() <= 1.)
            assert (targets.data.cpu().numpy().all() >= 0. and
                    targets.data.cpu().numpy().all() <= 1.)
            loss = self.criterion(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value = loss.item()
            t_f = time()

            if self.multidatasets:
                self._train_sampler.current_dataset  += 1
                self._train_sampler.current_dataset = self._train_sampler.current_dataset % len(self._train_sampler.samplers)

            if i % 100 == 0:
                logger.debug("{0}: {1} s to propagate sample".format(i, t_f - t_i))
            time_per_step.append(t_f - t_i)
            if i % self.nth_step_save_checkpoint == 0:
                checkpoint_dict = {
                    "step": i,
                    "arch": self.model.__class__.__name__,
                    "state_dict": self.model.state_dict(),
                    "min_loss": min_loss,
                    "optimizer": self.optimizer.state_dict()
                }
                if self.save_new_checkpoints is not None and \
                        self.save_new_checkpoints >= i:
                    checkpoint_filename = "checkpoint-{0}".format(
                        strftime("%m%d%H%M%S"))
                    self._save_checkpoint(
                        checkpoint_dict, False, filename=checkpoint_filename)
                    logger.debug("Saving checkpoint `{0}.pth.tar`".format(
                        checkpoint_filename))
                else:
                    self._save_checkpoint(
                        checkpoint_dict, False)

                if i and i % self.nth_step_report_stats == 0:
                    logger.info(("[STEP {0}] average number "
                                 "of steps per second: {1:.1f}").format(
                        i, 1. / np.average(time_per_step)))
                    time_per_step = []
                    self._train_logger.info(loss_value)
                    logger.info("training loss: {0}".format(loss_value))
                    valid_scores = self.validate()
                    if self.multidatasets:
                        #TODO: deal with this better
                        try:
                            self.model.module.model.current_classifier = 0
                        except:
                            try:
                                self.model.model.current_classifier = 0
                            except:
                                self.model.current_classifier = 0

                    validation_loss = valid_scores["loss"]
                    to_log = [str(validation_loss)]

                    for k in sorted(self._validation_metrics.metrics.keys()):
                        if k in valid_scores and valid_scores[k]:
                            to_log.append(str(valid_scores[k]))
                        else:
                            to_log.append("NA")
                    self._validation_logger.info("\t".join(to_log))
                    if not self.disable_scheduler:
                        scheduler.step(math.ceil(validation_loss * 1000.0) / 1000.0)

                    if validation_loss < min_loss:
                        min_loss = validation_loss
                        self._save_checkpoint({
                          "step": i,
                          "arch": self.model.__class__.__name__,
                          "state_dict": self.model.state_dict(),
                          "min_loss": min_loss,
                          "optimizer": self.optimizer.state_dict()}, True)
                        logger.debug("Updating `best_model.pth.tar`")
                        logger.info("validation loss: {0}".format(validation_loss))

        #self.sampler.save_dataset_to_file("train", close_filehandle=True)

    def train(self):
        """
        Trains the model on a batch of data.

        Returns
        -------
        float
            The training loss.

        """
        self.model.train()
        #self.sampler.set_mode("train")

        inputs, targets = self._get_batch()
        inputs = torch.Tensor(inputs)
        targets = torch.Tensor(targets)

        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)

        predictions = self.model(inputs.transpose(1, 2))
        loss = self.criterion(predictions, targets)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
        loss_value = loss.item()

        # reduce memory usage
        del loss
        del inputs
        del targets
        del predictions
        torch.cuda.empty_cache()
        return loss_value

    def _evaluate_on_data(self, data_seqs, data_targets):
        """
        Makes predictions for some labeled input data.

        Parameters
        ----------
        TODO

        Returns
        -------
        tuple(float, list(numpy.ndarray))
            Returns the average loss, and the list of all predictions.

        """
        self.model.eval()

        batch_losses = []
        all_predictions = np.zeros((data_targets.shape[0], data_targets.shape[1]))
        count = 0
        while count < data_targets.shape[0]:
            remainder = min(data_targets.shape[0] - count, self.batch_size)
            inputs = data_seqs[count:count + remainder, :, :].astype(float)
            targets = data_targets[count:count + remainder, :].astype(float)
            inputs = torch.Tensor(inputs)
            targets = torch.Tensor(targets)

            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            with torch.no_grad():
                inputs = Variable(inputs)
                targets = Variable(targets)
                predictions = self.model(
                    inputs.transpose(1, 2))
                loss = self.criterion(predictions, targets)

                all_predictions[count:count + remainder, :] = \
                    predictions.data.cpu().numpy()

                batch_losses.append(loss.item())
            count += remainder

            del inputs
            del targets
            del predictions
            del loss
            torch.cuda.empty_cache()
        return np.average(batch_losses), all_predictions

    def validate(self):
        """
        Measures model validation performance.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the validation set.

        """
        average_loss, all_predictions = self._evaluate_on_data(
            self._all_validation_seqs, self._all_validation_targets)
        average_scores = self._validation_metrics.update(
            all_predictions[:, self._check_cols],
            self._all_validation_targets[:, self._check_cols])
        for name, score in average_scores.items():
            logger.info("validation {0}: {1}".format(name, score))

        average_scores["loss"] = average_loss
        return average_scores

    def evaluate(self):
        """
        Measures the model test performance.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the test set.

        """
        del self._all_validation_seqs
        del self._all_validation_targets

        if self._all_test_seqs is None:
            self.create_test_set()
        average_loss, all_predictions = self._evaluate_on_data(
            self._all_test_seqs, self._all_test_targets)

        self._test_metrics.add_metric("average_precision", average_precision_score)

        average_scores = self._test_metrics.update(all_predictions,
                                                   self._all_test_targets)


        for name, score in average_scores.items():
            logger.info("test {0}: {1}".format(name, score))

        test_performance = os.path.join(
            self.output_dir, "test_performance.txt")
        feature_scores_dict = self._test_metrics.write_feature_scores_to_file(
            test_performance)

        output_file = os.path.join(self.output_dir, "test_targets_and_preds.h5")
        with h5py.File(output_file, 'w') as fh:
            fh.create_dataset("targets", data=self._all_test_targets.todense())
            fh.create_dataset("preds", data=all_predictions)

        average_scores["loss"] = average_loss

        #self._test_metrics.visualize(
        #    all_predictions, self._all_test_targets, self.output_dir)

        return (average_scores, feature_scores_dict)

    def _save_checkpoint(self,
                         state,
                         is_best,
                         filename="checkpoint"):
        """
        Saves snapshot of the model state to file. Will save a checkpoint
        with name `<filename>.pth.tar` and, if this is the model's best
        performance so far, will save the state to a `best_model.pth.tar`
        file as well.

        Models are saved in the state dictionary format. This is a more
        stable format compared to saving the whole model (which is another
        option supported by PyTorch). Note that we do save a number of
        additional, Selene-specific parameters in the dictionary
        and that the actual `model.state_dict()` is stored in the `state_dict`
        key of the dictionary loaded by `torch.load`.

        See: https://pytorch.org/docs/stable/notes/serialization.html for more
        information about how models are saved in PyTorch.

        Parameters
        ----------
        state : dict
            Information about the state of the model. Note that this is
            not `model.state_dict()`, but rather, a dictionary containing
            keys that can be used for continued training in Selene
            _in addition_ to a key `state_dict` that contains
            `model.state_dict()`.
        is_best : bool
            Is this the model's best performance so far?
        filename : str, optional
            Default is "checkpoint". Specify the checkpoint filename. Will
            append a file extension to the end of the `filename`
            (e.g. `checkpoint.pth.tar`).

        Returns
        -------
        None

        """
        logger.debug("[TRAIN] {0}: Saving model state to file.".format(
            state["step"]))
        cp_filepath = os.path.join(
            self.output_dir, filename)
        torch.save(state, "{0}.pth.tar".format(cp_filepath))
        if is_best:
            best_filepath = os.path.join(self.output_dir, "best_model")
            shutil.copyfile("{0}.pth.tar".format(cp_filepath),
                            "{0}.pth.tar".format(best_filepath))

