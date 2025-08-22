# Nonstop-Gated CBCT Project

*(Last Updated 8/15/2025)*

Code written by Noah Silverberg, based on initial version provided by Mitchell Yu. Feel free to contact me at noah.silverberg@yale.edu if you have any questions or encounter any issues. Or nsilverberg123@gmail.com if I've graduated...

## Code Overview

The data preparation, model training, and model usage code all lies within the ``pipeline.ipynb`` notebook, which heavily relies on utility functions from the ``pipeline`` folder. There are also a bunch of files that perform analysis and generate figures. See the ``Workflow`` section below for more information.

In particular, the following files exist within the ``pipeline`` folder:
* ``aggregate_ct.py``: Includes functionality for aggregating reconstructions. These aggregates are then used for image domain model (ID) training.
* ``aggregate_prj.py``: Similar to ``aggregate_ct.py``,  except for the projection domain (PD) training.
* ``apply_model.py``: Includes functions used for applying the model to data, i.e., either using the PD model on projections, or using the ID model on reconstructions.
* ``dsets.py``: Includes various ``torch.utils.data.Dataset`` class definitions that are used during training.
* ``network_instance.py``: Includes various model architecture definitions.
* ``paths.py``: Includes all logic relating to filepaths.
* ``proj.py``: Includes functionality related to pre-processing projections, including simulating nonstop-gated sinograms.
* ``train_appy_MK6_numpy.py``: Includes training application, which is used for training various model architectures with various hyperparameters in both domains.
* ``utils.py``: Miscellaneous utility functions.

## Requirements

You will need Python (I am using 3.9.18), Pytorch, and a few other basic numerical libraries (e.g., NumPy, SciPy, etc.). Specific versions can be found in ``configuration.yml``, and it is recommended that you create a new Conda environment  based on this file, to avoid any conflicts. To do this simply run:
```bash
conda env create -f configuration.yml
```

You will also need the Python Matlab engine installed. On our workstation, this was done by navigating to ``C:\Program Files/MATLAB/R2023a/extern/engines/python`` and running:
```bash
python -m pip install .
```

Of course, you might need to adjust your path based on your Matlab version and how you installed it. You also might need to use ``python3`` instead of ``python``.

In addition to these, you will need ``CudaTools`` and the FDK Matlab reconstruction code, written by Hao Zhang. *Note: some very minor modifications were made to the Matlab files to better interface with the Python (no functional changes were made). So you'll specifically need those versions.*

If you do not have access to these, you can instead use the open-source ``TIGRE`` package from CERN, and the Python implementation of Dr. Zhang's code, which is in ``pipeline/FDK_half``. Note, however, that we have done direct comparisons between the Python and Matlab reconstructions, and they are not equivalent. In particular, the Python version leads to significant trunctation artifacts. These can be remedied by applying a mask around the patient volume to try removing truncation artifacts, but the overall quality is still degraded slightly (and model performance is as well -- we tested a few different ways of training models to mitigate this effect but were unsuccessful).

You will also definitely need a GPU, since these models are quite large. For reference, we are using an NVIDIA A6000, and these models are taking many hours or days to train. It also takes a significant amount of time to process and analyze the data (the main bottleneck here is the PD model and FDK reconstruction, since the ID model is quite fast in comparison).

Finally, note that you'll need a lot of storage space. If you're not doing any stochastic models (such as MC Dropout), you'll take up much, much less space. But if you are, you'll need a few terabytes of storage, easily (for tuning and whatnot).

## Workflow

### Setup

The first thing you need to do is create a ``.txt`` file specifying which scans you would like to convert to Pytorch. The file specification is in a later section of this README. These files should be ``.mat`` files with keys ``odd_index``, ``angles``, and ``prj``.

The second thing you need to do is create a ``.txt`` file specifying which scans you would like to use for training. The file specification is later in this README. Note that you should specify these even if you already have a trained model, since these scans will also be the ones used for any other data preparation and processing.

Finally, you will need to create a configuration ``.yaml`` file. Please see the ``config.yaml`` file as an example. Note that you can create multiple ``.yaml`` files if you would like to train/process multiple models/ensembles sequentially. When you do this, each part of the ``pipeline.ipynb`` notebook will loop over the provided configuration files before moving onto the next part.

### Pipeline

All data pre-processing, training, and post-processing functionality lies in the ``pipeline.ipynb`` notebook. This notebook is relatively plug-and-play, and minimal tweaking should be required between different runs, since almost all needed functionality should be contained within the ``.yaml`` file. The only thing in this notebook you'll need to play with in the configuration cell, which is near the top. Everything you need to know should be explained there.

Nonetheless, inevitably things go wrong and you might need to deal with those. Or maybe you want to tweak a thing or two. A few common issues/desires are below, along with some potential solutions:
* Model crashes during training: use the ``start_checkpoint`` option in the config file to restart the training from the most recent checkpoint.
* Running low on storage: The projection files are much larger than the image ones. I already made it so that we only save projections for the validation and test sets, but these still do take up quite a bit of room -- you can just go into the code and comment out the line that saves these. It's pretty easy to find.
* Change GPU: If you go into the code at the top of the ``pipeline.ipynb`` notebook, you'll see a few lines where we tell ``os.environ`` which GPU(s) is (are) available, and we pick one. Just change this as needed.

One other note is that if you want to perform the duty cycle comparison test (where you switch from every second breathing cycle to every ``n``-th), you should use the ``sabotage.py`` script for that.

### Analysis

There are a few different files that perform analysis. The main workhorse here (which the rest depend on) is the ``compare.ipynb`` notebook. This contains all the code for SSIM, PSNR, RMV, etc. calculations. It's super easy to use, and handles all those computations for you. Then it will output two tables at the end (which you can then save as ``.csv`` files). The first table will give you per-scan per-model values. The second will summarize across all the scans for each model (i.e., take means and standard deviations of each of these values).

Then there are a few other files that include some basic analysis (mainly for the UQ paper, but some are for the DDCNN paper, too):
* ``evaluate_recons.py`` calculates the PSNR/SSIM of reconstructions from various models, and a few figures (see below) for the DDCNN paper.
* ``sabotage_violins.py`` performs t-tests for the different duty cycle RMVs, telling us which models are responsive to this out-of-distribution (OOD) detection task. It also generates relevant figures (see below). You'll need the ``results`` table saved as a ``.csv`` from the ``compare.ipynb`` notebook for this.
* ``basic_QC.py`` is a simple script that allows you to evlauate a toy QC mechanism, where you take the (unmodified) test set and use it to pick an RMV threshold for flagging. Then you see how many scans in other (modified) test sets would be flagged. You'll need the ``results`` table saved as a ``.csv`` from the ``compare.ipynb`` notebook for this.

### Figure Generation

There are some easy-to-use scripts for generating figures. Some of these are for the UQ paper, and some are for the DDCNN paper.
* ``evaluate_recons.py`` produces a few different figures for the DDCNN paper: (1) SSIM maps, (2) plots for each scan of PSNR and SSIM vs. slice number, (3) the same as #2 but with all models plotted together.
* ``sabotage_violins.py`` produces violin plots for the PSNR, SSIM, and RMV of each model, showing how these distributions change as we change the duty cycle. You'll need the ``results`` table saved as a ``.csv`` from the ``compare.ipynb`` notebook for this.
* ``uncert_vs_size.py`` generates line plots showing how the AUSE and Spearman correlations for the uncertainties evolve as we increase the number of samples taken from our UQ methods. It also includes violin plots in insets. You'll need the ``results`` table saved as a ``.csv`` from the ``compare.ipynb`` notebook for this.
* ``sabotage_figure.py`` generates a figure that displays the reconstructions from each model (vs. FDK) as we modify the duty-cycle.
* ``ddcnn_paper_figures.py`` generates a figure that displays the reconstructions from each model (vs. gated and FDK).

## File Structure

You can pick your file locations, generally. But you should absolutely be using the ``Directories`` and ``Files`` objects to manage filepaths (these are found in ``pipeline/paths.py``). You only need to really tweak the ``Directories`` object, then just make a ``Files`` object around it.

Below is an example instantiation of thsese objects, followed by an explanation:
```python
DIRECTORIES = Directories(
    mat_projections_dir       = 'mat',
    pt_projections_dir        = 'prj_pt',
    projections_aggregate_dir = 'phase/DS/aggregates/projections',
    projections_model_dir     = 'phase/DS/models/projections',
    projections_results_dir   = 'phase/DS/results/projections',
    projections_gated_dir     = 'gated/prj_mat',
    reconstructions_dir       = 'phase/DS/reconstructions',
    reconstructions_gated_dir = 'gated/fdk_recon',
    images_aggregate_dir      = 'phase/DS/aggregates/images',
    images_model_dir          = 'phase/DS/models/images',
    images_results_dir        = 'phase/DS/results/images',
    error_results_dir         = 'phase/DS/results/error_results',
)

FILES = Files(DIRECTORIES)
```

This is what the file structure here would look like:
```txt
.
├── mat/
├── prj_pt/
├── gated/
│   ├── prj_mat/
│   └── fdk_recon/
└── phase/
    └── DS/
        ├── aggregates/
        │   ├── projections/
        │   └── images/
        ├── models/
        │   ├── projections/
        │   └── images/
        ├── results/
        │   ├── projections/
        │   ├── images/
        │   └── error_results/
        └── reconstructions/
```
The ``mat`` folder would contain all the raw ``.mat`` files for each scan, with the ``prj``, ``odd_index``, and ``angles`` variables. Once you interpolate these and convert them to Pytorch tensors, they will be stored in ``prj_pt``.

The ``gated`` folder will contain things related to the gated versions of the scans (hence the name), and these are stored outside of the phase/dataversion folders since they should be independent of these things.

The ``phase/DS`` folder will contain all things related to the specific version of the experiment. This includes aggregated data, models, and model ouptuts. THis is all pretty straightforward, but one thing to note here is that ``reconstructions`` contains PD outputs that have been FDK-reconstructed, whereas ``results/images`` contains the ID outputs. Sorry, I know that might be a bit confusing. The thinking here is just that *direct* model outputs go into the ``results`` folder.

*Note: If you plan to start a new experiment using a new type of scan (for example, pancreas data when you usually use lung), make sure you update ALL of these, even the ``mat``, ``prj_pt``, and ``gated`` folders. If you do not do this, you will end up using some weird merged version of your data and the other type of data. This is because everything is saved with the same filenames, regardless of the data type (i.e., even if you use ``panc01_01.mat`` as your naming convention for pancreas data, all future files will still be called ``p01_01`` since this is how the ``Files`` objects are set up).*


## File Specifications for .txt

### Pytorch conversion scans

For the scans you want to convert to Pytorch, each line in the ``.txt`` file you create should be in the following format:
```txt
PATIENT_ID SCAN_ID SCAN_TYPE
```

For example:
```txt
01 01 HF  
01 02 FF  
...
```

Ensure that there is no gap between any of the scans.

### Aggregation scans

For the scans you would like to aggregate for a given data version, the file should be in the format:
```txt
SCAN_TYPE  
 
PATIENT_ID SCAN_ID  
...
 
PATIENT_ID SCAN_ID  
...  
  
PATIENT_ID SCAN_ID  
...
```

Where the ``SCAN_TYPE`` is either ``HF`` or ``FF``, and the blocks of specified scans are the train, validation, and test test, respectively.

So for example, in the following file:
```txt
HF  
 
01 01  
01 02  
 
02 01  
03 01  
  
04 01  
05 01
```

All the scans are half-fan, and the first two scans are used for training, the second two are used for validation, and the final two are for testing.

Ensure that there is a one line gap between each group (incl. after the scan type).