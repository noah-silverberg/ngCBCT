# Nonstop-Gated CBCT Project

Code written by Noah Silverberg, based on initial version provided by Mitchell Yu.

## Code Overview

The data preparation, model training, and model usage code all lies within the ``pipeline.ipynb`` notebook, which heavily relies on utility functions from the ``pipeline`` folder.

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
> conda env create -f configuration.yml

You will also need the Python Matlab engine installed. On our workstation, this was done by navigating to ``C:\Program Files/MATLAB/R2023a/extern/engines/python`` and running:
> python -m pip install .

Of course, you might need to adjust your path based on your Matlab version and how you installed it. You also might need to use ``python3`` instead of ``python``.

In addition to these, you will need ``CudaTools`` and the FDK Matlab reconstruction code, written by Hao Zhang. If you do not have access to these, you can instead use the open-source ``TIGRE`` package from CERN, and the Python implementation of Dr. Zhang's code, which is in ``pipeline/FDK_half``. Note, however, that we have done direct comparisons between the Python and Matlab reconstructions, and they are not equivalent. In particular, the Python version leads to significant trunctation artifacts. These can be remedied by applying a cicular mask around the patient volume, but the overall quality is still degraded slightly (and model performance is as well -- we tested a few different ways of training models to mitigate this effect but were unsuccessful).

You will also definitely need a GPU, since these models are quite large. For reference, we are using an NVIDIA A6000, and these models are taking about 20-26 hours to train.

## Workflow

### Setup

The first thing you need to do is create a ``.txt`` file specifying which scans you would like to convert to Pytorch. The file specification is in a later section of this README. These files should be ``.mat`` files with keys ``odd_index``, ``angles``, and ``prj``.

The second thing you need to do is create a ``.txt`` file specifying which scans you would like to use for training. Note that you should specify these even if you already have a trained model, since these scans will also be the ones used for any other data preparation and processing.

Finally, you will need to create a configuration ``.yaml`` file. Please see the ``config.yaml`` file as an example. Note that you can create multiple ``.yaml`` files if you would like to train/process multiple models/ensembles sequentially.

### Pipeline

Almost all user-facing functionality lies in the ``pipeline.ipynb`` notebook. This notebook is relatively plug-and-play, and minimal tweaking should be required between different runs, since almost all needed functionality should be contained within the ``.yaml`` file. Nonetheless, inevitably things go wrong and you might need to deal with those. Or maybe you want to tweak a thing or two. A few common issues/desires are below, along with some potential solutions:
* Model crashes during training: inside the code blocks for training, there is some commented out code that allows you to resume training from a checkpoint. Just uncomment that and specify the checkpoint epoch number, and you should be good to go from there.
* Running low on storage: The projection files are much larger than the image ones. I already made it so that we only save projections for the validation and test sets, but these still do take up quite a bit of room -- you can just go into the code and comment out the line that saves these. It's pretty easy to find.
* Change GPU: If you go into the code at the top of the ``pipeline.ipynb`` notebook, you'll see a few lines where we tell ``os.environ`` which GPU(s) is (are) available, and we pick one. Just change this as needed.

## File Specifications for .txt

### Pytorch conversion scans

For the scans you want to convert to Pytorch, each line in the ``.txt`` file you create should be in the following format:
> PATIENT_ID SCAN_ID SCAN_TYPE

For example:
> 01 01 HF  
> 01 02 FF  
> ...

Ensure that there is no gap between any of the scans.

### Aggregation scans

For the scans you would like to aggregate for a given data version, the file should be in the format:
> SCAN_TYPE  
>  
> PATIENT_ID SCAN_ID  
> ...
>  
> PATIENT_ID SCAN_ID  
> ...  
>   
> PATIENT_ID SCAN_ID  
> ...

Where the ``SCAN_TYPE`` is either ``HF`` or ``FF``, and the blocks of specified scans are the train, validation, and test test, respectively.

So for example, in the following file:
> HF  
>  
> 01 01  
> 01 02  
>  
> 02 01  
> 03 01  
>   
> 04 01  
> 05 01

All the scans are half-fan, and the first two scans are used for training, the second two are used for validation, and the final two are for testing.

Ensure that there is a one line gap between each group (incl. after the scan type).