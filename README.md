# File Specifications for .txt

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