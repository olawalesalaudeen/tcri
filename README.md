# tcri
Code and data for Olawale Salaudeen, Sanmi Koyejo, "Adapting to Shifts in Latent Confounders using Observed Concepts and Proxies" International Conference on Artificial Intelligence and Statistics, 2024

## Create Spurious PACS
Spurious PACS is a version of PACS that induces a stronger distribution shift by spuriously correlating original PACS domains with a binary label.

```
python create_spurious_PACS.py \
    --original_PACS_dir <> \
    --spurious_PACS_dir <> \
    --confound_strength 1.0
```

## Run Experiments
Please see DomainBed/README.md for more detail.

1. ```python DomainBed/setup.py install```

2. ColoredMNIST.

```
python -m domainbed.scripts.sweep launch \
    --command_launcher multi_gpu \
    --data_dir <> \
    --datasets ColoredMNIST \
    --algorithms ERM IRM GroupDRO VREx IB_ERM IB_IRM TCRI_HSIC \
    --n_trials 25 \
    --n_trials 5 \
    --output_dir <> \
    --single_test_envs
```

3. Spurious PACS
```
python -m domainbed.scripts.sweep launch \
    --command_launcher multi_gpu \
    --data_dir <> \
    --datasets SpuriousPACS \
    --algorithms ERM IRM GroupDRO VREx IB_ERM IB_IRM TCRI_HSIC \
    --n_trials 5 \
    --n_trials 3 \
    --output_dir <> \
    --single_test_envs
```

4. TerraIncognita
```
python -m domainbed.scripts.sweep launch \
    --command_launcher multi_gpu \
    --data_dir <> \
    --datasets TerraIncognita \
    --algorithms ERM IRM GroupDRO VREx IB_ERM IB_IRM TCRI_HSIC \
    --n_trials 5 \
    --n_trials 3 \
    --output_dir <> \
    --single_test_envs
```

### Ablation
The following flags *--ablat_TCRI_TCRI* and *--ablat_TCRI_TIC* can be used to ablate the TCRI and TIC penalties, respectively.

## Changes to DomainBed
We primarly run our experiments with the included version of DomainBed. The primary difference between this version and the original is the addition of the TCRI algorithm to *Domainbed/domainbed/algorithms.py* and SpuriousPACS to *Domainbed/domainbed/datasets*.py. Additionally, to facilitate model selection using the 'tcri' criterion, we store the 'tcri' and 'tic' values in *'results.jsonl'* which is a record of model quality during training.

When the algorithm is not TCRI, its 'tcri' and 'tic' value is set to -1 by default. This can be found at *DomainBed/domainbed/lib/misc.py*.
