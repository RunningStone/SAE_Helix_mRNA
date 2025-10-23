# Downloading UTR_LM Data from Shared Google Drive

This document describes the steps to download the UTR_LM dataset from a shared Google Drive folder.

## Dataset Information

- **Source**: Shared Google Drive folder
- **Original Link**: `https://drive.google.com/drive/folders/1oGGgQ33cbx340vXsH_Ds_Py6Ad0TslLD?usp=share_link`
- **Total Size**: ~10.9 GB
- **Total Files**: 524 files

## Dataset Contents

The `Data/Data` folder contains the following subdirectories:

- `Experimental_Data/` - Experimental datasets
- `FiveSpecies_EnsemblDatabase/` - Ensembl database for five species
- `IndependentTest_VaryLength_Sample/` - Independent test samples with varying lengths
- `MJ_MTtrans_RP_single_task/` - MJ MTtrans RP single task data
- `MRL_Random50Nuc_SynthesisLibrary_Sample/` - MRL random 50 nucleotide synthesis library samples
- `Pretrained_Data/` - Pre-trained model data
- `TE_REL_Endogenous_Cao/` - TE REL endogenous Cao dataset

## Download Steps

### 1. set short-cut for shared link in google drive and use rclone to download 


### 2. transfered into structure 

```python
dataset_entry = {
    'sequence': 'ATCGATCG...',  # ✅ all files included 
    'annotations': {
        'functional': {
            'mrl': 2.34,              # ✅ in MRL dataset
            'te': 1.52,               # ✅ in TE and HEK dataset
            'expression_level': 3.21   # ✅ rpkm_riboseq/rnaseq
        },
        'structural': {
            'secondary_structure': '(((....)))',  # ✅ in fasta file
            'mfe': -12.5,                         # ✅ in fasta header
            'gc_content': 0.55,                   # ✅ CGratio/CGperc
            'length': 75                          # ✅ multiple length fields
        },
        'regulatory': {
            'uorf_count': 2,           # ✅ uORF field exists
            'uaug_count': 1,           # ✅ uAUG field exists
            'codon_usage': {...}       # ✅ all codon frequencies
        }
    },
    'metadata': {
        'source': 'GSM3130435_egfp_unmod_1',  # ✅
        'cell_line': 'HEK293',                # ✅
        'data_type': 'endogenous'             # ✅
    }
}
```