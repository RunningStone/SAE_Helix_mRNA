"""
Data preprocessing script for mRNA dataset.
Converts UTR_LM dataset into structured format with functional, structural, and regulatory annotations.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class mRNADatasetPreprocessor:
    """Preprocessor for mRNA dataset with multiple annotation types."""
    
    def __init__(self, 
                 data_dir: str = "/home/pan/Experiments/EXPs/2025_10_FM_explainability/DATA/UTR_LM",
                 output_dir: str = "/home/pan/Experiments/EXPs/2025_10_FM_explainability/DATA/transfered_dataset",
                 chunk_size: int = 5000):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Root directory of UTR_LM dataset
            output_dir: Output directory for processed data
            chunk_size: Number of entries per chunk file
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data sources
        self.mrl_dir = self.data_dir / "MRL_Random50Nuc_SynthesisLibrary_Sample"
        self.te_dir = self.data_dir / "TE_REL_Endogenous_Cao"
        self.experimental_dir = self.data_dir / "Experimental_Data"
        
    def parse_fasta_structure(self, fasta_file: Path) -> Dict[str, Tuple[float, str]]:
        """
        Parse FASTA file with structure annotations.
        
        Returns:
            Dict mapping sequence to (mfe, secondary_structure)
        """
        structure_dict = {}
        
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            if lines[i].startswith('>'):
                # Parse header: >-19.6|.((((((((..........|Sample|4.1
                header = lines[i].strip()
                parts = header[1:].split('|')
                
                if len(parts) >= 2:
                    mfe = float(parts[0])
                    secondary_structure = parts[1]
                    
                    # Next line is sequence
                    if i + 1 < len(lines):
                        sequence = lines[i + 1].strip()
                        structure_dict[sequence] = (mfe, secondary_structure)
                
                i += 2
            else:
                i += 1
        
        logger.info(f"Parsed {len(structure_dict)} structures from {fasta_file.name}")
        return structure_dict
    
    def calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content of a sequence."""
        if len(sequence) == 0:
            return 0.0
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
    
    def extract_codon_usage(self, row: pd.Series) -> Dict[str, float]:
        """Extract codon usage from BiologyFeatures row."""
        codon_cols = [col for col in row.index if col.startswith('codon_')]
        codon_usage = {}
        
        for col in codon_cols:
            amino_acid = col.replace('codon_', '')
            if pd.notna(row[col]):
                codon_usage[amino_acid] = float(row[col])
        
        return codon_usage
    
    def process_mrl_dataset(self) -> List[Dict]:
        """Process MRL (Mean Ribosome Load) dataset."""
        entries = []
        
        # Find all BiologyFeatures files
        bio_feature_files = list(self.mrl_dir.glob("*_BiologyFeatures.csv"))
        logger.info(f"Found {len(bio_feature_files)} BiologyFeatures files")
        
        # Find corresponding structure files
        structure_files = list(self.mrl_dir.glob("Sample_*_energy_structure_*.fasta"))
        logger.info(f"Found {len(structure_files)} structure files")
        
        # Parse all structure files
        all_structures = {}
        for struct_file in tqdm(structure_files, desc="Parsing structure files"):
            structures = self.parse_fasta_structure(struct_file)
            all_structures.update(structures)
        
        logger.info(f"Total structures parsed: {len(all_structures)}")
        
        # Process each BiologyFeatures file
        for bio_file in tqdm(bio_feature_files, desc="Processing BiologyFeatures"):
            # Extract metadata from filename
            # Example: 4.1_train_data_GSM3130435_egfp_unmod_1_BiologyFeatures.csv
            filename = bio_file.stem
            parts = filename.split('_')
            
            source = None
            data_type = 'random'  # MRL is synthesis library (random)
            
            # Extract GSM ID if present
            for part in parts:
                if part.startswith('GSM'):
                    source = '_'.join([p for p in parts if 'GSM' in p or 'egfp' in p or 'mcherry' in p])
                    source = source.replace('_BiologyFeatures', '')
                    break
            
            # Read BiologyFeatures
            try:
                df = pd.read_csv(bio_file)
                logger.info(f"Processing {bio_file.name}: {len(df)} sequences")
                
                for idx, row in df.iterrows():
                    # Get sequence
                    sequence = row.get('utr', None)
                    if pd.isna(sequence) or sequence is None:
                        continue
                    
                    # Remove padding if present
                    sequence = str(sequence).replace('<pad>', '')
                    
                    # Build entry
                    entry = {
                        'sequence': sequence,
                        'annotations': {
                            'functional': {},
                            'structural': {},
                            'regulatory': {}
                        },
                        'metadata': {
                            'source': source if source else filename,
                            'cell_line': 'HEK293',  # MRL data is from HEK293
                            'data_type': data_type
                        }
                    }
                    
                    # Functional annotations
                    if 'rl' in row.index and pd.notna(row['rl']):
                        entry['annotations']['functional']['mrl'] = float(row['rl'])
                    
                    # Structural annotations
                    entry['annotations']['structural']['length'] = len(sequence)
                    
                    # GC content
                    if 'CGratio' in row.index and pd.notna(row['CGratio']):
                        entry['annotations']['structural']['gc_content'] = float(row['CGratio'])
                    elif 'CGperc' in row.index and pd.notna(row['CGperc']):
                        entry['annotations']['structural']['gc_content'] = float(row['CGperc'])
                    else:
                        entry['annotations']['structural']['gc_content'] = self.calculate_gc_content(sequence)
                    
                    # Try to find structure
                    if sequence in all_structures:
                        mfe, secondary_structure = all_structures[sequence]
                        entry['annotations']['structural']['mfe'] = mfe
                        entry['annotations']['structural']['secondary_structure'] = secondary_structure
                    
                    # Regulatory annotations
                    if 'uORF' in row.index and pd.notna(row['uORF']):
                        entry['annotations']['regulatory']['uorf_count'] = int(row['uORF'])
                    
                    if 'uAUG' in row.index and pd.notna(row['uAUG']):
                        entry['annotations']['regulatory']['uaug_count'] = int(row['uAUG'])
                    
                    # Codon usage
                    codon_usage = self.extract_codon_usage(row)
                    if codon_usage:
                        entry['annotations']['regulatory']['codon_usage'] = codon_usage
                    
                    entries.append(entry)
                    
            except Exception as e:
                logger.error(f"Error processing {bio_file.name}: {e}")
                continue
        
        logger.info(f"Processed {len(entries)} entries from MRL dataset")
        return entries
    
    def process_te_dataset(self) -> List[Dict]:
        """Process TE (Translation Efficiency) endogenous dataset."""
        entries = []
        
        # Process HEK_sequence.csv
        hek_file = self.te_dir / "HEK_sequence.csv"
        muscle_file = self.te_dir / "Muscle_sequence.csv"
        
        # Find structure files
        structure_files = list(self.te_dir.glob("*_energy_structure_*.fasta"))
        
        # Parse structures
        all_structures = {}
        for struct_file in structure_files:
            structures = self.parse_fasta_structure(struct_file)
            all_structures.update(structures)
        
        logger.info(f"Parsed {len(all_structures)} structures from TE dataset")
        
        # Process HEK data
        if hek_file.exists():
            try:
                df = pd.read_csv(hek_file)
                logger.info(f"Processing HEK data: {len(df)} sequences")
                
                for idx, row in df.iterrows():
                    sequence = row.get('utr', None)
                    if pd.isna(sequence) or sequence is None:
                        continue
                    
                    sequence = str(sequence).replace('<pad>', '')
                    
                    entry = {
                        'sequence': sequence,
                        'annotations': {
                            'functional': {},
                            'structural': {},
                            'regulatory': {}
                        },
                        'metadata': {
                            'source': row.get('ensembl_tx_id', 'unknown'),
                            'cell_line': 'HEK293',
                            'data_type': 'endogenous',
                            'gene_id': row.get('external_gene_id', None)
                        }
                    }
                    
                    # Functional annotations
                    if 'te' in row.index and pd.notna(row['te']):
                        entry['annotations']['functional']['te'] = float(row['te'])
                    
                    if 'rpkm_riboseq' in row.index and pd.notna(row['rpkm_riboseq']):
                        entry['annotations']['functional']['expression_level'] = float(row['rpkm_riboseq'])
                    
                    # Structural annotations
                    if 'length' in row.index and pd.notna(row['length']):
                        entry['annotations']['structural']['length'] = int(row['length'])
                    else:
                        entry['annotations']['structural']['length'] = len(sequence)
                    
                    entry['annotations']['structural']['gc_content'] = self.calculate_gc_content(sequence)
                    
                    # Try to find structure
                    if sequence in all_structures:
                        mfe, secondary_structure = all_structures[sequence]
                        entry['annotations']['structural']['mfe'] = mfe
                        entry['annotations']['structural']['secondary_structure'] = secondary_structure
                    
                    entries.append(entry)
                    
            except Exception as e:
                logger.error(f"Error processing HEK data: {e}")
        
        # Process Muscle data
        if muscle_file.exists():
            try:
                df = pd.read_csv(muscle_file)
                logger.info(f"Processing Muscle data: {len(df)} sequences")
                
                for idx, row in df.iterrows():
                    sequence = row.get('utr', None)
                    if pd.isna(sequence) or sequence is None:
                        continue
                    
                    sequence = str(sequence).replace('<pad>', '')
                    
                    entry = {
                        'sequence': sequence,
                        'annotations': {
                            'functional': {},
                            'structural': {},
                            'regulatory': {}
                        },
                        'metadata': {
                            'source': 'Muscle_' + str(idx),
                            'cell_line': 'Muscle',
                            'data_type': 'endogenous'
                        }
                    }
                    
                    # Add available annotations
                    entry['annotations']['structural']['length'] = len(sequence)
                    entry['annotations']['structural']['gc_content'] = self.calculate_gc_content(sequence)
                    
                    # Try to find structure
                    if sequence in all_structures:
                        mfe, secondary_structure = all_structures[sequence]
                        entry['annotations']['structural']['mfe'] = mfe
                        entry['annotations']['structural']['secondary_structure'] = secondary_structure
                    
                    entries.append(entry)
                    
            except Exception as e:
                logger.error(f"Error processing Muscle data: {e}")
        
        logger.info(f"Processed {len(entries)} entries from TE dataset")
        return entries
    
    def process_experimental_dataset(self) -> List[Dict]:
        """Process experimental data with labels."""
        entries = []
        
        exp_file = self.experimental_dir / "Experimental_data_revised_label.csv"
        
        if not exp_file.exists():
            logger.warning(f"Experimental file not found: {exp_file}")
            return entries
        
        try:
            df = pd.read_csv(exp_file)
            logger.info(f"Processing experimental data: {len(df)} sequences")
            
            for idx, row in df.iterrows():
                sequence = row.get('utr_originial_varylength', None)
                if pd.isna(sequence) or sequence is None:
                    continue
                
                sequence = str(sequence).replace('<pad>', '')
                
                entry = {
                    'sequence': sequence,
                    'annotations': {
                        'functional': {},
                        'structural': {},
                        'regulatory': {}
                    },
                    'metadata': {
                        'source': row.get('rvac_ID', 'unknown'),
                        'cell_line': 'experimental',
                        'data_type': 'experimental',
                        'label': float(row['label']) if 'label' in row.index and pd.notna(row['label']) else None
                    }
                }
                
                # Structural annotations
                if 'Length' in row.index and pd.notna(row['Length']):
                    entry['annotations']['structural']['length'] = int(row['Length'])
                else:
                    entry['annotations']['structural']['length'] = len(sequence)
                
                entry['annotations']['structural']['gc_content'] = self.calculate_gc_content(sequence)
                
                entries.append(entry)
                
        except Exception as e:
            logger.error(f"Error processing experimental data: {e}")
        
        logger.info(f"Processed {len(entries)} entries from experimental dataset")
        return entries
    
    def save_chunks(self, entries: List[Dict], dataset_name: str):
        """Save entries in chunks."""
        total_chunks = (len(entries) + self.chunk_size - 1) // self.chunk_size
        
        logger.info(f"Saving {len(entries)} entries in {total_chunks} chunks of size {self.chunk_size}")
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(entries))
            chunk_entries = entries[start_idx:end_idx]
            
            # Save chunk
            chunk_file = self.output_dir / f"{dataset_name}_chunk_{chunk_idx:04d}.json"
            with open(chunk_file, 'w') as f:
                json.dump(chunk_entries, f, indent=2)
            
            logger.info(f"Saved chunk {chunk_idx + 1}/{total_chunks}: {chunk_file.name} ({len(chunk_entries)} entries)")
    
    def save_metadata(self, dataset_stats: Dict):
        """Save dataset metadata and statistics."""
        metadata_file = self.output_dir / "dataset_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
    
    def process_all(self):
        """Process all datasets and save."""
        logger.info("=" * 80)
        logger.info("Starting mRNA dataset preprocessing")
        logger.info("=" * 80)
        
        all_entries = []
        dataset_stats = {
            'total_entries': 0,
            'datasets': {},
            'chunk_size': self.chunk_size
        }
        
        # Process MRL dataset
        logger.info("\n" + "=" * 80)
        logger.info("Processing MRL dataset")
        logger.info("=" * 80)
        mrl_entries = self.process_mrl_dataset()
        all_entries.extend(mrl_entries)
        dataset_stats['datasets']['mrl'] = len(mrl_entries)
        
        # Process TE dataset
        logger.info("\n" + "=" * 80)
        logger.info("Processing TE dataset")
        logger.info("=" * 80)
        te_entries = self.process_te_dataset()
        all_entries.extend(te_entries)
        dataset_stats['datasets']['te'] = len(te_entries)
        
        # Process experimental dataset
        logger.info("\n" + "=" * 80)
        logger.info("Processing Experimental dataset")
        logger.info("=" * 80)
        exp_entries = self.process_experimental_dataset()
        all_entries.extend(exp_entries)
        dataset_stats['datasets']['experimental'] = len(exp_entries)
        
        # Update total
        dataset_stats['total_entries'] = len(all_entries)
        dataset_stats['total_chunks'] = (len(all_entries) + self.chunk_size - 1) // self.chunk_size
        
        # Save all entries in chunks
        logger.info("\n" + "=" * 80)
        logger.info("Saving processed data")
        logger.info("=" * 80)
        self.save_chunks(all_entries, "mRNA_dataset")
        
        # Save metadata
        self.save_metadata(dataset_stats)
        
        logger.info("\n" + "=" * 80)
        logger.info("Processing complete!")
        logger.info(f"Total entries: {dataset_stats['total_entries']}")
        logger.info(f"Total chunks: {dataset_stats['total_chunks']}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 80)


def main():
    """Main function to run preprocessing."""
    preprocessor = mRNADatasetPreprocessor(
        data_dir="/home/pan/Experiments/EXPs/2025_10_FM_explainability/DATA/UTR_LM",
        output_dir="/home/pan/Experiments/EXPs/2025_10_FM_explainability/DATA/transfered_dataset",
        chunk_size=5000
    )
    
    preprocessor.process_all()


if __name__ == "__main__":
    main()
