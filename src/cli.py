#!/usr/bin/env python3

import argparse
import os
import sys
import pandas as pd
import numpy as np
import pickle
import re
from collections import Counter
import torch
# import lightning as L  # Removed - converting to vanilla PyTorch
from transformers import AutoTokenizer, AutoConfig
from konlpy.tag import Kkma
from tqdm import tqdm

from .sema import VOC_TopicLabeler, VOC_DataModule

class SemaInference:
    def __init__(self, model_path=None, checkpoint_path=None):
        self.model_name = model_path or 'team-lucid/deberta-v3-xlarge-korean'
        self.checkpoint_path = checkpoint_path or f"{self.model_name}_20ep_full_mar17_dropna.ckpt"
        self.max_len = 256
        self.batch_size = 12
        self.opt_thresh = 0.5
        
        self._load_components()
        
    def _load_components(self):
        print("Loading model components...")
        
        try:
            self.config = AutoConfig.from_pretrained(self.model_name, output_hidden_states=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            with open('data2.pkl', 'rb') as f:
                self.mlb = pickle.load(f)
            
            self.label_columns = self.mlb.classes_[:]
            
            with open('voc_etc.pkl', 'rb') as f:
                self.voc_etc = pickle.load(f)
                
            with open('keyword_doc.pkl', 'rb') as f:
                self.keyword = pickle.load(f)
            
            self.model = VOC_TopicLabeler.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                n_classes=len(self.label_columns),
                model=self.model_name
            )
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
            
            self.kkma = Kkma()
            
            print(f"✓ Model loaded successfully with {len(self.label_columns)} classes")
            
        except Exception as e:
            print(f"✗ Error loading model components: {e}")
            sys.exit(1)
    
    def _remove_non_english_korean(self, string):
        pattern = re.compile(r'[^a-zA-Z0-9\uac00-\ud7a3\s]', flags=re.UNICODE)
        return pattern.sub('', string)
    
    def _strip_emoji(self, text):
        RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        return RE_EMOJI.sub(r'', text)
    
    def _findall_vec(self, key, voc):
        try:
            return re.findall(key, voc)[0]
        except:
            return ''
    
    def _findall_vec2(self, df):
        return self._findall_vec(df['keyword'], df['VOC'])
    
    def _filter_voc_data(self, voc_data):
        print("Filtering VOC data...")
        
        voc_data = voc_data[voc_data['VOC'] != 'nan']
        voc_data['VOC'] = voc_data['VOC'].apply(self._remove_non_english_korean)
        voc_data['VOC'] = voc_data['VOC'].apply(self._strip_emoji)
        voc_data['VOC'] = voc_data['VOC'].replace(r'\s+', ' ', regex=True)
        
        # Apply filters
        filt0 = (voc_data['VOC'].str.strip().str.len() < 4).astype(int)
        filt1 = voc_data['VOC'].apply(lambda x: bool(re.match(r'^[_\W]+$', str(x).replace(' ', '')))).astype(int)
        filt2 = voc_data['VOC'].apply(lambda x: bool(re.match(r'[\d/-]+$', str(x).replace(' ', '')))).astype(int)
        filt3 = (voc_data.VOC.str.replace(' ', '').str.split('').apply(set).str.len() == 2)
        
        voc_data = voc_data[(filt0 + filt1 + filt2 + filt3) == 0]
        
        # Additional filtering with morphological analysis
        voc_tok = voc_data['VOC'].progress_apply(lambda x: Counter(self.kkma.morphs(x)))
        filt4 = voc_tok.isin(self.voc_etc).astype(int)
        voc_data = voc_data[~filt4.astype(bool)].reset_index()
        
        print(f"✓ Filtered to {len(voc_data)} valid entries")
        return voc_data
    
    def process_file(self, input_path, output_path=None):
        if not os.path.exists(input_path):
            print(f"✗ Input file not found: {input_path}")
            return False
            
        print(f"Processing: {input_path}")
        
        try:
            # Read input file
            voc_testset = pd.read_excel(input_path, dtype=str)
            
            # Process VOC columns
            voc = pd.concat([voc_testset.VOC1, voc_testset.VOC2]).sort_index().values
            voc_testset = pd.concat([voc_testset]*2).sort_index().iloc[:, 1:-2]
            voc_testset['VOC'] = voc
            voc_testset = voc_testset.dropna(subset='VOC')
            voc_testset.reset_index(inplace=True)
            voc_testset['label'] = pd.DataFrame(np.zeros((len(self.label_columns), voc_testset.shape[0])).T).astype(int).apply(list, axis=1)
            
            # Filter data
            voc_testset = self._filter_voc_data(voc_testset)
            
            if len(voc_testset) == 0:
                print("✗ No valid data remaining after filtering")
                return False
            
            # Setup data module
            print("Setting up inference...")
            data_module = VOC_DataModule(voc_testset, voc_testset, self.tokenizer, 
                                       batch_size=self.batch_size, max_token_len=self.max_len)
            data_module.setup()
            
            # Run inference
            print("Running inference...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            predictions = self.model.predict(data_module.predict_dataloader(), device=device)
            sema_predictions = np.vstack(predictions)
            pred_labels = (sema_predictions > self.opt_thresh).astype(int)
            
            # Process results
            voc_testset['pred'] = pd.Series(self.mlb.inverse_transform(pred_labels)).apply(list)
            voc_testset = voc_testset.explode('pred', ignore_index=True)
            
            del voc_testset['label']
            
            # Extract keywords
            print("Extracting keywords...")
            voc_testset['topic'] = voc_testset.pred.str.split('_').str[0]
            voc_testset['sentiment'] = voc_testset.pred.str.split('_').str[1]
            voc_testset['topic'] = voc_testset['topic'].fillna('기타')
            voc_testset['keyword'] = self.keyword.loc[voc_testset.topic].values
            voc_testset['keyword'] = voc_testset.apply(self._findall_vec2, axis=1)
            
            # Save output
            if output_path is None:
                output_path = input_path.replace('.xlsx', '_output.xlsx')
            
            voc_testset.to_excel(output_path, index=False)
            print(f"✓ Results saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error processing file: {e}")
            return False
    
    def _filter_unprocessed_files(self, input_files, input_dir, output_dir, force=False):
        """Filter out files that have already been processed"""
        if force or not output_dir or not os.path.exists(output_dir):
            return input_files
        
        # Get existing output files
        existing_output_files = set()
        for file in os.listdir(output_dir):
            if file.endswith('_output.xlsx'):
                # Extract original filename by removing '_output.xlsx' suffix
                original_name = file.replace('_output.xlsx', '.xlsx')
                existing_output_files.add(original_name)
        
        # Filter input files to only include unprocessed ones
        unprocessed_files = []
        already_processed = []
        
        for file in input_files:
            if file in existing_output_files:
                already_processed.append(file)
            else:
                unprocessed_files.append(file)
        
        if already_processed:
            print(f"⚠ Skipping {len(already_processed)} already processed files:")
            for file in already_processed:
                print(f"  - {file}")
        
        return unprocessed_files
    
    def process_directory(self, input_dir, output_dir=None, force=False):
        if not os.path.exists(input_dir):
            print(f"✗ Input directory not found: {input_dir}")
            return
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Find Excel files
        input_files = []
        for file in os.listdir(input_dir):
            if file.endswith('.xlsx') and not file.startswith('~'):
                input_files.append(file)
        
        if not input_files:
            print("✗ No Excel files found in input directory")
            return
        
        print(f"Found {len(input_files)} total Excel files")
        
        # Filter out already processed files
        unprocessed_files = self._filter_unprocessed_files(input_files, input_dir, output_dir, force=force)
        
        if not unprocessed_files:
            if force:
                print("✗ No files to process")
            else:
                print("✓ All files have already been processed")
            return
        
        print(f"Processing {len(unprocessed_files)} new files...")
        
        success_count = 0
        for file in unprocessed_files:
            input_path = os.path.join(input_dir, file)
            if output_dir:
                output_path = os.path.join(output_dir, file.replace('.xlsx', '_output.xlsx'))
            else:
                output_path = None
            
            print(f"\nProcessing: {file}")
            if self.process_file(input_path, output_path):
                success_count += 1
        
        print(f"\n✓ Successfully processed {success_count}/{len(unprocessed_files)} new files")
        print(f"Total files in directory: {len(input_files)}, Already processed: {len(input_files) - len(unprocessed_files)}, New: {len(unprocessed_files)}")

def main():
    parser = argparse.ArgumentParser(description='SEMA ML Model CLI - Voice of Customer Analysis')
    parser.add_argument('input', help='Input Excel file or directory path')
    parser.add_argument('-o', '--output', help='Output file or directory path')
    parser.add_argument('-m', '--model', help='Model name or path', default='team-lucid/deberta-v3-xlarge-korean')
    parser.add_argument('-c', '--checkpoint', help='Model checkpoint path')
    parser.add_argument('-b', '--batch-size', type=int, default=12, help='Batch size for inference')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--max-length', type=int, default=256, help='Maximum token length')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of all files, even if output exists')
    
    args = parser.parse_args()
    
    print("SEMA ML Model CLI")
    print("=" * 50)
    
    try:
        # Initialize inference engine
        inferencer = SemaInference(model_path=args.model, checkpoint_path=args.checkpoint)
        inferencer.batch_size = args.batch_size
        inferencer.opt_thresh = args.threshold
        inferencer.max_len = args.max_length
        
        # Process input
        if os.path.isfile(args.input):
            inferencer.process_file(args.input, args.output)
        elif os.path.isdir(args.input):
            inferencer.process_directory(args.input, args.output, force=args.force)
        else:
            print(f"✗ Invalid input path: {args.input}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n✗ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()