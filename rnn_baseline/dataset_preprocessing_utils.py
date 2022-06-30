from typing import Dict

import numpy as np
import pandas as pd
import pickle
from tqdm.notebook import tqdm

features = ['currency', 'operation_kind', 'card_type', 'operation_type', 'operation_type_group', 'ecommerce_flag',
            'payment_system', 'income_flag', 'mcc', 'country', 'city', 'mcc_category', 'day_of_week',
            'hour', 'weekofyear', 'amnt', 'days_before', 'hour_diff']

embedding_projection = {'currency': (11, 6),
                        'operation_kind': (7, 5),
                        'card_type': (175, 29),
                        'operation_type': (22, 9),
                        'operation_type_group': (4, 3),
                        'ecommerce_flag': (3, 3),
                        'payment_system': (7, 5),
                        'income_flag': (3, 3),
                        'mcc': (108, 22),
                        'country': (24, 9),
                        'city': (163, 28),
                        'mcc_category': (28, 10),
                        'day_of_week': (7, 5),
                        'hour': (24, 9),
                        'weekofyear': (53, 15),
                        'amnt': (10, 6),
                        'days_before': (23, 9),
                        'hour_diff': (10, 6)}


def pad_sequence(array, max_len) -> np.array:
    add_zeros = max_len - len(array[0])
    paded = np.empty(len(array), object)
    paded[:] = [np.concatenate([x, np.array([0]*add_zeros, dtype = x.dtype)]) for x in array]
    return paded


def truncate(x, num_last_transactions=750):
    return [x[col].values[-num_last_transactions:] for col in x.columns]


def transform_transactions_to_sequences(transactions_frame: pd.DataFrame,
                                        num_last_transactions=750) -> pd.DataFrame:
    return transactions_frame \
        .sort_values(['app_id', 'transaction_number']) \
        .groupby(['app_id'])[features] \
        .apply(lambda x: truncate(x, num_last_transactions=num_last_transactions)) \
        .reset_index().rename(columns={0: 'sequences'})



def create_padded_buckets(frame_of_sequences: pd.DataFrame, bucket_info: Dict[int, int],
                          save_to_file_path=None, has_target=True):
    
    frame_of_sequences['bucket_idx'] = frame_of_sequences.sequence_length.map(bucket_info)
    padded_seq = []
    targets = []
    app_ids = []
    products = []

    for size, bucket in tqdm(frame_of_sequences.groupby('bucket_idx'), desc='Extracting buckets'):
        padded_sequences = bucket.sequences.apply(lambda x: pad_sequence(x, size)).values
        #padded_sequences = np.array([np.array(x) for x in padded_sequences])
        padded_seq.append(padded_sequences)

        if has_target:
            targets.append(bucket.flag.values)

        app_ids.append(bucket.app_id.values)
        products.append(bucket['product'].values)

    frame_of_sequences.drop(columns=['bucket_idx'], inplace=True)

    dict_result = {
        'padded_sequences': np.array(padded_seq),
        'targets': np.array(targets) if targets else [],
        'app_id': np.array(app_ids),
        'products': np.array(products),
    }

    if save_to_file_path:
        with open(save_to_file_path, 'wb') as f:
            pickle.dump(dict_result, f)
    return dict_result


def create_buckets_from_transactions(path_to_dataset, save_to_path, frame_with_ids = None, 
                                     num_parts_to_preprocess_at_once: int = 1, 
                                     num_parts_total=50, has_target=False):
    block = 0
    for step in tqdm(range(0, num_parts_total, num_parts_to_preprocess_at_once), 
                                   desc="Transforming transactions data"):
        transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once, 
                                                             verbose=True)
        
        for dense_col in ['amnt', 'days_before', 'hour_diff']:
            transactions_frame[dense_col] = np.digitize(transactions_frame[dense_col], bins=dense_features_buckets[dense_col])
         df_memory_optimize(transactions_frame, embedding_projections)
            
        seq = transform_transactions_to_sequences(transactions_frame)
        seq['sequence_length'] = seq.sequences.apply(lambda x: len(x[1]))
        
        if frame_with_ids is not None:
            seq = seq.merge(frame_with_ids, on='app_id')

        block_as_str = str(block)
        if len(block_as_str) == 1:
            block_as_str = '00' + block_as_str
        else:
            block_as_str = '0' + block_as_str
            
        processed_fragment =  create_padded_buckets(seq, mapping_seq_len_to_padded_len, has_target=has_target, 
                                                    save_to_file_path=os.path.join(save_to_path, 
                                                                                   f'processed_chunk_{block_as_str}.pkl'))
        del transactions_frame
        gc.collect()
        block += 1
