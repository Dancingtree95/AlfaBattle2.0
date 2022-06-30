import numpy as np
import pickle
import torch

transaction_features = ['currency', 'operation_kind', 'card_type', 'operation_type',
                        'operation_type_group', 'ecommerce_flag', 'payment_system',
                        'income_flag', 'mcc', 'country', 'city', 'mcc_category',
                        'day_of_week', 'hour', 'weekofyear', 'amnt', 'days_before', 'hour_diff']


def batches_generator(list_of_paths, batch_size=32, shuffle=False, is_infinite=False,
                      verbose=False, device=None, output_format='torch', is_train=True):
    while True:
        if shuffle:
            np.random.shuffle(list_of_paths)

        for path in list_of_paths:
            if verbose:
                print(f'reading {path}')

            with open(path, 'rb') as f:
                data = pickle.load(f)
            padded_sequences, targets, products = data['padded_sequences'], data['targets'], data[
                'products']
            app_ids = data['app_id']
            indices = np.arange(len(products))

            if shuffle:
                np.random.shuffle(indices)
                padded_sequences = padded_sequences[indices]
                targets = targets[indices]
                products = products[indices]
                app_ids = app_ids[indices]

            for idx in range(len(products)):
                bucket, product = padded_sequences[idx], products[idx]
                app_id = app_ids[idx]
                
                if is_train:
                    target = targets[idx]
                
                indices = np.arange(len(product))
                if shuffle:
                    np.random.shuffle(indices)
                    bucket = bucket[indices]
                    if is_train:
                        target = target[indices]
                    product = product[indices]
                    app_id = app_id[indices]
                for jdx in range(0, len(bucket), batch_size):
                    batch_sequences = bucket[jdx: jdx + batch_size]
                    global bs
                    bs = batch_sequences
                    if is_train:
                        batch_targets = target[jdx: jdx + batch_size]
                    
                    batch_products = product[jdx: jdx + batch_size]
                    batch_app_ids = app_id[jdx: jdx + batch_size]
                    
                    if output_format == 'tf':
                        batch_sequences = [batch_sequences[:, i] for i in
                                           range(len(transaction_features))]
                        
                        # append product as input to tf model
                        batch_sequences.append(batch_products)
                        if is_train:
                            yield batch_sequences, batch_targets
                        else:
                             yield batch_sequences, batch_app_ids
                    else:
                        batch_sequences = [torch.LongTensor(np.array([app[i] for app in batch_sequences])).to(device)
                                           for i in range(len(transaction_features))]
                        if is_train:
                            yield dict(transactions_features=batch_sequences,
                                       product=torch.LongTensor(batch_products).to(device),
                                       label=torch.LongTensor(batch_targets).to(device),
                                       app_id=batch_app_ids)
                        else:
                            yield dict(transactions_features=batch_sequences,
                                       product=torch.LongTensor(batch_products).to(device),
                                       app_id=batch_app_ids)
        if not is_infinite:
            break