import os
import pandas as pd
import tqdm


def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0,
                                     num_parts_to_read: int = 2, columns=None, verbose=False) -> pd.DataFrame:
    """
    читает num_parts_to_read партиций, преобразует их к pd.DataFrame и возвращает
    :param path_to_dataset: путь до директории с партициями
    :param start_from: номер партиции, с которой начать чтение
    :param num_parts_to_read: количество партиций, которые требуется прочитать
    :param columns: список колонок, которые нужно прочитать из партиции
    :return: pd.DataFrame
    """

    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset) 
                              if filename.startswith('part')])
    
    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    for chunk_path in tqdm.tqdm_notebook(chunks, desc="Reading dataset with pandas"):
        chunk = pd.read_parquet(chunk_path,columns=columns)
        res.append(chunk)
    return pd.concat(res).reset_index(drop=True)

def df_memory_optimize(df, embedding_projections):
    for col_name in df.columns:
        card = embedding_projections.get(col_name, None)
        if card is None:
            continue
        else:
            card = card[0]
        if card <= 2**8:
            new_type = 'uint8'
        elif card <= 2**16:
            new_type = 'uint16'
        elif card <= 2**32:
            new_type = 'uint32'
        else:
            new_type = 'uint64'
        df[col_name] = df[col_name].astype(new_type)
            
        
    
