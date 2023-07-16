def define_cat(dataset, col):
    if dataset[col].dtype == "O" or dataset[col].fillna(0).unique().sum() == 1:
        return True
    else:
        return False
