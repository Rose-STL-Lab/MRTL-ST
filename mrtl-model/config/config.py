b_dims = [[3, 2], [6, 4], [9, 6], [18, 12]]
c_dims = [[3, 6], [6, 12]]

train_percent = 0.6
val_percent = 0.2

fn_train = 'train_18x12.pkl'
fn_val = 'val_18x12.pkl'
fn_test = 'test_18x12.pkl'

parent_logger_name = "mrtl"

num_workers = 6
max_epochs = 50
