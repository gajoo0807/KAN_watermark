import torch
from kan import *
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import argparse

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")



def load_calhous_dataset():
    # Load California housing dataset
    calhous = fetch_california_housing()
    data = calhous.data
    target = calhous.target

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.float32)

    # Split dataset into train and test sets
    train_data, test_data, train_target, test_target = train_test_split(data_tensor, target_tensor, test_size=0.2, random_state=42)

    # Create data loaders (optional, if you want to batch and shuffle the data)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_target), batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_target), batch_size=1, shuffle=False)

    train_inputs = torch.empty(0, 8, device=device)
    train_labels = torch.empty(0, dtype=torch.long, device=device)
    test_inputs = torch.empty(0, 8, device=device)
    test_labels = torch.empty(0, dtype=torch.long, device=device)

    # Concatenate all data into a single tensor on the specified device
    for data, labels in tqdm(train_loader):
        train_inputs = torch.cat((train_inputs, data.to(device)), dim=0)
        train_labels = torch.cat((train_labels, labels.to(device)), dim=0)

    for data, labels in tqdm(test_loader):
        test_inputs = torch.cat((test_inputs, data.to(device)), dim=0)
        test_labels = torch.cat((test_labels, labels.to(device)), dim=0)

    dataset = {}
    dataset['train_input'] = train_inputs
    dataset['test_input'] = test_inputs
    dataset['train_label'] = train_labels.reshape(-1, 1)
    dataset['test_label'] = test_labels.reshape(-1, 1)

    return dataset


def verification_model(ver_model_name, private_key, l, i, j, dataset):
    '''
    驗證指定Model與Private key在指定dataset上的差異
    '''
    
    model = KAN(width=[8, 3, 1], grid=3, k=3, seed=5, device=device)
    model.load_ckpt(f'{ver_model_name}.pth')
    private_key_path = f'private_key/{private_key}'

    compute_mse = model.watermark_verification(private_key_path, l, i, j, dataset)
    return compute_mse


if __name__ == '__main__':

    # 創建 ArgumentParser 對象
    parser = argparse.ArgumentParser(description='Verify the model ownership using fingerprinting.')
    # 添加參數到解析器中
    parser.add_argument('--ver_model', type=str, required=True, help='Name of the model to verify')
    parser.add_argument('--private_key', type=str, required=True, help='Path to the private key file')

    # 解析命令行參數
    args = parser.parse_args()
    mse_list = []
    calhous_dataset = load_calhous_dataset()
    for i in range(100):
        mse_list.append(verification_model(args.ver_model, args.private_key, 0, 0, 0, calhous_dataset))
    print(f"My Key：{args.private_key}, Verification Model：{args.ver_model}, MSE：{sum(mse_list)/len(mse_list)}")