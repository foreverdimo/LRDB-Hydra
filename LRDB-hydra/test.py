import logging
import torch
import hydra
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from srcs.utils import instantiate
from torch.utils.data import DataLoader

@hydra.main(config_path='conf', config_name='test')
def main(config):
    checkpoint = torch.load(config.checkpoint)

    loaded_config = OmegaConf.create(checkpoint['config'])

    # setup data_loader instances
    dataset = instantiate(config.dataset)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # restore network architecture
    model = instantiate(loaded_config.arch)

    # load trained weights
    state_dict = checkpoint['state_dict']
    if loaded_config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    result = []
    with torch.no_grad():
        for i, (dst, ref, dst_name) in enumerate(tqdm(data_loader)):
            dst, ref = dst.to(device), ref.to(device)
            output = model((dst, ref))
            result.append([dst_name[0], output.item()])

    d = pd.DataFrame(result)
    d.to_csv('./outputs/train/2021-01-29/lr0.0001/output.txt', header=None, index=False)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
