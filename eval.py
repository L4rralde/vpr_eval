import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse


from utils.validation import get_validation_recalls
# Dataloader
from dataloaders.NordlandDataset import NordlandDataset
#from dataloaders.MapillaryDataset import MSLS
#from dataloaders.MapillaryTestDataset import MSLSTest
from dataloaders.PittsburghDataset import PittsburghDataset
from dataloaders.SPEDDataset import SPEDDataset
import vpr.models as vpr_models
from vpr import make_compose_transform

#VAL_DATASETS = ['MSLS', 'MSLS_Test', 'pitts30k_test', 'pitts250k_test', 'Nordland', 'SPED']
#VAL_DATASETS = ['pitts30k_test', 'pitts250k_test', 'Nordland', 'SPED'] #ALl of these are available in my machine
#VAL_DATASETS = ['pitts30k_test', 'Nordland', 'SPED']
VAL_DATASETS = ['SPED']


def get_val_dataset(dataset_name, transform):
    dataset_name = dataset_name.lower()
    
    if 'nordland' in dataset_name:    
        ds = NordlandDataset(input_transform=transform)

    elif 'msls_test' in dataset_name:
        ds = MSLSTest(input_transform=transform)

    elif 'msls' in dataset_name:
        ds = MSLS(input_transform=transform)

    elif 'pitts' in dataset_name:
        ds = PittsburghDataset(which_ds=dataset_name, input_transform=transform)

    elif 'sped' in dataset_name:
        ds = SPEDDataset(input_transform=transform)
    else:
        raise ValueError
    
    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth

def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for batch in tqdm(dataloader, 'Calculating descritptors...'):
                imgs, labels = batch
                output = model(imgs.to(device)).cpu()
                descriptors.append(output)

    return torch.cat(descriptors)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Datasets parameters
    parser.add_argument(
        '--val_datasets',
        nargs='+',
        default=VAL_DATASETS,
        help='Validation datasets to use',
        choices=VAL_DATASETS,
    )
    parser.add_argument('--image_size', required=True, help='Image size (int)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--mean_std',  type=str, default="imagenet", help='Mean Std')
    parser.add_argument('--crop', nargs='?', const=True, default=False, help='central crop')

    args = parser.parse_args()

    # Parse image size
    if args.image_size:
        args.image_size = int(args.image_size)

    return args


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    args = parse_args()
    print(args)
    model = vpr_models.DinoV3.ViTBase().eval()
    model = model.to('cuda')

    input_transform = make_compose_transform(args.image_size, args.mean_std, args.crop)

    for val_name in args.val_datasets:
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_name, input_transform)
        val_loader = DataLoader(val_dataset, num_workers=8, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        print(f'Evaluating on {val_name}')
        descriptors = get_descriptors(model, val_loader, 'cuda')
        
        print(f'Descriptor dimension {descriptors.shape[1]}')
        r_list = descriptors[ : num_references]
        q_list = descriptors[num_references : ]

        print('total_size', descriptors.shape[0], num_queries + num_references)


        preds = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10, 15, 20, 25],
            gt=ground_truth,
            print_results=True,
            dataset_name=val_name,
            faiss_gpu=False,
        )

        del descriptors
        print('========> DONE!\n\n')
