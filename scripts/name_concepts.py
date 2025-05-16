import argparse
import os

import torch
import torch.nn.functional as F
from open_clip import get_tokenizer

from xclip.open_clip import OpenCLIP
from xclip.sparse_autoencoder import DiscoverThenName


def save_activations(args: argparse.Namespace) -> None:
    vocab_filename = os.path.split(args.vocab_file)[1]
    embedding_name = f'embeddings_{os.path.splitext(vocab_filename)[0]}.pth'
    if os.path.exists(os.path.join(args.out_dir, 'concepts', embedding_name)):
        return

    # load model and tokenizer
    tokenizer = get_tokenizer(args.img_enc_name)
    clip, *_ = OpenCLIP.from_pretrained(args.img_enc_name, ckpt_path=args.ckpt_path, precision='fp32')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip.to(device)
    clip.eval()

    with open(args.vocab_file, 'r') as f:
        concept_names = [line.strip() for line in f.readlines()]
    text = tokenizer(concept_names)

    batch_size = 256  # You can adjust the batch size as needed
    txt_feat_list = []
    with torch.inference_mode():
        for i in range(0, len(text), batch_size):
            batch_text = text[i : i + batch_size].to(device)
            batch_txt_feat = F.normalize(clip.encode_text(batch_text)).cpu()
            txt_feat_list.append(batch_txt_feat)

    txt_feat = torch.cat(txt_feat_list, dim=0)

    os.makedirs(os.path.join(args.out_dir, 'concepts'), exist_ok=True)
    torch.save(txt_feat, os.path.join(args.out_dir, 'concepts', embedding_name))


def name_concepts(args: argparse.Namespace) -> None:
    args.config_name = 'RN50'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_filename = os.path.split(args.vocab_file)[1]
    embedding_name = f'embeddings_{os.path.splitext(vocab_filename)[0]}.pth'
    embeddings_path = os.path.join(args.out_dir, 'concepts', embedding_name)
    vocab_txt_path = args.vocab_file
    method_obj = DiscoverThenName(
        args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_sae_from_args=True
    )

    concept_name_similarity_matrix = method_obj.get_concept_name_similarity_matrix()[0]
    all_concept_names = method_obj.vocab_txt_all[0]
    top_concept_idxs = concept_name_similarity_matrix.argmax(axis=0)

    with open(os.path.join(args.out_dir, 'concepts', 'concept_names.csv'), 'w') as f:
        for idx in range(top_concept_idxs.shape[0]):
            name = all_concept_names[top_concept_idxs[idx]]
            print(f'{idx},{name}')
            f.write(f'{idx},{name}\n')


def main(args: argparse.Namespace) -> None:
    save_activations(args)
    name_concepts(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_enc_name',
        type=str,
        default='RN50',
        help='Name of the clip image encoder',
        choices=['RN50'],
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='Path to the directory where the checkpoints and features should be saved',
    )
    parser.add_argument('--vocab_file', type=str, required=True, help='File containing the vocabulary')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint to load the model from')

    # SAE related
    parser.add_argument('--input_dim', type=int, default=1024, help='dimension of the input to the SAE')
    parser.add_argument('--expansion_factor', type=int, default=4)
    parser.add_argument(
        '--hook_points', nargs='*', help='Name of the model hook points to get the activations from', default=['out']
    )

    args = parser.parse_args()
    main(args)
